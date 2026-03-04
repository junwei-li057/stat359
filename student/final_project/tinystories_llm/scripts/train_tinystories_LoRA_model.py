import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datasets import load_dataset
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from tqdm import tqdm
import time
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, TaskType

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a TinyStories model with LoRA")
    
    # Dataset arguments 
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="HuggingFace dataset name")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl", help="Path to BPE tokenizer")
    parser.add_argument("--patch_file", type=str, default=None, help="Path to JSON patch file")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    
    # Model architecture arguments 
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Size of the intermediate layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--window_size", type=int, default=256, help="Attention window size")
    
    # ADD LoRA PARAMETERS
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs='+', default=["query", "value"], help="List of module names to apply LoRA to")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (LoRA typically uses larger LR than full fine-tuning)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="tinystories_lora", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Eval steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--pilot_run", action="store_true", help="Quick pilot run")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    
    return parser.parse_args()

# === Set Device ===
def get_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA requested but not available.")
    elif device_preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            raise ValueError("MPS requested but not available.")
    else:
        return torch.device("cpu")


# === Set Seed ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Load Tokenizer ===
def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)

# === Dataset ===
class TinyStoriesDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, split="train", max_samples=None, patch_file=None):
        """
        dataset: HuggingFace dataset
        tokenizer: BPETokenizer
        max_length: 最大序列长度
        split: 'train' 或 'validation'
        max_samples: 仅用于调试，限制样本数量
        patch_file: JSON 文件，包含 prompt 和 completion
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        ds_split = dataset[split]
        if max_samples is not None:
            ds_split = ds_split.select(range(min(max_samples, len(ds_split))))

        for item in ds_split:
            text = item["text"]
            self.examples.append({
                "input_text": text,  
                "target_text": text 
            })

        # patch data
        if patch_file is not None:
            with open(patch_file, "r", encoding="utf-8") as f:
                patch_texts = json.load(f)
            for item in patch_texts:
                prompt = item.get("prompt", "")
                completion = item.get("completion", "")
                if prompt and completion:
                    self.examples.append({
                        "input_text": prompt,
                        "target_text": completion
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = self.tokenizer.encode(example["input_text"], add_special_tokens=True)
        target_ids = self.tokenizer.encode(example["target_text"], add_special_tokens=True)

        tokens = input_ids + target_ids
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_len = self.max_length - len(tokens)
            tokens += [self.tokenizer.token2id.get('<pad>', 0)] * pad_len

        return torch.tensor(tokens, dtype=torch.long)



# === Learning Rate Scheduler ===
class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(0.0, 1.0 - progress)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

# === Training and Evaluation ===
def train_and_evaluate(args):
    device = get_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)
    
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    train_dataset = TinyStoriesDataset(dataset, tokenizer, max_length=args.max_seq_len, split="train", max_samples=args.max_train_samples, patch_file=args.patch_file)
    val_dataset = TinyStoriesDataset(dataset, tokenizer, max_length=args.max_seq_len, split="validation", max_samples=args.max_eval_samples)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_seq_len,
        window_size=args.window_size,
    )
    model = TinyStoriesForCausalLM(config).to(device)

    # 2. LOAD THE BASE MODEL
    base_model_path = "tinystories_model/best_model.pth"
    if os.path.exists(base_model_path):
        print(f"===> Loading pre-trained base model from {base_model_path}...")
        checkpoint = torch.load(base_model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("===> Base model loaded successfully!")
    else:
        print(f"!!! Warning: {base_model_path} not found. Training from scratch with LoRA is NOT recommended.")

    # 3. Insert LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps, total_steps)
    
    pad_token_id = tokenizer.token2id.get('<pad>', 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)


    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    train_losses = []
    if args.resume_from_checkpoint is not None and os.path.isfile(args.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and 'current_step' in checkpoint['scheduler_state_dict']:
            scheduler.current_step = checkpoint['scheduler_state_dict']['current_step']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")
    
    # AMP scaler (only if using CUDA and AMP is enabled)
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(device="cuda")
    else:
        autocast = None
        scaler = None
    # Set AMP flag on model for evaluation
    if use_amp:
        model.use_amp = True
    else:
        model.use_amp = False
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Get inputs and targets (shift right for causal language modeling)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass with AMP if enabled
            if use_amp:
                with autocast(device_type="cuda"):
                    outputs = model(input_ids=inputs)
                    logits = outputs["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / args.gradient_accumulation_steps
            else:
                outputs = model(input_ids=inputs)
                logits = outputs["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights if gradient accumulation steps reached
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update weights
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                
                # Log training loss
                if global_step % args.logging_steps == 0:
                    train_losses.append(loss.item() * args.gradient_accumulation_steps)
                    avg_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    # Log to TensorBoard
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    writer.add_scalar('Perplexity/train', np.exp(avg_loss), global_step)
                
                # Evaluate
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, criterion, device)
                    val_ppl = np.exp(val_loss)
                    print(f"Step {global_step}: Validation loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
                    writer.add_scalar('Loss/val', val_loss, global_step)
                    writer.add_scalar('Perplexity/val', val_ppl, global_step)
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_path = os.path.join(args.output_dir, "best_model.pth")
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved to {model_path}")
                    # Back to training mode
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': {
                            'current_step': scheduler.current_step,
                        },
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # Update epoch loss
            epoch_loss += loss.item() * args.gradient_accumulation_steps
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_ppl = np.exp(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Train Loss: {avg_epoch_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch+1)
        writer.add_scalar('Perplexity/train_epoch', train_ppl, epoch+1)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        val_ppl = np.exp(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        writer.add_scalar('Loss/val_epoch', val_loss, epoch+1)
        writer.add_scalar('Perplexity/val_epoch', val_ppl, epoch+1)
        model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()
    
    return model, device

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            if hasattr(model, 'use_amp') and model.use_amp:
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    outputs = model(input_ids=inputs)
                    logits = outputs["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                outputs = model(input_ids=inputs)
                logits = outputs["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_text(model, tokenizer, prompt, device, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
   
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)]).to(device)
    
    with torch.no_grad():
        target_model = model.base_model if hasattr(model, "base_model") else model
        
        output_ids = target_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    return tokenizer.decode(output_ids[0].tolist())

if __name__ == '__main__':
    args = parse_args()
    # Handle pilot run option
    if getattr(args, 'pilot_run', False):
        args.max_train_samples = 1000
        args.max_eval_samples = 1000
        print("[Pilot Run] Using 100 samples for training and evaluation.")
    start_time = time.time()
    model, device = train_and_evaluate(args)
    end_time = time.time()
    
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
    
    # Generate sample text
    tokenizer = load_tokenizer(args.tokenizer_path)
    prompt = "Once upon a time, there was a"
    generated_text = generate_text(model, tokenizer, prompt, device)
    
    print("\nSample generation:")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")