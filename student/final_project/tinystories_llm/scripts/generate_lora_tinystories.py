import torch
import argparse
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
import os
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)

def load_model(args, tokenizer, device):
    """
    Load and build the model. For LoRA models, you need to first build the Base Model,
    inject the LoRA structure, and then load the weights.
    """
    # 1. Initialize base model configuration
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len
    )
    
    # 2. Create base model
    print("Building base model...")
    model = TinyStoriesForCausalLM(config).to(device)
    
    # 3. Inject LoRA structure to match weight names
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 4. Load LoRA weights
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Support loading the entire checkpoint or only the state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def generate_one(model, tokenizer, prompt, args, device):
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)],
        dtype=torch.long
    ).to(device)

    eos_token_id = tokenizer.token2id.get('<eos>', None)

    with torch.no_grad():
        # Get underlying model to bypass peft's generation_config check
        target_model = model.base_model if hasattr(model, "base_model") else model
        
        output_ids = target_model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            do_sample=True,               # Enable sampling
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.2,       # Prevent "visit visit" repetition
            eos_token_id=eos_token_id,
        )

    return tokenizer.decode(output_ids[0].tolist())

def batch_generate(model, tokenizer, args, device):
    if not os.path.exists(args.eval_path):
        print(f"Error: Eval file {args.eval_path} not found.")
        return

    with open(args.eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    print(f"Generating batch outputs for {len(eval_set)} prompts...")

    for sample in tqdm(eval_set):
        prompt = sample["prompt"]
        generated = generate_one(model, tokenizer, prompt, args, device)

        results.append({
            "category": sample.get("category", "default"),
            "prompt": prompt,
            "expected_entities": sample.get("expected_entities", []),
            "generated": generated
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Batch generation completed. Results saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained TinyStories LoRA model.")

    # Path arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to LoRA .pth file')
    parser.add_argument('--tokenizer_path', type=str, default='bpe_tokenizer_tinystories.pkl')
    parser.add_argument('--eval_path', type=str, default=None, help='Path to evaluation JSON file')
    parser.add_argument('--output_path', type=str, default='lora_outputs.json')
    parser.add_argument('--prompt', type=str, default=None)

    # Model architecture parameters
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Generation control parameters
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument("--max_seq_len", type=int, default=256, help="Must match training")

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    # Device handling
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load tokenizer first
    tokenizer = load_tokenizer(args.tokenizer_path)

    # 2. Pass tokenizer into load_model
    model = load_model(args, tokenizer, device)

    # Execute generation
    if args.eval_path is not None:
        batch_generate(model, tokenizer, args, device)
    elif args.prompt is not None:
        output_text = generate_one(model, tokenizer, args.prompt, args, device)
        print("\n" + "="*30)
        print("Prompt:", args.prompt)
        print("-" * 30)
        print("Generated:", output_text)
        print("="*30 + "\n")
    else:
        print("Please provide --prompt or --eval_path")

if __name__ == '__main__':
    main()
