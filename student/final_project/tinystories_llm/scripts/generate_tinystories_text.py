import torch
import argparse
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
import os
import json
from tqdm import tqdm


def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)


def load_model(args, device):
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Load config
    config_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_args = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=train_args.get('hidden_size', 256),
            num_hidden_layers=train_args.get('num_layers', 4),
            num_attention_heads=train_args.get('num_heads', 8),
            intermediate_size=train_args.get('intermediate_size', 1024),
            hidden_dropout_prob=train_args.get('dropout', 0.1),
            attention_probs_dropout_prob=train_args.get('dropout', 0.1),
            max_position_embeddings=train_args.get('max_seq_len', 512),
            window_size=train_args.get('window_size', 256),
        )
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    # Load model
    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_one(model, tokenizer, prompt, args, device):
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)],
        dtype=torch.long
    ).to(device)

    eos_token_id = tokenizer.token2id.get('<eos>', None)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
        )

    return tokenizer.decode(output_ids[0].tolist())


def batch_generate(model, tokenizer, args, device):
    with open(args.eval_path, "r") as f:
        eval_set = json.load(f)

    results = []

    print("Generating batch outputs...")

    for sample in tqdm(eval_set):
        prompt = sample["prompt"]

        generated = generate_one(model, tokenizer, prompt, args, device)

        results.append({
            "category": sample["category"],
            "prompt": prompt,
            "expected_entities": sample.get("expected_entities", []),
            "generated": generated
        })

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved to", args.output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained TinyStories model.")

    parser.add_argument('--model_path', type=str, default='tinystories_model/best_model.pth')
    parser.add_argument('--tokenizer_path', type=str, default='bpe_tokenizer_tinystories.pkl')

    parser.add_argument('--prompt', type=str, default=None)

    parser.add_argument('--eval_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='base_outputs.json')

    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model, tokenizer = load_model(args, device)

    if args.eval_path is not None:
        batch_generate(model, tokenizer, args, device)
    else:
        if args.prompt is None:
            print("Please provide --prompt or --eval_path")
            return

        output_text = generate_one(model, tokenizer, args.prompt, args, device)
        print("Prompt:", args.prompt)
        print("Generated:", output_text)


if __name__ == '__main__':
    main()
