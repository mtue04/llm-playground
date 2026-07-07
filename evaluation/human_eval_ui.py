"""Human evaluation UI for comparing model outputs side-by-side."""
import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.generate import load_model_for_inference, generate
from tokenizer import LLMTokenizer


def run_human_eval(model, tokenizer, prompts, device, temperature=0.7,
                   top_k=50, top_p=0.9, max_new_tokens=150):
    """Interactive human evaluation: show model responses and collect ratings."""
    results = []

    print("=" * 60)
    print("HUMAN EVALUATION")
    print("Rate each response on a scale of 1-5:")
    print("  1 = Terrible  2 = Poor  3 = OK  4 = Good  5 = Excellent")
    print("Type 'q' to quit early.")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt}\n")

        _, response = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, device=device,
        )

        print(f"Response: {response.strip()}\n")

        while True:
            rating_input = input("Your rating (1-5, or 'q' to quit): ").strip()
            if rating_input.lower() == 'q':
                print("Evaluation ended early.")
                return results
            try:
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number 1-5.")

        comment = input("Comment (optional, press Enter to skip): ").strip()

        results.append({
            "prompt": prompt,
            "response": response.strip(),
            "rating": rating,
            "comment": comment,
        })

    # Summary
    avg_rating = sum(r["rating"] for r in results) / max(len(results), 1)
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"Total prompts evaluated: {len(results)}")
    print(f"Average rating: {avg_rating:.2f}/5.00")
    print(f"{'=' * 60}")

    return results


SAMPLE_PROMPTS = [
    "Explain what machine learning is in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of regular exercise?",
    "Describe how a computer works to a child.",
]


if __name__ == "__main__":
    import argparse
    from model.config import ModelConfig
    from model.architecture import GPTModel

    parser = argparse.ArgumentParser(description="Human evaluation of model outputs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results as JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_for_inference(args.checkpoint, args.model_config, device)
    tokenizer = LLMTokenizer()

    results = run_human_eval(model, tokenizer, SAMPLE_PROMPTS, device)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
