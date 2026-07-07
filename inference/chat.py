import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.generate import load_model_for_inference, generate
from tokenizer import LLMTokenizer


def chat_loop(checkpoint_path, model_config_path="configs/model_small.yaml"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint_path} (device: {device})...")

    model = load_model_for_inference(checkpoint_path, model_config_path, device)
    tokenizer = LLMTokenizer()

    print("Chatbot ready. Type 'quit' or 'exit' to exit.\n")

    conversation = ""

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        conversation += f"<|user|>{user_input}<|assistant|>"
        _, response_text = generate(
            model, tokenizer, conversation,
            max_new_tokens=150, temperature=0.7, top_k=50, top_p=0.9,
            device=device,
        )

        new_text = response_text.strip()
        print(f"Bot: {new_text}\n")

        conversation += new_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    args = parser.parse_args()
    chat_loop(args.checkpoint, args.model_config)