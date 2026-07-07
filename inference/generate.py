import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.config import ModelConfig
from model.architecture import GPTModel
from tokenizer import LLMTokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8,
             top_k=None, top_p=None, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True)
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context if it exceeds context_length
        idx_cond = idx[:, -model.config.context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_logits[sorted_mask] = -float("inf")

            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.eos_id:
            break

        idx = torch.cat([idx, next_id], dim=1)

    all_ids = idx[0].tolist()
    generated_ids = all_ids[prompt_len:]
    full_text = tokenizer.decode(all_ids)
    generated_text = tokenizer.decode(generated_ids)
    return full_text, generated_text


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """Greedy decoding - always selects the highest probability token."""
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length:]
        logits, _ = model(idx_cond)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_id.item() == tokenizer.eos_id:
            break
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


def load_model_for_inference(checkpoint_path, model_config_path="configs/model_small.yaml", device="cuda"):
    config = ModelConfig.from_yaml(model_config_path)
    model = GPTModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--prompt", type=str, default="Hello,")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_for_inference(args.checkpoint, device=device)
    tokenizer = LLMTokenizer()

    full_output, generated_text = generate(model, tokenizer, args.prompt, args.max_new_tokens,
                       args.temperature, args.top_k, args.top_p, device)
    print(generated_text)