import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.trainer_utils import get_batch


@torch.no_grad()
def compute_perplexity(model, bin_path, batch_size, context_length, device, n_batches=50):
    model.eval()
    total_loss = 0.0

    for _ in range(n_batches):
        x, y = get_batch(bin_path, batch_size, context_length, device)
        _, loss = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


if __name__ == "__main__":
    import argparse
    from model.config import ModelConfig
    from model.architecture import GPTModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain_final.pt")
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--val_bin", type=str, default="data/processed/val.bin")
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig.from_yaml(args.model_config)
    model = GPTModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ppl, loss = compute_perplexity(model, args.val_bin, batch_size=8,
                                     context_length=config.context_length,
                                     device=device, n_batches=args.n_batches)
    print(f"Val loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    print(f"(Comparison: random model perplexity ≈ {config.vocab_size})")
