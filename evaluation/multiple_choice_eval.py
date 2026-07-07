import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@torch.no_grad()
def score_choice(model, tokenizer, question, choice, device):
    """Computes log-probability of choice appended to question."""
    full_text = question + " " + choice
    q_ids = tokenizer.encode(question, add_bos=True)
    full_ids = tokenizer.encode(full_text, add_bos=True)

    idx = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(idx[:, :-1])
    targets = idx[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    choice_start = len(q_ids) - 1
    choice_log_prob = token_log_probs[0, choice_start:].sum().item()
    return choice_log_prob


def run_multiple_choice_eval(model, tokenizer, examples, device):
    """Runs multiple choice evaluation and returns accuracy."""
    correct = 0
    for ex in examples:
        scores = [score_choice(model, tokenizer, ex["question"], c, device) for c in ex["choices"]]
        predicted_idx = scores.index(max(scores))
        if predicted_idx == ex["answer_idx"]:
            correct += 1

    accuracy = correct / len(examples)
    return accuracy


SAMPLE_QUESTIONS = [
    {
        "question": "The capital of France is",
        "choices": ["Hanoi.", "Bangkok.", "Tokyo.", "Paris."],
        "answer_idx": 3,
    },
    {
        "question": "The sun rises in the",
        "choices": ["West.", "East.", "North.", "South."],
        "answer_idx": 1,
    },
]

if __name__ == "__main__":
    import argparse
    from model.config import ModelConfig
    from model.architecture import GPTModel
    from tokenizer import LLMTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig.from_yaml(args.model_config)
    model = GPTModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = LLMTokenizer()
    acc = run_multiple_choice_eval(model, tokenizer, SAMPLE_QUESTIONS, device)
    print(f"Accuracy on {len(SAMPLE_QUESTIONS)} sample questions: {acc:.2%}")
