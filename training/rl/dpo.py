import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model.config import ModelConfig
from model.architecture import GPTModel
from tokenizer import LLMTokenizer
from training.trainer_utils import save_checkpoint, load_checkpoint


def get_sequence_logprobs(model, input_ids, attention_mask):
    """Computes sequence log-probabilities for target sequences."""
    logits, _ = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """Computes DPO loss and accuracy."""
    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp

    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, accuracy


def main(model_config_path, sft_checkpoint_path, preference_dataset, max_steps=500,
         beta=0.1, lr=1e-5, batch_size=2, context_length=512):
    model_cfg = ModelConfig.from_yaml(model_config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LLMTokenizer()

    policy_model = GPTModel(model_cfg).to(device)
    load_checkpoint(sft_checkpoint_path, policy_model, device=device)

    ref_model = copy.deepcopy(policy_model).to(device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    def tokenize_and_pad(text):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:context_length]
        mask = [1] * len(ids)
        pad_len = context_length - len(ids)
        ids = ids + [tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len
        return ids, mask

    policy_model.train()
    for step in range(max_steps):
        batch = [preference_dataset[i % len(preference_dataset)]
                  for i in range(step * batch_size, (step + 1) * batch_size)]

        chosen_ids, chosen_mask = zip(*[tokenize_and_pad(b["chosen"]) for b in batch])
        rejected_ids, rejected_mask = zip(*[tokenize_and_pad(b["rejected"]) for b in batch])

        chosen_ids = torch.tensor(chosen_ids, device=device)
        chosen_mask = torch.tensor(chosen_mask, device=device)
        rejected_ids = torch.tensor(rejected_ids, device=device)
        rejected_mask = torch.tensor(rejected_mask, device=device)

        policy_chosen_logp = get_sequence_logprobs(policy_model, chosen_ids, chosen_mask)
        policy_rejected_logp = get_sequence_logprobs(policy_model, rejected_ids, rejected_mask)
        with torch.no_grad():
            ref_chosen_logp = get_sequence_logprobs(ref_model, chosen_ids, chosen_mask)
            ref_rejected_logp = get_sequence_logprobs(ref_model, rejected_ids, rejected_mask)

        loss, accuracy = dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp, beta=beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step:4d} | dpo_loss {loss.item():.4f} | accuracy {accuracy.item():.2%}")

    save_checkpoint(policy_model, optimizer, max_steps, model_cfg, "checkpoints", "dpo_final.pt")
    print("DPO complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--sft_checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    example_dataset = [
        {"chosen": "Helpful and polite answer.", "rejected": "Rude or incorrect answer."},
    ]
    main(args.model_config, args.sft_checkpoint, example_dataset,
         max_steps=args.max_steps, beta=args.beta)
