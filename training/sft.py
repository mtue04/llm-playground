import argparse
import os
import sys

import torch
import yaml
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.config import ModelConfig
from model.architecture import GPTModel
from tokenizer import LLMTokenizer
from training.trainer_utils import get_lr, save_checkpoint, load_checkpoint


def build_sft_batch(examples, tokenizer, template, context_length, device):
    """Tokenize a batch of instruction/response examples and create target masks.
    Uses -1 (IGNORE_INDEX) for instructions and actual token IDs for responses.
    """
    IGNORE_INDEX = -1
    all_input_ids, all_targets = [], []

    for ex in examples:
        prompt_part = template.split("{response}")[0].format(instruction=ex["instruction"])
        full_text = template.format(instruction=ex["instruction"], response=ex["response"])

        prompt_ids = tokenizer.encode(prompt_part, add_bos=True)
        full_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)
        full_ids = full_ids[:context_length]

        # Target is shifted right by 1, masking the prompt with IGNORE_INDEX
        targets = full_ids[1:] + [IGNORE_INDEX]
        n_prompt_tokens = min(len(prompt_ids), len(targets))
        targets[:n_prompt_tokens] = [IGNORE_INDEX] * n_prompt_tokens

        # Pad to context_length
        pad_len = context_length - len(full_ids)
        full_ids = full_ids + [tokenizer.pad_id] * pad_len
        targets = targets + [IGNORE_INDEX] * pad_len

        all_input_ids.append(full_ids[:context_length])
        all_targets.append(targets[:context_length])

    x = torch.tensor(all_input_ids, dtype=torch.long, device=device)
    y = torch.tensor(all_targets, dtype=torch.long, device=device)
    return x, y


def main(model_config_path, train_config_path):
    model_cfg = ModelConfig.from_yaml(model_config_path)
    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    device = train_cfg["system"]["device"] if torch.cuda.is_available() else "cpu"
    tokenizer = LLMTokenizer()

    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["optimizer"]["learning_rate"])

    # Load pretrained weights
    pretrained_path = train_cfg["system"]["pretrained_checkpoint"]
    if os.path.exists(pretrained_path):
        load_checkpoint(pretrained_path, model, optimizer=None, device=device)
    else:
        print(f"WARNING: {pretrained_path} not found. Starting SFT from randomly initialized model "
              f"(for testing only).")

    print("Loading SFT dataset...")
    dataset = load_dataset(train_cfg["data"]["sft_dataset"], split="train")
    dataset = dataset.select(range(min(len(dataset), train_cfg["data"]["max_samples"])))

    # Format dataset examples
    examples = []
    for ex in dataset:
        messages = ex.get("messages", [])
        if len(messages) >= 2:
            examples.append({
                "instruction": messages[0]["content"],
                "response": messages[1]["content"],
            })
    print(f"Processed SFT samples: {len(examples)}")

    template = train_cfg["data"]["prompt_template"]
    batch_size = train_cfg["training"]["batch_size"]
    max_steps = train_cfg["training"]["max_steps"]
    grad_accum_steps = train_cfg["training"].get("gradient_accumulation_steps", 1)

    model.train()
    for step in range(max_steps):
        lr = get_lr(
            step, train_cfg["optimizer"]["warmup_steps"], max_steps,
            train_cfg["optimizer"]["learning_rate"], train_cfg["optimizer"]["min_learning_rate"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            global_sample_idx = step * batch_size * grad_accum_steps + micro_step * batch_size
            batch_examples = [examples[i % len(examples)] for i in
                               range(global_sample_idx, global_sample_idx + batch_size)]
            x, y = build_sft_batch(batch_examples, tokenizer, template, model_cfg.context_length, device)

            logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # Scale loss for accumulation
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["optimizer"]["grad_clip"])
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step:4d} | sft_loss {accum_loss:.4f} | lr {lr:.2e}")

        if step % train_cfg["training"]["checkpoint_interval"] == 0 and step > 0:
            save_checkpoint(model, optimizer, step, model_cfg,
                             train_cfg["system"]["checkpoint_dir"], f"sft_step{step}.pt")

    save_checkpoint(model, optimizer, max_steps, model_cfg,
                     train_cfg["system"]["checkpoint_dir"], "sft_final.pt")
    print("SFT complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train_sft.yaml")
    args = parser.parse_args()
    main(args.model_config, args.train_config)