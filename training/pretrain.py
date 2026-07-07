import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import ModelConfig
from model.architecture import GPTModel
from training.trainer_utils import get_batch, get_lr, estimate_loss, save_checkpoint, load_checkpoint


def main(model_config_path, train_config_path, resume_from=None):
    model_cfg = ModelConfig.from_yaml(model_config_path)
    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    device = train_cfg["system"]["device"] if torch.cuda.is_available() else "cpu"
    checkpoint_dir = train_cfg["system"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optimizer"]["learning_rate"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
        betas=(train_cfg["optimizer"]["beta1"], train_cfg["optimizer"]["beta2"])
    )

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(resume_from, model, optimizer, device=device)
    elif resume_from:
        print(f"WARNING: Checkpoint {resume_from} not found. Starting from scratch.")

    train_bin = os.path.join(train_cfg["data"]["process_dir"], train_cfg["data"]["train_bin"])
    val_bin = os.path.join(train_cfg["data"]["process_dir"], train_cfg["data"]["val_bin"])

    print(f"Training on {train_bin}, validating on {val_bin}")
    print(f"Model: {model.num_parameters() / 1e6:.2f}M parameters")

    batch_size = train_cfg["training"]["batch_size"]
    max_steps = train_cfg["training"]["max_steps"]
    eval_interval = train_cfg["training"]["eval_interval"]
    eval_iters = train_cfg["training"]["eval_iters"]
    checkpoint_interval = train_cfg["training"]["checkpoint_interval"]

    drive_backup_dir = train_cfg["system"].get("drive_backup_dir", None)

    model.train()
    for step in range(start_step, max_steps):
        lr = get_lr(
            step,
            train_cfg["optimizer"]["warmup_steps"],
            max_steps,
            float(train_cfg["optimizer"]["learning_rate"]),
            float(train_cfg["optimizer"]["min_learning_rate"])
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_bin, batch_size, model_cfg.context_length, device)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["optimizer"]["grad_clip"])
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step:4d} | train_loss {loss.item():.4f} | lr {lr:.2e}")

        if step % eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_bin, batch_size, model_cfg.context_length, device, eval_iters)
            print(f"step {step:4d} | validation_loss {val_loss:.4f}")

        if step % checkpoint_interval == 0 and step > start_step:
            save_checkpoint(
                model, optimizer, step, model_cfg,
                checkpoint_dir, f"pretrain_step{step}.pt",
                drive_backup_dir=drive_backup_dir
            )

    save_checkpoint(
        model, optimizer, max_steps, model_cfg,
        checkpoint_dir, "pretrain_final.pt",
        drive_backup_dir=drive_backup_dir
    )
    print("Pretraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train_pretrain.yaml")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    main(args.model_config, args.train_config, args.resume_from)
