import math
import os
import shutil
import numpy as np
import torch


def get_batch(bin_path, batch_size, context_length, device):
    """Get a batch of data from the binary file"""
    data = np.memmap(bin_path, dtype=np.uint32, mode='r')

    ix = torch.randint(len(data) - context_length - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i + context_length].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + context_length].astype(np.int64)) for i in ix
    ])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Get the learning rate for the current step"""
    if warmup_steps == 0 or step >= warmup_steps:
        if max_steps <= warmup_steps:
            return min_lr
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        lr = max_lr * step / warmup_steps
    return lr


@torch.no_grad()
def estimate_loss(model, bin_path, batch_size, context_length, device, eval_iters):
    """Estimate the loss of the model on the validation set"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(bin_path, batch_size, context_length, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(model, optimizer, step, config, checkpoint_dir, name, drive_backup_dir=None):
    """Save a checkpoint of the model and optimizer state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
    }, path)
    print(f"Saved checkpoint: {path}")

    if drive_backup_dir and os.path.exists(os.path.dirname(drive_backup_dir) or "."):
        try:
            os.makedirs(drive_backup_dir, exist_ok=True)
            shutil.copy(path, os.path.join(drive_backup_dir, name))
            print(f"Backed up to Drive: {drive_backup_dir}/{name}")
        except Exception as e:
            print(f"Warning: Drive backup failed ({e}). Local checkpoint is safe.")


def load_checkpoint(path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path} (step {ckpt.get('step', '?')})")
    return ckpt.get("step", 0)
