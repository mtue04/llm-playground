import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from model.architecture import GPTModel


class RewardModel(nn.Module):
    def __init__(self, config, pretrained_backbone_path=None):
        super().__init__()
        self.backbone = GPTModel(config)
        if pretrained_backbone_path:
            ckpt = torch.load(pretrained_backbone_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(ckpt["model_state_dict"])

        self.reward_head = nn.Linear(config.d_model, 1, bias=False)

    def forward(self, idx):
        """Returns reward scalar for each sequence in the batch from the last token."""
        x = self.backbone.token_embedding(idx)
        cos = self.backbone.rope_cos.to(x.device)
        sin = self.backbone.rope_sin.to(x.device)
        for block in self.backbone.blocks:
            x = block(x, cos, sin)
        x = self.backbone.final_norm(x)

        last_hidden = x[:, -1, :]  # (batch, d_model)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


def reward_model_loss(chosen_rewards, rejected_rewards):
    """Bradley-Terry loss: reward(chosen) > reward(rejected)."""
    return -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
