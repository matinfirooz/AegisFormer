from __future__ import annotations

import torch
import torch.nn.functional as F


def enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, passes: int = 20):
    model.eval()
    enable_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=1) for _ in range(passes)], dim=0)
    mean_prob = probs.mean(dim=0)
    predictive_entropy = -(mean_prob * mean_prob.clamp_min(1e-8).log()).sum(dim=1)
    expected_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=2).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy
    return {
        "mean_probability": mean_prob,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }
