from __future__ import annotations

import torch
import torch.nn.functional as F


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels)
    ece = torch.zeros((), device=logits.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            gap = conf[mask].mean() - acc[mask].float().mean()
            ece += mask.float().mean() * gap.abs()
    return ece.item()


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, logits.shape[1]).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return F.cross_entropy(logits, labels).item()
