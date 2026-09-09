from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int):
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    y1 = F.one_hot(y, num_classes).float()
    y2 = F.one_hot(y[perm], num_classes).float()
    return mixed_x, lam * y1 + (1 - lam) * y2


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int):
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[-2:]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx = torch.randint(0, w, (1,), device=x.device).item()
    cy = torch.randint(0, h, (1,), device=x.device).item()
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    y1p, y2p = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1p:y2p, x1:x2] = x[perm, :, y1p:y2p, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2p - y1p) / (w * h))
    one = F.one_hot(y, num_classes).float()
    two = F.one_hot(y[perm], num_classes).float()
    return mixed, lam_adj * one + (1 - lam_adj) * two


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    n = z1.shape[0]
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)])
    return F.cross_entropy(sim, targets)
