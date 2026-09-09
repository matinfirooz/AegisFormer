from __future__ import annotations

import torch
import torch.nn.functional as F


def _bounds(mean, std, device):
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    lo = (torch.zeros_like(mean) - mean) / std
    hi = (torch.ones_like(mean) - mean) / std
    return lo, hi, std


def fgsm(model, x, y, epsilon, mean, std):
    lo, hi, std_t = _bounds(mean, std, x.device)
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    eps_norm = epsilon / std_t
    x_adv = x_adv + eps_norm * grad.sign()
    return torch.maximum(torch.minimum(x_adv.detach(), hi), lo)


def pgd(model, x, y, epsilon, alpha, steps, mean, std, random_start=True):
    lo, hi, std_t = _bounds(mean, std, x.device)
    eps_norm = epsilon / std_t
    alpha_norm = alpha / std_t
    x0 = x.detach()
    if random_start:
        delta = torch.empty_like(x0).uniform_(-1, 1) * eps_norm
        x_adv = torch.maximum(torch.minimum(x0 + delta, hi), lo)
    else:
        x_adv = x0.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha_norm * grad.sign()
        x_adv = torch.maximum(torch.minimum(x_adv, x0 + eps_norm), x0 - eps_norm)
        x_adv = torch.maximum(torch.minimum(x_adv, hi), lo)
    return x_adv.detach()
