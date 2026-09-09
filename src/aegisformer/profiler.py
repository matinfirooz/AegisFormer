from __future__ import annotations

import statistics
import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, example: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs
        batch, out_c, out_h, out_w = out.shape
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += batch * out_c * out_h * out_w * kernel_ops

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal macs
        x = inp[0]
        n = x.numel() // x.shape[-1]
        macs += n * m.in_features * m.out_features

    def mha_hook(m: nn.MultiheadAttention, inp, out):
        nonlocal macs
        x = inp[0]
        batch, tokens, dim = x.shape
        # Q/K/V projections + output projection + QK^T + attention @ V.
        macs += batch * (4 * tokens * dim * dim + 2 * tokens * tokens * dim)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(example)
    for hook in hooks:
        hook.remove()
    return int(macs)


@torch.no_grad()
def benchmark_latency(model: nn.Module, example: torch.Tensor, warmup: int = 10, runs: int = 50):
    model.eval()
    for _ in range(warmup):
        _ = model(example)
    if example.is_cuda:
        torch.cuda.synchronize()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(example)
        if example.is_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_p50": statistics.median(timings),
        "latency_ms_min": min(timings),
        "throughput_img_s": example.shape[0] / (statistics.mean(timings) / 1000),
    }
