from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def save_attention_rollout(model, x: torch.Tensor, output: str | Path):
    model.eval()
    rollout = model.attention_rollout(x[:1])[0]
    rollout = F.interpolate(rollout[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    rollout = rollout.detach().cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(rollout)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    return output
