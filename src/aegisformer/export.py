from __future__ import annotations

from pathlib import Path

import torch

from .models import build_model
from .utils import load_checkpoint, resolve_device


def export_model(cfg: dict, checkpoint: str, output: str, fmt: str = "torchscript"):
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device).eval()
    load_checkpoint(model, checkpoint, device)
    image_size = int(cfg["model"].get("image_size", 32))
    example = torch.randn(1, 3, image_size, image_size, device=device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "torchscript":
        traced = torch.jit.trace(model, example)
        traced.save(str(output))
    elif fmt == "onnx":
        torch.onnx.export(
            model, example, str(output), input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
        )
    else:
        raise ValueError("fmt must be torchscript or onnx")
    return output
