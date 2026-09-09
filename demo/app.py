from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from aegisformer.config import load_config
from aegisformer.data import CIFAR10_MEAN, CIFAR10_STD
from aegisformer.models import build_model
from aegisformer.utils import load_checkpoint, resolve_device

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_app(config: str, checkpoint: str):
    cfg = load_config(config)
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device).eval()
    load_checkpoint(model, checkpoint, device)
    size = int(cfg["model"].get("image_size", 32))
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    def predict(img: Image.Image):
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0].cpu()
        return {name: float(probs[i]) for i, name in enumerate(CLASSES)}

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Input image"),
        outputs=gr.Label(num_top_classes=5, label="Prediction"),
        title="AegisFormer — Robust Vision Transformer",
        description="CIFAR-10 inference demo for a ViT trained with the AegisFormer research stack.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    build_app(args.config, args.checkpoint).launch(share=args.share)
