from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .evaluate import evaluate_checkpoint
from .export import export_model
from .models import build_model
from .pretrain import pretrain_simclr
from .profiler import benchmark_latency, count_parameters, estimate_macs
from .train import train_supervised
from .utils import load_checkpoint, resolve_device


def _parser():
    p = argparse.ArgumentParser(prog="aegisformer", description="AegisFormer research CLI")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Supervised or adversarially robust training")
    train.add_argument("--config", required=True)
    train.add_argument("--pretrained", default=None, help="Optional SimCLR backbone checkpoint")

    pre = sub.add_parser("pretrain", help="SimCLR self-supervised pretraining")
    pre.add_argument("--config", required=True)

    ev = sub.add_parser("evaluate", help="Evaluate accuracy, calibration, and robustness")
    ev.add_argument("--config", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--attack", choices=["none", "fgsm", "pgd"], default="none")
    ev.add_argument("--calibrate", action="store_true")

    prof = sub.add_parser("profile", help="Profile parameters, MACs, latency, throughput")
    prof.add_argument("--config", required=True)
    prof.add_argument("--checkpoint", default=None)
    prof.add_argument("--batch-size", type=int, default=1)

    exp = sub.add_parser("export", help="Export to TorchScript or ONNX")
    exp.add_argument("--config", required=True)
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--output", required=True)
    exp.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    return p


def main():
    args = _parser().parse_args()
    cfg = load_config(args.config)

    if args.command == "train":
        train_supervised(cfg, pretrained=args.pretrained)
    elif args.command == "pretrain":
        pretrain_simclr(cfg)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(cfg, args.checkpoint, attack=args.attack, calibrate=args.calibrate)
        print(json.dumps(result, indent=2))
    elif args.command == "profile":
        device = resolve_device(cfg.get("device", "auto"))
        model = build_model(cfg).to(device).eval()
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, device)
        size = int(cfg["model"].get("image_size", 32))
        x = torch.randn(args.batch_size, 3, size, size, device=device)
        result = {
            "device": str(device),
            "parameters": count_parameters(model),
            "macs_estimate": estimate_macs(model, x),
            **benchmark_latency(model, x),
        }
        print(json.dumps(result, indent=2))
    elif args.command == "export":
        out = export_model(cfg, args.checkpoint, args.output, args.format)
        print(f"Exported: {out}")


if __name__ == "__main__":
    main()
