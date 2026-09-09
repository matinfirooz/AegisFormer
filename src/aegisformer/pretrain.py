from __future__ import annotations

import csv

import torch
from torch import nn
from tqdm import tqdm

from .data import build_dataloaders
from .losses import nt_xent_loss
from .models import build_model
from .utils import cosine_with_warmup, ensure_dir, resolve_device, seed_everything


class SimCLR(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, projection_dim: int):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        return self.projector(self.backbone.forward_features(x))


def pretrain_simclr(cfg: dict):
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = ensure_dir(cfg.get("output_dir", "runs/simclr"))
    bundle = build_dataloaders(cfg, contrastive=True)
    backbone = build_model({**cfg, "model": {**cfg["model"], "num_classes": 10}}).to(device)
    pcfg = cfg["pretrain"]
    model = SimCLR(backbone, int(cfg["model"]["embed_dim"]), int(pcfg.get("projection_dim", 128))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pcfg["lr"]), weight_decay=float(pcfg.get("weight_decay", 0.05)))
    scheduler = cosine_with_warmup(optimizer, int(pcfg["epochs"]), len(bundle.train), warmup_epochs=5)
    amp_enabled = bool(pcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []

    for epoch in range(1, int(pcfg["epochs"]) + 1):
        model.train()
        total = 0.0
        seen = 0
        for (x1, x2), _ in tqdm(bundle.train, desc=f"simclr {epoch:03d}", leave=False):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = nt_xent_loss(model(x1), model(x2), float(pcfg.get("temperature", 0.2)))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * x1.size(0)
            seen += x1.size(0)
        avg = total / seen
        history.append({"epoch": epoch, "contrastive_loss": avg})
        print(f"epoch={epoch:03d} contrastive_loss={avg:.4f}")
        torch.save({"backbone": backbone.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "backbone_last.pt")
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
    return backbone
