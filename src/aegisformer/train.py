from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import fgsm, pgd
from .data import build_dataloaders
from .losses import cutmix_batch, mixup_batch, soft_target_cross_entropy
from .models import build_model
from .utils import cosine_with_warmup, ensure_dir, resolve_device, seed_everything


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _attack(model, x, y, rcfg, mean, std):
    name = rcfg.get("attack", "pgd").lower()
    if name == "fgsm":
        return fgsm(model, x, y, float(rcfg["epsilon"]), mean, std)
    return pgd(
        model, x, y,
        float(rcfg["epsilon"]), float(rcfg["alpha"]), int(rcfg["steps"]),
        mean, std,
    )


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.numel()
    return total_loss / total, correct / total


def train_supervised(cfg: dict, pretrained: str | None = None):
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = ensure_dir(cfg.get("output_dir", "runs/default"))
    bundle = build_dataloaders(cfg)
    cfg["model"]["num_classes"] = bundle.num_classes
    model = build_model(cfg).to(device)

    if pretrained:
        ckpt = torch.load(pretrained, map_location=device)
        state = ckpt.get("backbone", ckpt.get("model", ckpt))
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained backbone. missing={len(missing)} unexpected={len(unexpected)}")

    tcfg = cfg["train"]
    rcfg = cfg.get("robust", {})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.05))
    )
    scheduler = cosine_with_warmup(
        optimizer, int(tcfg["epochs"]), len(bundle.train), int(tcfg.get("warmup_epochs", 0))
    )
    amp_enabled = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_acc = -1.0
    history = []

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        bar = tqdm(bundle.train, desc=f"train {epoch:03d}", leave=False)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            mixup_alpha = float(tcfg.get("mixup_alpha", 0.0))
            cutmix_alpha = float(tcfg.get("cutmix_alpha", 0.0))
            if cutmix_alpha > 0 and torch.rand(()) < 0.5:
                x_clean, y_soft = cutmix_batch(x, y, cutmix_alpha, bundle.num_classes)
            else:
                x_clean, y_soft = mixup_batch(x, y, mixup_alpha, bundle.num_classes)

            smoothing = float(tcfg.get("label_smoothing", 0.0))
            if smoothing > 0:
                y_soft = (1.0 - smoothing) * y_soft + smoothing / bundle.num_classes

            adversarial = bool(rcfg.get("adversarial_training", False))
            x_adv = None
            if adversarial:
                model.eval()
                with torch.enable_grad():
                    x_adv = _attack(model, x, y, rcfg, bundle.mean, bundle.std)
                model.train()

            with _autocast(device, amp_enabled):
                logits = model(x_clean)
                clean_loss = soft_target_cross_entropy(logits, y_soft)
                loss = clean_loss
                if x_adv is not None:
                    adv_logits = model(x_adv)
                    adv_loss = F.cross_entropy(adv_logits, y, label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
                    clean_w = float(rcfg.get("clean_loss_weight", 0.5))
                    loss = clean_w * clean_loss + (1.0 - clean_w) * adv_loss

            scaler.scale(loss).backward()
            if float(tcfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * y.size(0)
            running_correct += logits.argmax(1).eq(y).sum().item()
            seen += y.size(0)
            bar.set_postfix(loss=f"{running_loss/seen:.4f}", acc=f"{running_correct/seen:.3f}")

        val_loss, val_acc = evaluate_accuracy(model, bundle.val, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": running_correct / seen,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"epoch={epoch:03d} train_acc={row['train_accuracy']:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_accuracy": val_acc}
        torch.save(ckpt, out_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, out_dir / "best.pt")

        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Artifacts: {Path(out_dir).resolve()}")
    return model
