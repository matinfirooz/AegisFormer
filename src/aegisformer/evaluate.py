from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import fgsm, pgd
from .calibration import TemperatureScaler
from .data import build_dataloaders
from .metrics import brier_score, expected_calibration_error, negative_log_likelihood
from .models import build_model
from .utils import load_checkpoint, resolve_device


@torch.no_grad()
def _collect(model, loader, device):
    logits_all, labels_all = [], []
    model.eval()
    for x, y in tqdm(loader, desc="inference", leave=False):
        logits_all.append(model(x.to(device)).cpu())
        labels_all.append(y)
    return torch.cat(logits_all), torch.cat(labels_all)


def evaluate_checkpoint(cfg: dict, checkpoint: str, attack: str = "none", calibrate: bool = False):
    device = resolve_device(cfg.get("device", "auto"))
    bundle = build_dataloaders(cfg)
    cfg["model"]["num_classes"] = bundle.num_classes
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, device)

    val_logits, val_labels = _collect(model, bundle.val, device)
    test_logits, test_labels = _collect(model, bundle.test, device)

    scaler = None
    if calibrate:
        scaler = TemperatureScaler().to(device)
        scaler.fit(val_logits.to(device), val_labels.to(device))
        test_logits = scaler(test_logits.to(device)).cpu()

    result = {
        "clean_accuracy": test_logits.argmax(1).eq(test_labels).float().mean().item(),
        "nll": negative_log_likelihood(test_logits, test_labels),
        "ece": expected_calibration_error(test_logits, test_labels),
        "brier": brier_score(test_logits, test_labels),
    }
    if scaler is not None:
        result["temperature"] = scaler.temperature.item()

    if attack != "none":
        rcfg = cfg.get("robust", {})
        correct = total = 0
        model.eval()
        for x, y in tqdm(bundle.test, desc=attack, leave=False):
            x, y = x.to(device), y.to(device)
            if attack == "fgsm":
                x_adv = fgsm(model, x, y, float(rcfg.get("epsilon", 8/255)), bundle.mean, bundle.std)
            elif attack == "pgd":
                x_adv = pgd(model, x, y, float(rcfg.get("epsilon", 8/255)),
                            float(rcfg.get("alpha", 2/255)), int(rcfg.get("steps", 10)),
                            bundle.mean, bundle.std)
            else:
                raise ValueError("attack must be one of: none, fgsm, pgd")
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
            correct += pred.eq(y).sum().item(); total += y.numel()
        result[f"{attack}_accuracy"] = correct / total

    return result
