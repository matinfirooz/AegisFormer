from __future__ import annotations

import torch

from aegisformer.attacks import pgd
from aegisformer.models.vit import VisionTransformer
from aegisformer.profiler import count_parameters, estimate_macs
from aegisformer.uncertainty import mc_dropout_predict


def main():
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32, patch_size=4, num_classes=10,
        embed_dim=96, depth=2, num_heads=3, dropout=0.1,
    )
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    logits = model(x)
    assert logits.shape == (4, 10)

    adv = pgd(
        model, x, y, epsilon=2/255, alpha=1/255, steps=1,
        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
    )
    assert adv.shape == x.shape

    uncertainty = mc_dropout_predict(model, x, passes=3)
    assert uncertainty["predictive_entropy"].shape == (4,)

    params = count_parameters(model)
    macs = estimate_macs(model.eval(), x[:1])
    print({"status": "ok", "logits": list(logits.shape), "parameters": params, "macs_estimate": macs})


if __name__ == "__main__":
    main()
