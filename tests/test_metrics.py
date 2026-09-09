import torch

from aegisformer.losses import nt_xent_loss
from aegisformer.metrics import brier_score, expected_calibration_error


def test_calibration_metrics_ranges():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 0])
    ece = expected_calibration_error(logits, y)
    brier = brier_score(logits, y)
    assert 0.0 <= ece <= 1.0
    assert brier >= 0.0


def test_nt_xent_is_finite():
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)
