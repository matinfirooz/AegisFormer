import torch

from aegisformer.attacks import fgsm, pgd
from aegisformer.models.vit import VisionTransformer


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def model():
    return VisionTransformer(image_size=32, patch_size=4, embed_dim=48, depth=1, num_heads=3)


def test_fgsm_shape():
    m = model()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    out = fgsm(m, x, y, 2/255, MEAN, STD)
    assert out.shape == x.shape
    assert not out.requires_grad


def test_pgd_shape():
    m = model()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    out = pgd(m, x, y, 2/255, 1/255, 2, MEAN, STD)
    assert out.shape == x.shape
