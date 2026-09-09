import torch

from aegisformer.models.vit import VisionTransformer


def tiny_model():
    return VisionTransformer(
        image_size=32, patch_size=4, num_classes=10,
        embed_dim=96, depth=2, num_heads=3, stochastic_depth=0.0,
    )


def test_forward_shape():
    model = tiny_model()
    y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_attention_rollout_shape():
    model = tiny_model().eval()
    mask = model.attention_rollout(torch.randn(2, 3, 32, 32))
    assert mask.shape == (2, 8, 8)
