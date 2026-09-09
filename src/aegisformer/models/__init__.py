from .vit import VisionTransformer


def build_model(cfg: dict):
    mcfg = cfg["model"]
    name = mcfg.get("name", "vit_tiny")
    if name != "vit_tiny":
        raise ValueError(f"Unknown model: {name}")
    kwargs = {k: v for k, v in mcfg.items() if k != "name"}
    return VisionTransformer(**kwargs)


__all__ = ["VisionTransformer", "build_model"]
