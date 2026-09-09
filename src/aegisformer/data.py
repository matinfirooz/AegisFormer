from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _supervised_transform(image_size: int, augmentation: str, train: bool):
    if train:
        ops = [transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()]
        if augmentation == "strong":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
        ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        if augmentation == "strong":
            ops.append(transforms.RandomErasing(p=0.25))
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _simclr_transform(image_size: int):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_dataloaders(cfg: dict, contrastive: bool = False) -> DataBundle:
    dcfg = cfg["data"]
    dataset_name = dcfg.get("dataset", "cifar10").lower()
    if dataset_name != "cifar10":
        raise ValueError("This starter release currently supports CIFAR-10; extend data.py for new datasets.")

    image_size = int(dcfg.get("image_size", 32))
    root = dcfg.get("root", "data")
    batch_size = int(dcfg.get("batch_size", 128))
    num_workers = int(dcfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()

    if contrastive:
        train_transform = TwoCropTransform(_simclr_transform(image_size))
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
        val_transform = _supervised_transform(image_size, "standard", False)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=val_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        return DataBundle(train_loader, val_loader, val_loader, 10, CIFAR10_MEAN, CIFAR10_STD)

    augmentation = dcfg.get("augmentation", "standard")
    full_train = datasets.CIFAR10(
        root, train=True, download=True,
        transform=_supervised_transform(image_size, augmentation, True),
    )
    val_set_source = datasets.CIFAR10(
        root, train=True, download=True,
        transform=_supervised_transform(image_size, augmentation, False),
    )
    test_set = datasets.CIFAR10(
        root, train=False, download=True,
        transform=_supervised_transform(image_size, augmentation, False),
    )

    val_split = float(dcfg.get("val_split", 0.1))
    n_val = int(len(full_train) * val_split)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    indices = torch.randperm(len(full_train), generator=gen).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_set_source, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return DataBundle(train_loader, val_loader, test_loader, 10, CIFAR10_MEAN, CIFAR10_STD)
