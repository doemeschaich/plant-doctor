"""
Dataset class for PlantVillage + DataLoader factory.
Reads CSV manifests produced by split_data.py.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ImageNet normalization stats — used because we'll use ImageNet-pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224  # Standard size for most ImageNet-pretrained models


class PlantDataset(Dataset):
    """Loads images from a CSV manifest with filepath + label columns.

    CSV file paths are interpreted relative to `root_dir`. This allows the
    same CSV (with project-relative paths) to be used from any working
    directory — just pass the appropriate root_dir.
    """

    def __init__(self, csv_path: Path, transform=None, root_dir: Path = Path(".")):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.root_dir = Path(root_dir)
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.root_dir / row["filepath"]
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool):
    """Returns transform pipeline. Augmentations only for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(
    processed_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    root_dir: Path = Path("."),
):
    """Returns train, val, test DataLoaders + the class list."""
    train_ds = PlantDataset(processed_dir / "train.csv", get_transforms(train=True), root_dir)
    val_ds = PlantDataset(processed_dir / "val.csv", get_transforms(train=False), root_dir)
    test_ds = PlantDataset(processed_dir / "test.csv", get_transforms(train=False), root_dir)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, train_ds.classes