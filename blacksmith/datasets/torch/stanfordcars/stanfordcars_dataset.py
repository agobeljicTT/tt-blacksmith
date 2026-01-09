# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import StanfordCars as stanfordcars_dataset

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig

# Constants for ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # Size of the center crop for ViT
NUM_CLASSES = 10


class StanfordCarsDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split="train", collate_fn=None):
        self.config = config
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    def _prepare_dataset(self):
        dtype = eval(self.config.dtype)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),  # Center crop to 224x224 for ViT
            transforms.ToTensor(),
            transforms.Normalize(  # Normalize to ImageNet statistics
                mean=MEAN,
                std=STD
            ),
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ])
        target_transform = transforms.Compose(
            [
                transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
                transforms.Lambda(lambda y: F.one_hot(y, num_classes=NUM_CLASSES).to(dtype)),
            ]
        )

        self.dataset = mnist_dataset(
            root="data",
            train=self.split == "train",
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=self.split == "train", drop_last=True
        )
