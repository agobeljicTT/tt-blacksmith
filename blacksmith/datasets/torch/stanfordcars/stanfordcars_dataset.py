# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset

# Constants for ImageNet

# Usually Stanford Cars dataset is used to fine-tune models
# previously trained on ImageNet, so we use ImageNet statistics

DATASET_PATH = "tanganke/stanford_cars"


class StanfordCarsDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "stanfordcars")
            split: Dataset split to use ("train", "test")
            collate_fn: Unused, will fail if not None
        """
        assert split in ["train", "test"], "Split must be one of: train, test"
        assert collate_fn is None, "collate_fn is not supported for this dataset"

        self.config = config
        self.split = split
        self.image_processor = ViTImageProcessor.from_pretrained(config.model_name)

        self._prepare_dataset()

    def _get_transform_function(self):
        dtype = eval(self.config.dtype)

        img_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Looks like there are grayscale images in the dataset
                transforms.RandomResizedCrop(self.image_processor.size["height"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(  # Normalize to ImageNet statistics
                    mean=self.image_processor.image_mean,
                    std=self.image_processor.image_std,
                ),
                transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
            ]
        )
        label_transform = transforms.Compose(
            [
                transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
            ]
        )

        def transform_function(batch):
            transformed_batch = {}
            transformed_batch["image"] = [img_transform(img) for img in batch["image"]]
            transformed_batch["label"] = [label_transform(label) for label in batch["label"]]
            return transformed_batch

        return transform_function

    def _prepare_dataset(self):
        transform_function = self._get_transform_function()
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)

        self.dataset = raw_dataset.with_transform(transform_function)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.split == "train",
            drop_last=True,
        )
