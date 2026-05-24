from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticSkinLesionDataset(Dataset):
    """Example complete multimodal skin lesion dataset."""

    def __init__(
        self,
        num_samples: int = 512,
        num_classes: int = 4,
        clinical_dim: int = 10,
        image_size: int = 32,
        seed: int = 7,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.clinical_dim = int(clinical_dim)
        self.image_size = int(image_size)

        labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        class_image_patterns = torch.randn(num_classes, 3, image_size, image_size, generator=generator) * 0.6
        class_clinical_patterns = torch.randn(num_classes, clinical_dim, generator=generator) * 0.8

        image_noise = torch.randn(num_samples, 3, image_size, image_size, generator=generator) * 0.5
        clinical_noise = torch.randn(num_samples, clinical_dim, generator=generator) * 0.4
        self.images = class_image_patterns[labels] + image_noise
        self.clinical = class_clinical_patterns[labels] + clinical_noise
        self.labels = labels
        self.sample_ids = [f"sample_{index:05d}" for index in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.images[index].float(),
            "clinical": self.clinical[index].float(),
            "label": self.labels[index].long(),
            "sample_id": self.sample_ids[index],
        }


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 7):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
