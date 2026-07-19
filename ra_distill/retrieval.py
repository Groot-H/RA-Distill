from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class RetrievalBank:
    image_reps: torch.Tensor
    clinical_reps: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]

    @classmethod
    @torch.no_grad()
    def from_teacher(
        cls,
        teacher,
        dataloader: DataLoader,
        device: torch.device,
    ) -> "RetrievalBank":
        teacher.eval()
        image_reps = []
        clinical_reps = []
        labels = []
        sample_ids: list[str] = []
        for batch in dataloader:
            image = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            image_rep = teacher.encode_image(image)
            clinical_rep = teacher.encode_clinical(clinical)
            image_reps.append(F.normalize(image_rep, dim=-1).cpu())
            clinical_reps.append(F.normalize(clinical_rep, dim=-1).cpu())
            labels.append(batch["label"].cpu())
            sample_ids.extend([str(x) for x in batch["sample_id"]])
        return cls(
            image_reps=torch.cat(image_reps, dim=0),
            clinical_reps=torch.cat(clinical_reps, dim=0),
            labels=torch.cat(labels, dim=0),
            sample_ids=sample_ids,
        )

