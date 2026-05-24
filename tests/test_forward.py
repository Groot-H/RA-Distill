from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_distill.models import FullModalityTeacher, RADistillStudent
from ra_distill.retrieval import RetrievalBank
from ra_distill.synthetic import SyntheticSkinLesionDataset


def test_ra_distill_forward_clinical_missing():
    dataset = SyntheticSkinLesionDataset(num_samples=16, num_classes=3, clinical_dim=5, image_size=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    teacher = FullModalityTeacher(clinical_dim=5, num_classes=3, embed_dim=32)
    bank = RetrievalBank.from_teacher(teacher, loader, device=torch.device("cpu"))
    student = RADistillStudent(teacher, bank, missing="clinical", k=3, embed_dim=32)
    batch = next(iter(loader))
    output = student(batch["image"], batch["clinical"])
    assert output.logits.shape == (8, 3)
    assert output.reconstructed_missing_rep.shape == (8, 32)
    assert output.prompt_rep.shape == (8, 32)

