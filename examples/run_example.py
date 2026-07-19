from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_distill.models import FullModalityTeacher, RADistillStudent
from ra_distill.synthetic import SyntheticSkinLesionDataset, split_dataset
from ra_distill.train import build_retrieval_bank, evaluate_student, train_student, train_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RA-Distill example")
    parser.add_argument("--missing", choices=["clinical", "image"], default="clinical")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--clinical-dim", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retrieval-temperature", type=float, default=0.2)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=0.5)
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = SyntheticSkinLesionDataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        clinical_dim=args.clinical_dim,
    )
    train_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    bank_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    teacher = FullModalityTeacher(
        clinical_dim=args.clinical_dim,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
    )
    print("\n[1/3] Train full-modality teacher")
    train_teacher(teacher, train_loader, device=device, epochs=args.teacher_epochs)

    print("\n[2/3] Build retrieval representation bank from the training set")
    bank = build_retrieval_bank(teacher, bank_loader, device=device)
    print(f"Bank size: {len(bank.sample_ids)} complete training samples")

    print(f"\n[3/3] Train RA-Distill student under {args.missing}-missing setting")
    student = RADistillStudent(
        teacher=teacher,
        bank=bank,
        missing=args.missing,
        k=args.k,
        retrieval_temperature=args.retrieval_temperature,
        embed_dim=args.embed_dim,
    )
    train_student(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        device=device,
        epochs=args.student_epochs,
        lambda_rec=args.lambda_rec,
        lambda_kd=args.lambda_kd,
        distill_temperature=args.distill_temperature,
    )
    metrics = evaluate_student(student, test_loader, device=device)
    print(f"\nTest metrics: {metrics}")


if __name__ == "__main__":
    main()
