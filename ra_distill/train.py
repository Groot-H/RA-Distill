from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import RADistillLoss
from .models import FullModalityTeacher, RADistillStudent
from .retrieval import RetrievalBank


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().detach().cpu())


def average_logs(logs: list[dict[str, float]]) -> dict[str, float]:
    if not logs:
        return {}
    result = {}
    keys = logs[0].keys()
    for key in keys:
        result[key] = sum(row[key] for row in logs) / len(logs)
    return result


def train_teacher(
    teacher: FullModalityTeacher,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
) -> None:
    teacher.to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        teacher.train()
        logs = []
        progress = tqdm(train_loader, desc=f"teacher epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(image, clinical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            row = {"loss": float(loss.detach().cpu()), "acc": accuracy(logits, labels)}
            logs.append(row)
            progress.set_postfix(row)
        print(f"Teacher epoch {epoch}: {average_logs(logs)}")


@torch.no_grad()
def collect_teacher_logits(
    teacher: FullModalityTeacher,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    teacher.eval()
    logits_by_id = {}
    for batch in dataloader:
        image = batch["image"].to(device)
        clinical = batch["clinical"].to(device)
        logits = teacher(image, clinical).cpu()
        for sample_id, sample_logits in zip(batch["sample_id"], logits):
            logits_by_id[str(sample_id)] = sample_logits
    return logits_by_id


def train_student(
    student: RADistillStudent,
    teacher: FullModalityTeacher,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    lambda_rec: float = 1.0,
    lambda_kd: float = 0.5,
    distill_temperature: float = 2.0,
) -> None:
    student.to(device)
    teacher.to(device)
    teacher.eval()
    teacher_logits_by_id = collect_teacher_logits(teacher, train_loader, device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = RADistillLoss(
        lambda_rec=lambda_rec,
        lambda_kd=lambda_kd,
        distill_temperature=distill_temperature,
    )

    for epoch in range(1, epochs + 1):
        student.train()
        logs = []
        progress = tqdm(train_loader, desc=f"student epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            teacher_logits = torch.stack([teacher_logits_by_id[str(x)] for x in batch["sample_id"]]).to(device)

            optimizer.zero_grad(set_to_none=True)
            output = student(image=image, clinical=clinical, teacher_logits=teacher_logits)
            loss, row = criterion(output, labels)
            loss.backward()
            optimizer.step()
            row["acc"] = accuracy(output.logits, labels)
            row["retrieval_weight_top1"] = float(output.retrieval_weights[:, 0].mean().detach().cpu())
            logs.append(row)
            progress.set_postfix(row)
        print(f"Student epoch {epoch}: {average_logs(logs)}")


@torch.no_grad()
def evaluate_student(
    student: RADistillStudent,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    metric_sums = defaultdict(float)
    batches = 0
    for batch in dataloader:
        image = batch["image"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        output = student(image=image, clinical=clinical)
        metric_sums["acc"] += accuracy(output.logits, labels)
        batches += 1
    return {key: value / max(batches, 1) for key, value in metric_sums.items()}


def build_retrieval_bank(
    teacher: FullModalityTeacher,
    train_loader: DataLoader,
    device: torch.device,
) -> RetrievalBank:
    return RetrievalBank.from_teacher(teacher=teacher, dataloader=train_loader, device=device)

