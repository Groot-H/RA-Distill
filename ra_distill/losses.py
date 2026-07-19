from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .models import RADistillOutput


@dataclass
class RADistillLoss:
    lambda_rec: float = 1.0
    lambda_kd: float = 0.5
    distill_temperature: float = 2.0

    def __call__(
        self,
        output: RADistillOutput,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = nn.functional.cross_entropy(output.logits, target)
        rec_loss = F.mse_loss(output.reconstructed_missing_rep, output.target_missing_rep)

        if output.teacher_logits is None:
            kd_loss = torch.zeros((), device=output.logits.device)
        else:
            temperature = max(float(self.distill_temperature), 1e-6)
            kd_loss = F.kl_div(
                F.log_softmax(output.logits / temperature, dim=1),
                F.softmax(output.teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)

        loss = ce_loss + float(self.lambda_rec) * rec_loss + float(self.lambda_kd) * kd_loss
        logs = {
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "rec_loss": float(rec_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
        }
        return loss, logs

