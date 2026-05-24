from __future__ import annotations

import math
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .retrieval import RetrievalBank


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class TinyImageEncoder(nn.Module):
    """Image encoder used by the example implementation."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            output_dim=embed_dim,
            dropout=dropout,
            use_batch_norm=False,
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.net(clinical)


class AttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        return self.norm2(tokens + self.ffn(tokens))


class FullModalityTeacher(nn.Module):
    """Full-modality teacher trained on complete image-clinical pairs."""

    def __init__(
        self,
        clinical_dim: int,
        num_classes: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_encoder = TinyImageEncoder(embed_dim=embed_dim)
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            dropout=dropout,
        )
        self.fusion = AttentionFusionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.classifier = build_mlp(embed_dim, [embed_dim], num_classes, dropout=dropout)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_clinical(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.clinical_encoder(clinical)

    def fuse(self, image_rep: torch.Tensor, clinical_rep: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([image_rep, clinical_rep], dim=1)
        fused_tokens = self.fusion(tokens)
        return fused_tokens.mean(dim=1)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        image_rep = self.encode_image(image)
        clinical_rep = self.encode_clinical(clinical)
        fused = self.fuse(image_rep, clinical_rep)
        return self.classifier(fused)


class AdaptiveRetriever(nn.Module):
    """Top-k retrieval with learnable query/key projections."""

    def __init__(self, embed_dim: int, k: int = 3, temperature: float = 0.2) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.key_projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.k = int(k)
        self.temperature = float(temperature)

    def forward(
        self,
        query: torch.Tensor,
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_proj = F.normalize(self.query_projection(query), dim=-1)
        key_proj = F.normalize(self.key_projection(bank_keys), dim=-1)
        scores = query_proj @ key_proj.t()
        k = min(max(self.k, 1), bank_keys.shape[0])
        top_scores, top_indices = torch.topk(scores, k=k, dim=1)
        weights = torch.softmax(top_scores / max(self.temperature, 1e-6), dim=1)
        retrieved_values = bank_values[top_indices]
        weighted_value = torch.sum(weights.unsqueeze(-1) * retrieved_values, dim=1)
        return {
            "scores": top_scores,
            "indices": top_indices,
            "weights": weights,
            "retrieved_values": retrieved_values,
            "weighted_value": weighted_value,
        }


class ConditionalMoEReconstructor(nn.Module):
    """Query-conditioned aggregation of retrieved missing-modality representations."""

    def __init__(self, embed_dim: int, k: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.k = int(k)
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(self.k)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, retrieved_missing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actual_k = retrieved_missing.shape[1]
        if actual_k > len(self.experts):
            raise ValueError("retrieved k is larger than the configured number of experts")

        projected_query = self.query_projection(query)
        projected_keys = self.key_projection(retrieved_missing)
        routing_logits = torch.einsum("bd,bkd->bk", projected_query, projected_keys) / math.sqrt(query.shape[-1])
        routing_weights = torch.softmax(routing_logits, dim=1)

        expert_outputs = []
        for expert_index in range(actual_k):
            expert_outputs.append(self.experts[expert_index](retrieved_missing[:, expert_index]))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        reconstructed = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return self.norm(reconstructed), routing_weights


class SampleAwarePromptGenerator(nn.Module):
    """Generate a sample-aware prompt representation from retrieval context."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        retrieved_missing: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=query.unsqueeze(1),
            key=retrieved_missing,
            value=retrieved_missing,
            need_weights=False,
        )
        weighted_context = torch.sum(routing_weights.unsqueeze(-1) * retrieved_missing, dim=1)
        prompt_input = torch.cat([attended.squeeze(1), weighted_context], dim=-1)
        return self.prompt_mlp(prompt_input)


@dataclass
class RADistillOutput:
    logits: torch.Tensor
    available_rep: torch.Tensor
    reconstructed_missing_rep: torch.Tensor
    target_missing_rep: torch.Tensor
    prompt_rep: torch.Tensor
    teacher_logits: torch.Tensor | None
    retrieval_weights: torch.Tensor
    retrieval_indices: torch.Tensor


class RADistillStudent(nn.Module):
    """RA-Distill student for clinical-missing or image-missing diagnosis."""

    def __init__(
        self,
        teacher: FullModalityTeacher,
        bank: RetrievalBank,
        missing: str = "clinical",
        k: int = 3,
        retrieval_temperature: float = 0.2,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if missing not in {"clinical", "image"}:
            raise ValueError("missing must be either 'clinical' or 'image'")
        self.missing = missing
        self.bank = bank
        self.image_encoder = copy.deepcopy(teacher.image_encoder)
        self.clinical_encoder = copy.deepcopy(teacher.clinical_encoder)
        self.retriever = AdaptiveRetriever(embed_dim=embed_dim, k=k, temperature=retrieval_temperature)
        self.reconstructor = ConditionalMoEReconstructor(embed_dim=embed_dim, k=k, dropout=dropout)
        self.prompt_generator = SampleAwarePromptGenerator(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.fusion = copy.deepcopy(teacher.fusion)
        self.classifier = copy.deepcopy(teacher.classifier)

        # Use teacher-initialized modality encoders to keep representations aligned with the retrieval bank.
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.clinical_encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> RADistillOutput:
        image_rep = self.image_encoder(image)
        clinical_rep = self.clinical_encoder(clinical)

        if self.missing == "clinical":
            available_rep = image_rep
            target_missing_rep = clinical_rep.detach()
            bank_keys = self.bank.image_reps.to(image.device)
            bank_values = self.bank.clinical_reps.to(image.device)
            token_order = "image_first"
        else:
            available_rep = clinical_rep
            target_missing_rep = image_rep.detach()
            bank_keys = self.bank.clinical_reps.to(image.device)
            bank_values = self.bank.image_reps.to(image.device)
            token_order = "clinical_first"

        retrieval = self.retriever(
            query=available_rep,
            bank_keys=bank_keys,
            bank_values=bank_values,
        )
        reconstructed_missing_rep, routing_weights = self.reconstructor(
            query=available_rep,
            retrieved_missing=retrieval["retrieved_values"],
        )
        prompt_rep = self.prompt_generator(
            query=available_rep,
            retrieved_missing=retrieval["retrieved_values"],
            routing_weights=routing_weights,
        )

        if token_order == "image_first":
            tokens = torch.stack([available_rep, reconstructed_missing_rep, prompt_rep], dim=1)
        else:
            tokens = torch.stack([reconstructed_missing_rep, available_rep, prompt_rep], dim=1)
        fused = self.fusion(tokens).mean(dim=1)
        logits = self.classifier(fused)

        return RADistillOutput(
            logits=logits,
            available_rep=available_rep,
            reconstructed_missing_rep=reconstructed_missing_rep,
            target_missing_rep=target_missing_rep,
            prompt_rep=prompt_rep,
            teacher_logits=teacher_logits,
            retrieval_weights=retrieval["weights"],
            retrieval_indices=retrieval["indices"],
        )
