from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from lab03.config import BuildConfig
from lab03.graph_pipeline import load_graph_bundle


@dataclass(slots=True)
class TrainingSummary:
    epochs: int
    final_loss: float
    best_loss: float
    hidden_dim: int
    dropout: float


class ResidualGCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x, edge_index)
        hidden = self.norm1(hidden)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.norm2(hidden)
        return F.normalize(x + hidden, p=2, dim=-1)


def build_pyg_data(raw_embeddings: np.ndarray, edge_index: np.ndarray) -> Data:
    x = torch.tensor(raw_embeddings, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edges)


def reconstruction_loss(
    z: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    negative_ratio: float,
    align_weight: float,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return (1.0 - F.cosine_similarity(z, x).mean()) * align_weight

    num_neg_samples = max(1, int(edge_index.size(1) * negative_ratio))
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=z.size(0),
        num_neg_samples=num_neg_samples,
        method="sparse",
    )

    pos_logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    align_loss = 1.0 - F.cosine_similarity(z, x).mean()
    return pos_loss + neg_loss + align_weight * align_loss


def train_gnn(config: BuildConfig) -> TrainingSummary:
    bundle = load_graph_bundle(config.artifacts_dir)
    raw_embeddings = bundle["raw_embeddings"]
    edge_index = bundle["edge_index"]
    data = build_pyg_data(raw_embeddings, edge_index)

    input_dim = data.x.size(-1)
    hidden_dim = config.gnn_hidden_dim or input_dim
    model = ResidualGCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=config.gnn_dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gnn_lr,
        weight_decay=config.gnn_weight_decay,
    )

    best_loss = float("inf")
    last_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(config.gnn_epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        loss = reconstruction_loss(
            z,
            data.x,
            data.edge_index,
            negative_ratio=config.gnn_negative_ratio,
            align_weight=config.gnn_align_weight,
        )
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        history.append({"epoch": epoch + 1, "loss": last_loss})
        if last_loss < best_loss:
            best_loss = last_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        structural_embeddings = model(data.x, data.edge_index).cpu().numpy().astype(np.float32)

    np.save(config.artifacts_dir / "structural_embeddings.npy", structural_embeddings)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
        },
        config.artifacts_dir / "gnn_checkpoint.pt",
    )
    (config.artifacts_dir / "gnn_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    summary = TrainingSummary(
        epochs=config.gnn_epochs,
        final_loss=last_loss,
        best_loss=best_loss,
        hidden_dim=hidden_dim,
        dropout=config.gnn_dropout,
    )
    (config.artifacts_dir / "gnn_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def ensure_structural_embeddings(config: BuildConfig) -> Path:
    target = config.artifacts_dir / "structural_embeddings.npy"
    if target.exists():
        return target
    train_gnn(config)
    return target
