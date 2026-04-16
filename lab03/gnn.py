from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling

from lab03.artifact_store import load_graph_bundle
from lab03.config import BuildConfig


@dataclass(slots=True)
class TrainingSummary:
    epochs: int
    final_loss: float
    best_loss: float
    hidden_dim: int
    dropout: float
    num_node_types: int
    num_relation_types: int


def _derive_node_type_ids(nodes: list[dict]) -> np.ndarray:
    node_types = [str(node.get("node_type", "unknown")) for node in nodes]
    node_type_to_id = {node_type: idx for idx, node_type in enumerate(sorted(set(node_types)))}
    return np.asarray([node_type_to_id[node_type] for node_type in node_types], dtype=np.int64)


def _derive_edge_type_ids(edges: list[dict], num_edges: int) -> np.ndarray:
    if not edges:
        return np.zeros(num_edges, dtype=np.int64)
    edge_types = [str(edge.get("label", "RELATED_TO")) for edge in edges]
    edge_type_to_id = {edge_type: idx for idx, edge_type in enumerate(sorted(set(edge_types)))}
    return np.asarray([edge_type_to_id[edge_type] for edge_type in edge_types], dtype=np.int64)


class RelationAwareRGCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_node_types: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, input_dim)
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, input_dim, num_relations=num_relations)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.relation_embedding = nn.Embedding(num_relations, input_dim)
        self.dropout = dropout

        nn.init.xavier_uniform_(self.node_type_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        typed_input = x + self.node_type_embedding(node_type)
        hidden = self.conv1(typed_input, edge_index, edge_type)
        hidden = self.norm1(hidden)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.conv2(hidden, edge_index, edge_type)
        hidden = self.norm2(hidden)
        return F.normalize(x + hidden, p=2, dim=-1)

    def score_edges(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        relation = self.relation_embedding(edge_type)
        return (embeddings[edge_index[0]] * relation * embeddings[edge_index[1]]).sum(dim=-1)


def build_pyg_data(
    raw_embeddings: np.ndarray,
    edge_index: np.ndarray,
    node_type_ids: np.ndarray,
    edge_type_ids: np.ndarray,
) -> Data:
    x = torch.tensor(raw_embeddings, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)
    node_type = torch.tensor(node_type_ids, dtype=torch.long)
    edge_type = torch.tensor(edge_type_ids, dtype=torch.long)
    return Data(x=x, edge_index=edges, node_type=node_type, edge_type=edge_type)


def reconstruction_loss(
    model: RelationAwareRGCNEncoder,
    embeddings: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    negative_ratio: float,
    align_weight: float,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return (1.0 - F.cosine_similarity(embeddings, x).mean()) * align_weight

    num_neg_samples = max(1, int(edge_index.size(1) * negative_ratio))
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=embeddings.size(0),
        num_neg_samples=num_neg_samples,
        method="sparse",
    )

    if edge_type.numel() == 0:
        neg_edge_type = torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=edge_index.device)
    else:
        sampled_indices = torch.randint(
            edge_type.size(0),
            (neg_edge_index.size(1),),
            device=edge_index.device,
        )
        neg_edge_type = edge_type[sampled_indices]

    pos_logits = model.score_edges(embeddings, edge_index, edge_type)
    neg_logits = model.score_edges(embeddings, neg_edge_index, neg_edge_type)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    align_loss = 1.0 - F.cosine_similarity(embeddings, x).mean()
    return pos_loss + neg_loss + align_weight * align_loss


def train_gnn(config: BuildConfig) -> TrainingSummary:
    bundle = load_graph_bundle(config.artifacts_dir)
    raw_embeddings = bundle["raw_embeddings"]
    edge_index = bundle["edge_index"]

    node_type_ids = bundle.get("node_type_ids")
    if node_type_ids is None:
        node_type_ids = _derive_node_type_ids(bundle["nodes"])

    edge_type_ids = bundle.get("edge_type_ids")
    if edge_type_ids is None:
        edge_type_ids = _derive_edge_type_ids(bundle.get("edges", []), edge_index.shape[1])

    data = build_pyg_data(raw_embeddings, edge_index, node_type_ids, edge_type_ids)

    input_dim = data.x.size(-1)
    hidden_dim = config.gnn_hidden_dim or input_dim
    num_node_types = max(1, int(data.node_type.max().item()) + 1) if data.node_type.numel() else 1
    num_relations = max(1, int(data.edge_type.max().item()) + 1) if data.edge_type.numel() else 1

    model = RelationAwareRGCNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        num_node_types=num_node_types,
        dropout=config.gnn_dropout,
    )
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
        embeddings = model(data.x, data.edge_index, data.edge_type, data.node_type)
        loss = reconstruction_loss(
            model,
            embeddings,
            data.x,
            data.edge_index,
            data.edge_type,
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
        structural_embeddings = (
            model(data.x, data.edge_index, data.edge_type, data.node_type)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    np.save(config.artifacts_dir / "structural_embeddings.npy", structural_embeddings)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_node_types": num_node_types,
            "num_relations": num_relations,
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
        num_node_types=num_node_types,
        num_relation_types=num_relations,
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
