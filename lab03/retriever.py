from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from llama_index.core.base.base_retriever import BaseRetriever
except ImportError:  # pragma: no cover - version fallback
    from llama_index.core.retrievers import BaseRetriever  # type: ignore

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from lab03.artifact_store import load_graph_bundle
from lab03.config import BuildConfig


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _load_embedder(model_name: str):
    """Load fastembed if available (no torch needed), otherwise fall back to sentence-transformers."""
    fastembed_error: Exception | None = None
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name)
    except Exception as exc:
        fastembed_error = exc

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise ImportError(
            "No compatible query embedding backend is available. Install `fastembed` or "
            "fix the local `sentence-transformers` / `transformers` versions."
        ) from exc if fastembed_error is None else exc


def _embed(embedder, text: str) -> np.ndarray:
    """Encode a single query string using whichever embedder was loaded."""
    try:
        from fastembed import TextEmbedding
        if isinstance(embedder, TextEmbedding):
            vec = np.array(list(embedder.embed([text]))[0], dtype=np.float32)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            return vec
    except Exception:
        pass
    # sentence-transformers path
    return embedder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)


class HybridGraphRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[dict],
        raw_embeddings: np.ndarray,
        structural_embeddings: np.ndarray,
        embed_model_name: str,
        alpha: float = 0.65,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.raw_embeddings = _normalize_rows(raw_embeddings.astype(np.float32))
        self.structural_embeddings = _normalize_rows(structural_embeddings.astype(np.float32))
        self.embedder = _load_embedder(embed_model_name)
        self.alpha = alpha
        self.top_k = top_k
        self.candidate_indices = [
            idx for idx, node in enumerate(nodes) if node.get("node_type") == "chunk"
        ]

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path,
        embed_model_name: str,
        alpha: float = 0.65,
        top_k: int = 5,
    ) -> "HybridGraphRetriever":
        bundle = load_graph_bundle(artifacts_dir)
        structural_path = artifacts_dir / "structural_embeddings.npy"
        if structural_path.exists():
            structural_embeddings = np.load(structural_path)
        else:
            structural_embeddings = bundle["raw_embeddings"]
        return cls(
            nodes=bundle["nodes"],
            raw_embeddings=bundle["raw_embeddings"],
            structural_embeddings=structural_embeddings,
            embed_model_name=embed_model_name,
            alpha=alpha,
            top_k=top_k,
        )

    def set_runtime_params(self, alpha: float | None = None, top_k: int | None = None) -> None:
        if alpha is not None:
            self.alpha = float(alpha)
        if top_k is not None:
            self.top_k = int(top_k)

    def _score_query(self, query: str) -> list[dict]:
        query_vector = _embed(self.embedder, query)

        raw_scores = self.raw_embeddings[self.candidate_indices] @ query_vector
        structural_scores = self.structural_embeddings[self.candidate_indices] @ query_vector
        final_scores = self.alpha * raw_scores + (1.0 - self.alpha) * structural_scores

        ranking = np.argsort(-final_scores)[: self.top_k]
        results: list[dict] = []
        for position in ranking:
            node_index = self.candidate_indices[int(position)]
            node = self.nodes[node_index]
            results.append(
                {
                    "node": node,
                    "raw_score": float(raw_scores[position]),
                    "structural_score": float(structural_scores[position]),
                    "final_score": float(final_scores[position]),
                }
            )
        return results

    def retrieve_with_diagnostics(self, query: str) -> list[dict]:
        return self._score_query(query)

    def results_to_nodes(self, results: list[dict]) -> list[NodeWithScore]:
        nodes: list[NodeWithScore] = []
        for result in results:
            node = result["node"]
            text_node = TextNode(
                id_=node["node_id"],
                text=node["text"],
                metadata={
                    "doc_id": node.get("doc_id", ""),
                    "condition": node.get("condition", ""),
                    "question_type": node.get("question_type", ""),
                    "question": node.get("question", ""),
                    "answer": node.get("answer", ""),
                    "raw_score": result["raw_score"],
                    "structural_score": result["structural_score"],
                    "final_score": result["final_score"],
                },
            )
            nodes.append(NodeWithScore(node=text_node, score=result["final_score"]))
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        results = self._score_query(query)
        return self.results_to_nodes(results)


def load_runtime_retriever(config: BuildConfig) -> HybridGraphRetriever:
    return HybridGraphRetriever.from_artifacts(
        artifacts_dir=config.artifacts_dir,
        embed_model_name=config.embed_model_name,
        alpha=config.alpha,
        top_k=config.top_k,
    )
