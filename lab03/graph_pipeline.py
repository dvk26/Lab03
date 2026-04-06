from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from lab03.config import BuildConfig
from lab03.dataset_utils import MedicalRecord, records_to_documents
from lab03.graph_extractor import MedicalGraphExtractor, relation_label_for, split_sentences

try:
    from llama_index.core.indices import PropertyGraphIndex
except ImportError:  # pragma: no cover - version fallback
    from llama_index.core import PropertyGraphIndex  # type: ignore


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def build_property_graph_index(records: Sequence[MedicalRecord], config: BuildConfig) -> Path:
    documents = records_to_documents(records)
    extractor = MedicalGraphExtractor(
        max_sentences=config.max_sentences_per_doc,
        min_sentence_chars=config.min_sentence_chars,
    )
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[extractor],
        embed_kg_nodes=False,
    )
    persist_dir = config.storage_dir / "property_graph"
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(str(persist_dir))
    return persist_dir


def _add_node(
    graph: nx.DiGraph,
    node_id: str,
    node_type: str,
    text: str,
    **metadata: str,
) -> None:
    if graph.has_node(node_id):
        return
    graph.add_node(node_id, node_type=node_type, text=text, **metadata)


def _add_edge(graph: nx.DiGraph, source: str, target: str, label: str) -> None:
    if graph.has_edge(source, target):
        return
    graph.add_edge(source, target, label=label)


def build_structured_graph(records: Sequence[MedicalRecord], config: BuildConfig) -> nx.DiGraph:
    graph = nx.DiGraph()
    for record in records:
        condition_slug = _safe_id(record.condition)
        condition_id = f"condition::{condition_slug}"
        aspect_id = f"aspect::{record.question_type}"
        chunk_id = f"chunk::{record.doc_id}"

        _add_node(
            graph,
            condition_id,
            "condition",
            record.condition,
            condition=record.condition,
        )
        _add_node(
            graph,
            aspect_id,
            "aspect",
            record.question_type,
            question_type=record.question_type,
        )
        _add_node(
            graph,
            chunk_id,
            "chunk",
            record.answer,
            doc_id=record.doc_id,
            question=record.question,
            answer=record.answer,
            condition=record.condition,
            question_type=record.question_type,
        )

        typed_relation = relation_label_for(record.question_type)
        _add_edge(graph, condition_id, aspect_id, "HAS_ASPECT")
        _add_edge(graph, aspect_id, chunk_id, "DESCRIBES")
        _add_edge(graph, condition_id, chunk_id, "HAS_DOCUMENT")
        _add_edge(graph, chunk_id, condition_id, "ABOUT_CONDITION")
        _add_edge(graph, chunk_id, aspect_id, "IN_ASPECT")

        sentences = split_sentences(
            record.answer,
            max_sentences=config.max_sentences_per_doc,
            min_chars=config.min_sentence_chars,
        )
        previous_sentence_id: str | None = None
        for index, sentence in enumerate(sentences):
            sentence_id = f"sentence::{record.doc_id}::{index}"
            _add_node(
                graph,
                sentence_id,
                "sentence",
                sentence,
                doc_id=record.doc_id,
                condition=record.condition,
                question_type=record.question_type,
            )
            _add_edge(graph, chunk_id, sentence_id, "HAS_SENTENCE")
            _add_edge(graph, sentence_id, chunk_id, "PART_OF")
            _add_edge(graph, aspect_id, sentence_id, "HAS_SECTION_SENTENCE")
            _add_edge(graph, condition_id, sentence_id, typed_relation)
            if previous_sentence_id:
                _add_edge(graph, previous_sentence_id, sentence_id, "NEXT_SENTENCE")
                _add_edge(graph, sentence_id, previous_sentence_id, "PREV_SENTENCE")
            previous_sentence_id = sentence_id
    return graph


def graph_to_serializable(graph: nx.DiGraph) -> tuple[list[dict], list[dict]]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        payload = {"node_id": node_id}
        payload.update(attrs)
        nodes.append(payload)

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": source,
                "target": target,
                "label": attrs.get("label", "RELATED_TO"),
            }
        )
    return nodes, edges


def build_edge_index(nodes: Sequence[dict], edges: Sequence[dict]) -> np.ndarray:
    node_to_idx = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    pairs: list[tuple[int, int]] = []
    for edge in edges:
        src = node_to_idx[edge["source"]]
        dst = node_to_idx[edge["target"]]
        pairs.append((src, dst))
        pairs.append((dst, src))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).T


def encode_nodes(nodes: Sequence[dict], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [node["text"] for node in nodes]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def save_records(records: Sequence[MedicalRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def save_graph_bundle(
    records: Sequence[MedicalRecord],
    graph: nx.DiGraph,
    raw_embeddings: np.ndarray,
    config: BuildConfig,
) -> dict:
    nodes, edges = graph_to_serializable(graph)
    edge_index = build_edge_index(nodes, edges)

    chunk_node_ids = [node["node_id"] for node in nodes if node.get("node_type") == "chunk"]
    manifest = {
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "num_records": len(records),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "embed_model_name": config.embed_model_name,
        "chunk_node_ids": chunk_node_ids,
        "property_graph_dir": str((config.storage_dir / "property_graph").as_posix()),
    }

    (config.artifacts_dir / "graph_nodes.json").write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.artifacts_dir / "graph_edges.json").write_text(
        json.dumps(edges, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.artifacts_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    np.save(config.artifacts_dir / "raw_embeddings.npy", raw_embeddings)
    np.save(config.artifacts_dir / "edge_index.npy", edge_index)
    save_records(records, config.artifacts_dir / "records.jsonl")
    return manifest


def load_json(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_graph_bundle(artifacts_dir: Path) -> dict:
    nodes = load_json(artifacts_dir / "graph_nodes.json")
    edges = load_json(artifacts_dir / "graph_edges.json")
    manifest = load_json(artifacts_dir / "manifest.json")
    raw_embeddings = np.load(artifacts_dir / "raw_embeddings.npy")
    edge_index = np.load(artifacts_dir / "edge_index.npy")
    return {
        "nodes": nodes,
        "edges": edges,
        "manifest": manifest,
        "raw_embeddings": raw_embeddings,
        "edge_index": edge_index,
    }


def build_base_artifacts(records: Sequence[MedicalRecord], config: BuildConfig) -> dict:
    config.ensure_dirs()
    build_property_graph_index(records, config)
    graph = build_structured_graph(records, config)
    nodes, _ = graph_to_serializable(graph)
    raw_embeddings = encode_nodes(nodes, config.embed_model_name)
    manifest = save_graph_bundle(records, graph, raw_embeddings, config)
    return manifest
