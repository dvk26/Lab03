from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

from lab03.config import BuildConfig
from lab03.dataset_utils import MedicalRecord, records_to_documents
from lab03.graph_extractor import MedicalGraphExtractor, relation_label_for, split_sentences

try:
    from llama_index.core.indices import PropertyGraphIndex
except ImportError:  # pragma: no cover - version fallback
    from llama_index.core import PropertyGraphIndex  # type: ignore


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def _chunk_node_id(doc_id: str) -> str:
    return f"chunk::{doc_id}"


def _condition_node_id(condition: str) -> str:
    return f"condition::{_safe_id(condition)}"


def _aspect_node_id(question_type: str) -> str:
    return f"aspect::{question_type.lower()}"


def _sentence_node_id(doc_id: str, sentence_index: int) -> str:
    return f"sentence::{doc_id}::{sentence_index}"


def _statement_sort_key(sentence_id: str) -> tuple[str, int]:
    prefix = "sentence::"
    stripped = sentence_id[len(prefix) :] if sentence_id.startswith(prefix) else sentence_id
    doc_id, _, raw_index = stripped.rpartition("::")
    return doc_id, int(raw_index or 0)


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
    persist_dir = config.property_graph_dir
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(str(persist_dir))
    return persist_dir


def ensure_property_graph(records: Sequence[MedicalRecord], config: BuildConfig) -> Path:
    property_graph_path = config.property_graph_dir / "property_graph_store.json"
    if property_graph_path.exists():
        return config.property_graph_dir
    return build_property_graph_index(records, config)


def _add_node(
    graph: nx.DiGraph,
    node_id: str,
    node_type: str,
    text: str,
    **metadata: str,
) -> None:
    payload = {"node_type": node_type, "text": text}
    payload.update(metadata)

    if graph.has_node(node_id):
        existing = graph.nodes[node_id]
        for key, value in payload.items():
            if value not in {None, ""} and key not in existing:
                existing[key] = value
        return

    graph.add_node(node_id, **payload)


def _add_edge(graph: nx.DiGraph, source: str, target: str, label: str) -> None:
    if graph.has_edge(source, target):
        return
    graph.add_edge(source, target, label=label)


def build_structured_graph(records: Sequence[MedicalRecord], config: BuildConfig) -> nx.DiGraph:
    graph = nx.DiGraph()
    for record in records:
        condition_id = _condition_node_id(record.condition)
        aspect_id = _aspect_node_id(record.question_type)
        chunk_id = _chunk_node_id(record.doc_id)

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
        for sentence_index, sentence in enumerate(sentences):
            sentence_id = _sentence_node_id(record.doc_id, sentence_index)
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
            _add_edge(graph, aspect_id, sentence_id, "IN_SECTION")
            _add_edge(graph, condition_id, sentence_id, typed_relation)
            if previous_sentence_id:
                _add_edge(graph, previous_sentence_id, sentence_id, "NEXT_SENTENCE")
            previous_sentence_id = sentence_id
    return graph


def load_property_graph_store(property_graph_dir: Path) -> dict:
    path = property_graph_dir / "property_graph_store.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_structured_graph_from_property_store(
    property_graph_dir: Path,
    config: BuildConfig,
) -> nx.DiGraph:
    payload = load_property_graph_store(property_graph_dir)
    graph = nx.DiGraph()

    normalized_ids: dict[str, str] = {}
    chunk_metadata_by_doc: dict[str, dict[str, str]] = {}
    statement_nodes_by_doc: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    sentence_groups: dict[str, list[str]] = defaultdict(list)

    for raw_id, node in payload.get("nodes", {}).items():
        label = str(node.get("label", ""))
        properties = dict(node.get("properties", {}))

        if label == "text_chunk":
            doc_id = str(properties.get("doc_id", "")).strip()
            if not doc_id:
                continue
            normalized_ids[raw_id] = _chunk_node_id(doc_id)
            chunk_metadata_by_doc[doc_id] = {
                "doc_id": doc_id,
                "question": str(properties.get("question", "")).strip(),
                "answer": str(node.get("text", "")).strip(),
                "condition": str(properties.get("condition", "")).strip() or "Unknown Condition",
                "question_type": str(properties.get("question_type", "")).strip() or "overview",
            }
            continue

        if label == "CONDITION":
            condition = str(node.get("name") or raw_id).strip()
            if condition:
                normalized_ids[raw_id] = _condition_node_id(condition)
            continue

        if label == "ASPECT":
            question_type = str(properties.get("question_type") or node.get("name") or raw_id).strip()
            if question_type:
                normalized_ids[raw_id] = _aspect_node_id(question_type)
            continue

        if label == "STATEMENT":
            doc_id = str(properties.get("doc_id", "")).strip()
            if not doc_id:
                continue
            _, _, raw_index = str(raw_id).rpartition("::")
            sentence_index = int(raw_index or 0)
            normalized_ids[raw_id] = _sentence_node_id(doc_id, sentence_index)
            statement_nodes_by_doc[doc_id].append((raw_id, node))

    for raw_id, node in payload.get("nodes", {}).items():
        label = str(node.get("label", ""))
        properties = dict(node.get("properties", {}))

        if label == "text_chunk":
            doc_id = str(properties.get("doc_id", "")).strip()
            metadata = chunk_metadata_by_doc.get(doc_id)
            if not metadata:
                continue

            chunk_id = normalized_ids[raw_id]
            condition = metadata["condition"]
            question_type = metadata["question_type"]
            condition_id = _condition_node_id(condition)
            aspect_id = _aspect_node_id(question_type)

            _add_node(
                graph,
                condition_id,
                "condition",
                condition,
                condition=condition,
            )
            _add_node(
                graph,
                aspect_id,
                "aspect",
                question_type,
                question_type=question_type,
            )
            _add_node(
                graph,
                chunk_id,
                "chunk",
                metadata["answer"],
                doc_id=doc_id,
                question=metadata["question"],
                answer=metadata["answer"],
                condition=condition,
                question_type=question_type,
            )
            _add_edge(graph, condition_id, aspect_id, "HAS_ASPECT")
            _add_edge(graph, aspect_id, chunk_id, "DESCRIBES")
            _add_edge(graph, condition_id, chunk_id, "HAS_DOCUMENT")
            _add_edge(graph, chunk_id, condition_id, "ABOUT_CONDITION")
            _add_edge(graph, chunk_id, aspect_id, "IN_ASPECT")

            for statement_raw_id, statement_node in sorted(
                statement_nodes_by_doc.get(doc_id, []),
                key=lambda item: _statement_sort_key(normalized_ids[item[0]]),
            ):
                statement_properties = dict(statement_node.get("properties", {}))
                sentence_id = normalized_ids.get(statement_raw_id)
                if not sentence_id:
                    continue

                _add_node(
                    graph,
                    sentence_id,
                    "sentence",
                    str(statement_properties.get("text", "")).strip(),
                    doc_id=doc_id,
                    condition=condition,
                    question_type=str(statement_properties.get("question_type", question_type)).strip()
                    or question_type,
                )
                _add_edge(graph, chunk_id, sentence_id, "HAS_SENTENCE")
                _add_edge(graph, sentence_id, chunk_id, "PART_OF")
                sentence_groups[doc_id].append(sentence_id)

    for raw_id, node in payload.get("nodes", {}).items():
        label = str(node.get("label", ""))
        properties = dict(node.get("properties", {}))

        if label == "CONDITION":
            condition = str(node.get("name") or raw_id).strip()
            if condition:
                _add_node(
                    graph,
                    normalized_ids[raw_id],
                    "condition",
                    condition,
                    condition=condition,
                )
            continue

        if label == "ASPECT":
            question_type = str(properties.get("question_type") or node.get("name") or raw_id).strip()
            if question_type:
                _add_node(
                    graph,
                    normalized_ids[raw_id],
                    "aspect",
                    question_type,
                    question_type=question_type.lower(),
                )

    for relation in payload.get("relations", {}).values():
        source = normalized_ids.get(str(relation.get("source_id", "")))
        target = normalized_ids.get(str(relation.get("target_id", "")))
        label = str(relation.get("label", "RELATED_TO"))
        if not source or not target:
            continue
        _add_edge(graph, source, target, label)

    for sentence_ids in sentence_groups.values():
        previous_sentence_id: str | None = None
        for sentence_id in sorted(sentence_ids, key=_statement_sort_key):
            if previous_sentence_id:
                _add_edge(graph, previous_sentence_id, sentence_id, "NEXT_SENTENCE")
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


def augment_edges_for_pyg(edges: Sequence[dict]) -> list[dict]:
    augmented: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        label = edge.get("label", "RELATED_TO")

        direct_key = (source, target, label)
        if direct_key not in seen:
            augmented.append({"source": source, "target": target, "label": label})
            seen.add(direct_key)

        reverse_label = f"REV_{label}"
        reverse_key = (target, source, reverse_label)
        if reverse_key not in seen:
            augmented.append(
                {
                    "source": target,
                    "target": source,
                    "label": reverse_label,
                }
            )
            seen.add(reverse_key)

    return augmented


def build_edge_index(nodes: Sequence[dict], edges: Sequence[dict]) -> np.ndarray:
    node_to_idx = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    pairs: list[tuple[int, int]] = []
    for edge in edges:
        source = node_to_idx[edge["source"]]
        target = node_to_idx[edge["target"]]
        pairs.append((source, target))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).T


def build_node_type_ids(nodes: Sequence[dict]) -> tuple[np.ndarray, dict[str, int]]:
    node_types = [str(node.get("node_type", "unknown")) for node in nodes]
    node_type_to_id = {node_type: idx for idx, node_type in enumerate(sorted(set(node_types)))}
    node_type_ids = np.asarray([node_type_to_id[node_type] for node_type in node_types], dtype=np.int64)
    return node_type_ids, node_type_to_id


def build_edge_type_ids(edges: Sequence[dict]) -> tuple[np.ndarray, dict[str, int]]:
    edge_types = [str(edge.get("label", "RELATED_TO")) for edge in edges]
    edge_type_to_id = {edge_type: idx for idx, edge_type in enumerate(sorted(set(edge_types)))}
    edge_type_ids = np.asarray([edge_type_to_id[edge_type] for edge_type in edge_types], dtype=np.int64)
    return edge_type_ids, edge_type_to_id


def _load_embedder(model_name: str):
    fastembed_error: Exception | None = None
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        fastembed_error = exc

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise ImportError(
            "No compatible embedding backend is available. Install `fastembed` or "
            "fix the local `sentence-transformers` / `transformers` versions."
        ) from exc if fastembed_error is None else exc


def encode_nodes(nodes: Sequence[dict], model_name: str) -> np.ndarray:
    texts = [node["text"] for node in nodes]
    embedder = _load_embedder(model_name)

    try:
        from fastembed import TextEmbedding

        if isinstance(embedder, TextEmbedding):
            embeddings = np.asarray(list(embedder.embed(texts)), dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            return (embeddings / norms).astype(np.float32)
    except Exception:
        pass

    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def load_cached_embeddings_if_compatible(nodes: Sequence[dict], artifacts_dir: Path) -> np.ndarray | None:
    nodes_path = artifacts_dir / "graph_nodes.json"
    embeddings_path = artifacts_dir / "raw_embeddings.npy"
    if not nodes_path.exists() or not embeddings_path.exists():
        return None

    cached_nodes = load_json(nodes_path)
    cached_node_ids = [node["node_id"] for node in cached_nodes]
    current_node_ids = [node["node_id"] for node in nodes]
    if cached_node_ids != current_node_ids:
        return None

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(nodes):
        return None
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
    nodes, base_edges = graph_to_serializable(graph)
    edges = augment_edges_for_pyg(base_edges)
    edge_index = build_edge_index(nodes, edges)
    node_type_ids, node_type_to_id = build_node_type_ids(nodes)
    edge_type_ids, edge_type_to_id = build_edge_type_ids(edges)

    chunk_node_ids = [node["node_id"] for node in nodes if node.get("node_type") == "chunk"]
    manifest = {
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "num_records": len(records),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_node_types": len(node_type_to_id),
        "num_relation_types": len(edge_type_to_id),
        "embed_model_name": config.embed_model_name,
        "chunk_node_ids": chunk_node_ids,
        "property_graph_dir": str(config.property_graph_dir.as_posix()),
        "node_type_to_id": node_type_to_id,
        "edge_type_to_id": edge_type_to_id,
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
    np.save(config.artifacts_dir / "node_type_ids.npy", node_type_ids)
    np.save(config.artifacts_dir / "edge_type_ids.npy", edge_type_ids)
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

    bundle = {
        "nodes": nodes,
        "edges": edges,
        "manifest": manifest,
        "raw_embeddings": raw_embeddings,
        "edge_index": edge_index,
    }

    node_type_path = artifacts_dir / "node_type_ids.npy"
    if node_type_path.exists():
        bundle["node_type_ids"] = np.load(node_type_path)

    edge_type_path = artifacts_dir / "edge_type_ids.npy"
    if edge_type_path.exists():
        bundle["edge_type_ids"] = np.load(edge_type_path)

    return bundle


def build_base_artifacts(records: Sequence[MedicalRecord], config: BuildConfig) -> dict:
    config.ensure_dirs()
    property_graph_dir = ensure_property_graph(records, config)

    property_graph_path = property_graph_dir / "property_graph_store.json"
    if property_graph_path.exists():
        graph = build_structured_graph_from_property_store(property_graph_dir, config)
    else:
        graph = build_structured_graph(records, config)

    nodes, _ = graph_to_serializable(graph)
    try:
        raw_embeddings = encode_nodes(nodes, config.embed_model_name)
    except Exception:
        cached_embeddings = load_cached_embeddings_if_compatible(nodes, config.artifacts_dir)
        if cached_embeddings is None:
            raise
        raw_embeddings = cached_embeddings
    manifest = save_graph_bundle(records, graph, raw_embeddings, config)
    return manifest
