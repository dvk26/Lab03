from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
