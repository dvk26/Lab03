from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from lab03.dataset_utils import MedicalRecord
from lab03.retriever import HybridGraphRetriever


def evaluate_retriever(
    retriever: HybridGraphRetriever,
    records: Sequence[MedicalRecord],
    limit: int | None = None,
) -> dict:
    subset = list(records[:limit]) if limit else list(records)
    hits = 0
    reciprocal_ranks: list[float] = []
    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "hits": 0.0, "rr_sum": 0.0}
    )

    for record in subset:
        ranked = retriever.retrieve(record.question)
        rank = None
        for idx, item in enumerate(ranked, start=1):
            if item.node.metadata.get("doc_id") == record.doc_id:
                rank = idx
                break

        question_stats = by_type[record.question_type]
        question_stats["count"] += 1
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            question_stats["hits"] += 1
            question_stats["rr_sum"] += 1.0 / rank
        else:
            reciprocal_ranks.append(0.0)

    total = max(1, len(subset))
    summary = {
        "num_queries": len(subset),
        "hit_rate_at_k": hits / total,
        "mrr_at_k": sum(reciprocal_ranks) / total,
        "by_question_type": {},
    }

    for question_type, stats in sorted(by_type.items()):
        count = max(1.0, stats["count"])
        summary["by_question_type"][question_type] = {
            "count": int(stats["count"]),
            "hit_rate_at_k": stats["hits"] / count,
            "mrr_at_k": stats["rr_sum"] / count,
        }

    return summary


def save_evaluation_report(report: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
