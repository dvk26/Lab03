from __future__ import annotations

import json

from lab03.config import BuildConfig
from lab03.dataset_utils import MedicalRecord
from lab03.evaluation import evaluate_retriever, save_evaluation_report
from lab03.retriever import load_runtime_retriever


def load_records(records_path) -> list[MedicalRecord]:
    records: list[MedicalRecord] = []
    with open(records_path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(MedicalRecord(**payload))
    return records


def main() -> None:
    config = BuildConfig()
    config.ensure_dirs()
    retriever = load_runtime_retriever(config)
    records = load_records(config.artifacts_dir / "records.jsonl")
    report = evaluate_retriever(retriever, records, limit=config.eval_limit)
    save_evaluation_report(report, config.evaluation_dir / "retrieval_metrics.json")
    print(report)


if __name__ == "__main__":
    main()

