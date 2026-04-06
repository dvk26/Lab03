from __future__ import annotations

from lab03.config import BuildConfig
from lab03.dataset_utils import load_medical_records
from lab03.evaluation import evaluate_retriever, save_evaluation_report
from lab03.gnn import train_gnn
from lab03.graph_pipeline import build_base_artifacts
from lab03.retriever import load_runtime_retriever


def main() -> None:
    config = BuildConfig()
    config.ensure_dirs()

    records = load_medical_records(config)
    build_base_artifacts(records, config)
    train_gnn(config)

    retriever = load_runtime_retriever(config)
    report = evaluate_retriever(retriever, records, limit=config.eval_limit)
    save_evaluation_report(report, config.evaluation_dir / "retrieval_metrics.json")

    print("Completed build, training, and retrieval evaluation.")
    print(report)


if __name__ == "__main__":
    main()

