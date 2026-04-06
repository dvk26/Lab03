from __future__ import annotations

from lab03.config import BuildConfig
from lab03.dataset_utils import load_medical_records
from lab03.graph_pipeline import build_base_artifacts


def main() -> None:
    config = BuildConfig()
    config.ensure_dirs()
    records = load_medical_records(config)
    manifest = build_base_artifacts(records, config)
    print("Built base artifacts.")
    print(manifest)


if __name__ == "__main__":
    main()

