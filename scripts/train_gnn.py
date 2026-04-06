from __future__ import annotations

from lab03.config import BuildConfig
from lab03.gnn import train_gnn


def main() -> None:
    config = BuildConfig()
    config.ensure_dirs()
    summary = train_gnn(config)
    print("Trained GNN.")
    print(summary)


if __name__ == "__main__":
    main()

