from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _is_space_runtime() -> bool:
    return bool(os.getenv("SPACE_ID"))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_model_filename() -> str:
    if _is_space_runtime():
        return "Qwen3.5-4B.Q3_K_S.gguf"
    return "Qwen3.5-4B.Q4_K_M.gguf"


def _default_llm_context_window() -> int:
    if _is_space_runtime():
        return 2048
    return 4096


def _default_llm_batch_size() -> int:
    if _is_space_runtime():
        return 64
    return 256


def _default_llm_threads() -> int:
    cpu_count = os.cpu_count() or 2
    if _is_space_runtime():
        return min(4, max(1, cpu_count - 1))
    return max(1, cpu_count - 1)


@dataclass(slots=True)
class BuildConfig:
    dataset_id: str = os.getenv("DATASET_ID", "Tonic/medquad")
    dataset_split: str = os.getenv("DATASET_SPLIT", "train")
    max_records: int = _env_int("MAX_RECORDS", 1500)
    sample_seed: int = _env_int("SAMPLE_SEED", 7)
    embed_model_name: str = os.getenv(
        "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", "storage"))
    evaluation_dir: Path = Path(os.getenv("EVALUATION_DIR", "evaluation"))
    model_repo: str = os.getenv("MODEL_REPO", "Jackrong/Qwen3.5-4B-Neo-GGUF")
    model_filename: str = os.getenv("MODEL_FILENAME", _default_model_filename())
    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))
    llm_context_window: int = _env_int("LLM_CONTEXT_WINDOW", _default_llm_context_window())
    llm_batch_size: int = _env_int("LLM_BATCH_SIZE", _default_llm_batch_size())
    llm_threads: int = _env_int("LLM_THREADS", _default_llm_threads())
    llm_verbose: bool = _env_bool("LLM_VERBOSE", False)
    alpha: float = _env_float("RETRIEVAL_ALPHA", 0.65)
    top_k: int = _env_int("RETRIEVAL_TOP_K", 5)
    max_sentences_per_doc: int = _env_int("MAX_SENTENCES_PER_DOC", 5)
    min_sentence_chars: int = _env_int("MIN_SENTENCE_CHARS", 35)
    eval_limit: int = _env_int("EVAL_LIMIT", 200)
    gnn_hidden_dim: int = _env_int("GNN_HIDDEN_DIM", 384)
    gnn_epochs: int = _env_int("GNN_EPOCHS", 60)
    gnn_lr: float = _env_float("GNN_LR", 0.001)
    gnn_weight_decay: float = _env_float("GNN_WEIGHT_DECAY", 0.0001)
    gnn_dropout: float = _env_float("GNN_DROPOUT", 0.1)
    gnn_align_weight: float = _env_float("GNN_ALIGN_WEIGHT", 0.1)
    gnn_negative_ratio: float = _env_float("GNN_NEGATIVE_RATIO", 1.0)

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
