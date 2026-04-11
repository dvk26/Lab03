from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download

from lab03.config import BuildConfig

DEFAULT_SPACE_MODEL_FILENAMES = (
    "Qwen3.5-4B.Q3_K_S.gguf",
    "Qwen3.5-4B.Q2_K.gguf",
    "Qwen3.5-4B.Q4_K_M.gguf",
)


def strip_reasoning_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def model_filename_candidates(config: BuildConfig) -> list[str]:
    if os.getenv("MODEL_FILENAME"):
        return [config.model_filename]

    candidates = [config.model_filename]
    if os.getenv("SPACE_ID") and config.model_repo == "Jackrong/Qwen3.5-4B-Neo-GGUF":
        candidates.extend(DEFAULT_SPACE_MODEL_FILENAMES)

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def ensure_allowed_model(config: BuildConfig, model_filename: str) -> Path:
    local_path = config.models_dir.resolve() / model_filename
    if local_path.exists():
        return local_path
    downloaded = hf_hub_download(
        repo_id=config.model_repo,
        filename=model_filename,
        local_dir=str(config.models_dir.resolve()),
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN"),
    )
    return Path(downloaded).resolve()


class LlamaCppGenerator:
    def __init__(self, config: BuildConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise ImportError(
                "llama-cpp-python is not installed. On Windows this may require C/C++ build tools. "
                "The Hugging Face Space target environment remains the primary deployment environment."
            ) from exc

        load_errors: list[str] = []
        self.model_filename: str | None = None
        self.model_path: Path | None = None

        for model_filename in model_filename_candidates(config):
            try:
                model_path = ensure_allowed_model(config, model_filename)
                self.client = Llama(
                    model_path=str(model_path),
                    n_ctx=config.llm_context_window,
                    n_threads=config.llm_threads,
                    n_batch=min(config.llm_batch_size, config.llm_context_window),
                    n_gpu_layers=0,
                    chat_format="chatml",
                    verbose=config.llm_verbose,
                )
                self.model_filename = model_filename
                self.model_path = model_path
                break
            except Exception as exc:  # pragma: no cover - depends on runtime resources
                load_errors.append(f"{model_filename}: {exc}")
        else:
            attempted = "\n".join(load_errors) if load_errors else "No model candidates were tried."
            raise RuntimeError(
                "Unable to initialize llama-cpp with the configured GGUF model.\n"
                f"Repository: {config.model_repo}\n"
                f"Attempted files:\n{attempted}\n\n"
                "On Hugging Face Spaces, prefer a smaller quant, keep Python 3.10, "
                "and reduce LLM_CONTEXT_WINDOW / LLM_BATCH_SIZE if memory is tight."
            )

    def _build_context(self, retrieved_nodes: Sequence) -> str:
        parts = []
        for index, node in enumerate(retrieved_nodes, start=1):
            metadata = dict(node.node.metadata)
            parts.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"Condition: {metadata.get('condition', '')}",
                        f"Question Type: {metadata.get('question_type', '')}",
                        node.node.text,
                    ]
                )
            )
        return "\n\n".join(parts)

    def generate_answer(self, question: str, retrieved_nodes: Sequence) -> str:
        context = self._build_context(retrieved_nodes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious healthcare QA assistant for an academic demo. "
                    "Answer strictly from the provided context. "
                    "If the context is insufficient, say that clearly. "
                    "Do not invent medical facts, diagnoses, or treatments. "
                    "Do not reveal chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved Context:\n{context}\n\n"
                    "Write a concise answer grounded in the context. "
                    "After the answer, add a short 'Evidence' section with source labels like [Context 1]."
                ),
            },
        ]
        response = self.client.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        content = response["choices"][0]["message"]["content"]
        return strip_reasoning_blocks(content)
