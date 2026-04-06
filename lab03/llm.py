from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download

from lab03.config import BuildConfig


def strip_reasoning_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def ensure_allowed_model(config: BuildConfig) -> Path:
    local_path = config.models_dir / config.model_filename
    if local_path.exists():
        return local_path
    downloaded = hf_hub_download(
        repo_id=config.model_repo,
        filename=config.model_filename,
        local_dir=config.models_dir,
    )
    return Path(downloaded)


class LlamaCppGenerator:
    def __init__(self, config: BuildConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise ImportError(
                "llama-cpp-python is not installed. On Windows this may require C/C++ build tools. "
                "The Hugging Face Space target environment remains the primary deployment environment."
            ) from exc

        model_path = ensure_allowed_model(config)
        threads = max(1, (os.cpu_count() or 2) - 1)
        self.client = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=threads,
            n_batch=256,
            chat_format="chatml",
            verbose=False,
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
