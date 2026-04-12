from __future__ import annotations

import json

import gradio as gr

from lab03.config import BuildConfig
from lab03.llm import LlamaCppGenerator
from lab03.retriever import HybridGraphRetriever


class Runtime:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config
        self._retriever: HybridGraphRetriever | None = None
        self._generator: LlamaCppGenerator | None = None

    def ensure_artifacts(self) -> None:
        required = [
            self.config.artifacts_dir / "graph_nodes.json",
            self.config.artifacts_dir / "graph_edges.json",
            self.config.artifacts_dir / "manifest.json",
            self.config.artifacts_dir / "raw_embeddings.npy",
            self.config.artifacts_dir / "edge_index.npy",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing artifacts. Run `python scripts/build_all.py` locally and push the generated "
                "`artifacts/` files to the Space repo.\n"
                + "\n".join(missing)
            )

    @property
    def retriever(self) -> HybridGraphRetriever:
        if self._retriever is None:
            self.ensure_artifacts()
            self._retriever = HybridGraphRetriever.from_artifacts(
                artifacts_dir=self.config.artifacts_dir,
                embed_model_name=self.config.embed_model_name,
                alpha=self.config.alpha,
                top_k=self.config.top_k,
            )
        return self._retriever

    @property
    def generator(self) -> LlamaCppGenerator:
        if self._generator is None:
            self._generator = LlamaCppGenerator(self.config)
        return self._generator


def format_contexts(results: list[dict]) -> str:
    parts = []
    for index, result in enumerate(results, start=1):
        node = result["node"]
        parts.append(
            "\n".join(
                [
                    f"### Context {index}",
                    f"- Condition: {node.get('condition', '')}",
                    f"- Question type: {node.get('question_type', '')}",
                    f"- Raw score: {result['raw_score']:.4f}",
                    f"- Structural score: {result['structural_score']:.4f}",
                    f"- Final score: {result['final_score']:.4f}",
                    "",
                    node.get("text", ""),
                ]
            )
        )
    return "\n\n".join(parts)


def build_demo() -> gr.Blocks:
    config = BuildConfig()
    runtime = Runtime(config)

    def answer_question(question: str, alpha: float, top_k: int):
        if not question.strip():
            return "Please enter a question.", "", "{}"

        diagnostics: list[dict] = []
        try:
            runtime.retriever.set_runtime_params(alpha=alpha, top_k=top_k)
            diagnostics = runtime.retriever.retrieve_with_diagnostics(question)
            retrieved_nodes = runtime.retriever.results_to_nodes(diagnostics)
            answer = runtime.generator.generate_answer(question, retrieved_nodes)
            diagnostics_json = json.dumps(diagnostics, indent=2, ensure_ascii=False)
            return answer, format_contexts(diagnostics), diagnostics_json
        except Exception as exc:
            diagnostics_json = json.dumps(
                {"error": str(exc), "alpha": alpha, "top_k": top_k},
                indent=2,
                ensure_ascii=False,
            )
            context_text = format_contexts(diagnostics) if diagnostics else ""
            return f"Runtime error: {exc}", context_text, diagnostics_json

    with gr.Blocks(title="Lab03 GNN GraphRAG") as demo:
        gr.Markdown(
            """
            # Lab03: GNN-based GraphRAG for Healthcare QA
            This Space loads a healthcare property graph, refines node vectors with a GCN,
            retrieves with a hybrid raw + structural score, and generates answers with a
            quantized Qwen GGUF model running through `llama-cpp`.
            """
        )

        question = gr.Textbox(
            label="Question",
            placeholder="Example: What are the symptoms of Lyme disease?",
            lines=3,
        )
        with gr.Row():
            alpha = gr.Slider(
                label="Raw embedding weight (alpha)",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=config.alpha,
            )
            top_k = gr.Slider(
                label="Top-K contexts",
                minimum=1,
                maximum=10,
                step=1,
                value=config.top_k,
            )

        submit = gr.Button("Run Retrieval + Generation")
        answer = gr.Markdown(label="Answer")
        contexts = gr.Markdown(label="Retrieved Contexts")
        diagnostics = gr.Code(label="Retriever Diagnostics", language="json")

        submit.click(
            fn=answer_question,
            inputs=[question, alpha, top_k],
            outputs=[answer, contexts, diagnostics],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(show_error=True)
