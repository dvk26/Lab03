---
title: lab03-gnn-graphrag
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# Lab 03: GNN-based GraphRAG for LLM Inference

This repository implements the full Lab 03 pipeline from the provided lab brief:

1. load a healthcare corpus from Hugging Face
2. build a LlamaIndex `PropertyGraphIndex`
3. export the graph to a PyTorch Geometric-compatible format
4. train a GNN that produces structure-aware node vectors
5. combine raw semantic similarity and GNN structural similarity in a custom `BaseRetriever`
6. generate the answer with an allowed quantized GGUF model running through `llama-cpp`
7. serve the pipeline through a Gradio app suitable for Hugging Face Spaces

## Dataset

The default dataset is `Tonic/medquad`. Each row contains an instruction-style medical question and an answer. The preprocessing step extracts:

- `question`
- `answer`
- `question_type`
- `condition`

This dataset is a good fit for the lab because it is healthcare-specific, manageable on free-tier hardware, and gives natural query-document pairs for retrieval evaluation.

## Implementation Summary

### Property graph construction

`lab03/graph_extractor.py` defines a deterministic `MedicalGraphExtractor` for LlamaIndex. For each document it creates:

- a `CONDITION` node
- an `ASPECT` node such as `SYMPTOMS`, `TREATMENT`, or `PREVENTION`
- `STATEMENT` nodes derived from answer sentences

This keeps graph construction reproducible on a Hugging Face Free Space and still satisfies the requirement to use `PropertyGraphIndex`.

### PyG graph export

`lab03/graph_pipeline.py` builds a heterogeneous graph with:

- `condition` nodes
- `aspect` nodes
- `chunk` nodes
- `sentence` nodes

Edges include:

- `HAS_ASPECT`
- `DESCRIBES`
- `HAS_DOCUMENT`
- `HAS_SENTENCE`
- typed condition-to-sentence relations such as `HAS_SYMPTOM_INFO`
- sentence order edges

The pipeline saves:

- `artifacts/graph_nodes.json`
- `artifacts/graph_edges.json`
- `artifacts/raw_embeddings.npy`
- `artifacts/edge_index.npy`
- `artifacts/records.jsonl`

### GNN model

`lab03/gnn.py` implements a residual 2-layer `GCNConv` encoder:

```text
h1 = GCN(x, edge_index)
h2 = GCN(h1, edge_index)
z  = normalize(x + h2)
```

Training uses unsupervised edge reconstruction with negative sampling plus an alignment term that keeps the updated vectors compatible with the original semantic space.

### Hybrid retriever

`lab03/retriever.py` subclasses `BaseRetriever` and uses:

```text
s_raw(i)    = cosine(q, x_i)
s_struct(i) = cosine(q, z_i)
s_final(i)  = alpha * s_raw(i) + (1 - alpha) * s_struct(i)
```

Only chunk nodes are returned to the generator, but structural message passing is learned over the full graph.

### Generator

`lab03/llm.py` loads an allowed GGUF model through `llama-cpp-python`. The default is:

- repository: `Jackrong/Qwen3.5-4B-Neo-GGUF`
- file: `Qwen3.5-4B.Q4_K_M.gguf`

## Project Structure

```text
app.py
requirements.txt
lab03/
  config.py
  dataset_utils.py
  graph_extractor.py
  graph_pipeline.py
  gnn.py
  retriever.py
  llm.py
  evaluation.py
scripts/
  build_artifacts.py
  train_gnn.py
  evaluate.py
  build_all.py
artifacts/
storage/
evaluation/
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: `llama-cpp-python` may need C/C++ build tools on a local Windows machine. The primary deployment target for this project is Hugging Face Spaces on Linux, and the repo includes `pre-requirements.txt` and `packages.txt` to make that build path more reliable.

Build the graph artifacts:

```bash
python scripts/build_artifacts.py
```

Train the GNN:

```bash
python scripts/train_gnn.py
```

Evaluate retrieval:

```bash
python scripts/evaluate.py
```

Launch the Gradio app:

```bash
python app.py
```

One-command build path:

```bash
python scripts/build_all.py
python app.py
```

## Environment Variables

You can override the defaults with:

- `DATASET_ID`
- `DATASET_SPLIT`
- `MAX_RECORDS`
- `EMBED_MODEL_NAME`
- `ARTIFACTS_DIR`
- `STORAGE_DIR`
- `MODEL_REPO`
- `MODEL_FILENAME`
- `RETRIEVAL_ALPHA`
- `RETRIEVAL_TOP_K`
- `GNN_EPOCHS`

## Output Artifacts

After the build and training stages, you should have:

- a persisted LlamaIndex graph under `storage/property_graph`
- base graph artifacts under `artifacts/`
- `artifacts/structural_embeddings.npy`
- `artifacts/gnn_checkpoint.pt`
- retrieval metrics in `evaluation/retrieval_metrics.json`

## Evaluation

`scripts/evaluate.py` uses each dataset question as a query and treats the corresponding source answer chunk as the relevant document. It reports:

- `hit_rate_at_k`
- `mrr_at_k`
- per-question-type metrics

This is enough to compare raw-only, structural-only, and hybrid retrieval by changing `alpha`.

## Hugging Face Space Notes

For free-tier deployment, the best pattern is:

1. preprocess and train once
2. keep artifacts in the repo or a linked dataset repo
3. let the Space only load artifacts and run inference

Do not rebuild the graph or retrain the GNN on every app start.

## Lab Requirement Mapping

- healthcare corpus from Hugging Face: `Tonic/medquad`
- LlamaIndex property graph: `PropertyGraphIndex.from_documents(...)`
- PyTorch Geometric conversion: `lab03/graph_pipeline.py` + `lab03/gnn.py`
- GNN model: residual 2-layer GCN
- two similarity metrics: raw and structural
- weighted linear fusion: `alpha * raw + (1 - alpha) * structural`
- custom `BaseRetriever`: `lab03/retriever.py`
- final generation via allowed GGUF and `llama-cpp`: `lab03/llm.py`
- README as final report: this file

## Source References

- LlamaIndex Property Graph guide: https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/
- LlamaIndex retriever API: https://docs.llamaindex.ai/en/stable/api_reference/retrievers/
- LlamaIndex custom graph retriever example: https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_custom_retriever/
- PyTorch Geometric install docs: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- PyG `GCNConv`: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/en/spaces-overview
- Hugging Face Spaces dependencies: https://huggingface.co/docs/hub/en/spaces-dependencies
- Dataset `Tonic/medquad`: https://huggingface.co/datasets/Tonic/medquad
- Embedding model `sentence-transformers/all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Allowed model `Jackrong/Qwen3.5-4B-Neo-GGUF`: https://huggingface.co/Jackrong/Qwen3.5-4B-Neo-GGUF
