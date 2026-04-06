from __future__ import annotations

import re
from typing import Iterable

try:
    from llama_index.core.graph_stores.types import (
        EntityNode,
        KG_NODES_KEY,
        KG_RELATIONS_KEY,
        Relation,
    )
    from llama_index.core.schema import BaseNode, TransformComponent
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "LlamaIndex graph types are required. Install dependencies from requirements.txt."
    ) from exc


def relation_label_for(question_type: str) -> str:
    return {
        "symptoms": "HAS_SYMPTOM_INFO",
        "treatment": "HAS_TREATMENT_INFO",
        "prevention": "HAS_PREVENTION_INFO",
        "risk": "HAS_RISK_INFO",
        "causes": "HAS_CAUSE_INFO",
        "diagnosis": "HAS_DIAGNOSIS_INFO",
        "complications": "HAS_COMPLICATION_INFO",
        "prognosis": "HAS_PROGNOSIS_INFO",
        "transmission": "HAS_TRANSMISSION_INFO",
        "overview": "HAS_OVERVIEW_INFO",
    }.get(question_type, "HAS_OVERVIEW_INFO")


def split_sentences(text: str, max_sentences: int, min_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    clean = [sentence.strip() for sentence in sentences if len(sentence.strip()) >= min_chars]
    return clean[:max_sentences]


class MedicalGraphExtractor(TransformComponent):
    """Deterministic extractor for free-tier reproducibility."""

    max_sentences: int = 5
    min_sentence_chars: int = 35

    def __init__(self, max_sentences: int = 5, min_sentence_chars: int = 35) -> None:
        self.max_sentences = max_sentences
        self.min_sentence_chars = min_sentence_chars

    def __call__(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        for llama_node in llama_nodes:
            metadata = dict(llama_node.metadata)
            doc_id = str(metadata.get("doc_id", llama_node.node_id))
            condition = str(metadata.get("condition", "Unknown Condition"))
            question_type = str(metadata.get("question_type", "overview"))
            relation_label = relation_label_for(question_type)

            existing_nodes = list(metadata.pop(KG_NODES_KEY, []))
            existing_relations = list(metadata.pop(KG_RELATIONS_KEY, []))

            condition_node = EntityNode(
                name=condition,
                label="CONDITION",
                properties={"doc_id": doc_id},
            )
            aspect_name = question_type.upper()
            aspect_node = EntityNode(
                name=aspect_name,
                label="ASPECT",
                properties={"question_type": question_type},
            )

            existing_nodes.extend([condition_node, aspect_node])
            existing_relations.append(
                Relation(
                    label="HAS_ASPECT",
                    source_id=condition_node.id,
                    target_id=aspect_node.id,
                    properties={"doc_id": doc_id},
                )
            )

            for idx, sentence in enumerate(
                split_sentences(llama_node.get_content(), self.max_sentences, self.min_sentence_chars)
            ):
                statement_node = EntityNode(
                    name=f"{doc_id}::statement::{idx}",
                    label="STATEMENT",
                    properties={
                        "doc_id": doc_id,
                        "question_type": question_type,
                        "text": sentence,
                    },
                )
                existing_nodes.append(statement_node)
                existing_relations.append(
                    Relation(
                        label=relation_label,
                        source_id=condition_node.id,
                        target_id=statement_node.id,
                        properties={"doc_id": doc_id},
                    )
                )
                existing_relations.append(
                    Relation(
                        label="IN_SECTION",
                        source_id=aspect_node.id,
                        target_id=statement_node.id,
                        properties={"doc_id": doc_id},
                    )
                )

            llama_node.metadata = metadata
            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes
