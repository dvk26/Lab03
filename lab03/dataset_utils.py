from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from datasets import load_dataset
from llama_index.core import Document

from lab03.config import BuildConfig

_INSTRUCTION_RE = re.compile(
    r"###\s*Instruction:\s*(?P<question>.*?)\s*###\s*Response:\s*(?P<answer>.*)",
    re.DOTALL | re.IGNORECASE,
)

_QUESTION_TYPE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("symptoms", ("symptom", "signs")),
    ("treatment", ("treatments", "treatment", "therapy", "therapies", "medicines")),
    ("prevention", ("prevent", "prevention", "avoid")),
    ("risk", ("risk", "who gets", "who is more likely")),
    ("causes", ("cause", "causes", "caused by")),
    ("diagnosis", ("diagnos", "test", "screen", "detect")),
    ("complications", ("complication", "problems can", "related problems")),
    ("prognosis", ("outlook", "prognosis", "life expectancy")),
    ("transmission", ("spread", "transmit", "catch")),
    ("overview", ("what is", "what are", "define", "overview")),
]

_QUESTION_PREFIXES = [
    "what are the symptoms of",
    "what are the signs and symptoms of",
    "what are the treatments for",
    "what is the treatment for",
    "how to prevent",
    "how can i prevent",
    "who is at risk for",
    "who gets",
    "what causes",
    "what is the cause of",
    "how to diagnose",
    "how is",
    "what tests diagnose",
    "what complications can",
    "how does",
    "what is",
    "what are",
]


@dataclass(slots=True)
class MedicalRecord:
    doc_id: str
    question: str
    answer: str
    question_type: str
    condition: str
    source_text: str
    source_dataset: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_instruction_response(text: str) -> tuple[str, str]:
    match = _INSTRUCTION_RE.search(text)
    if not match:
        stripped = " ".join(text.split())
        return stripped, stripped
    question = " ".join(match.group("question").split())
    answer = " ".join(match.group("answer").split())
    return question, answer


def infer_question_type(question: str) -> str:
    question_lower = question.lower()
    for label, keywords in _QUESTION_TYPE_PATTERNS:
        if any(keyword in question_lower for keyword in keywords):
            return label
    return "overview"


def normalize_condition(question: str, question_type: str) -> str:
    cleaned = question.lower().strip().rstrip(" ?.")
    for prefix in _QUESTION_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    cleaned = cleaned.replace(" ?", "").strip(" -:")
    if not cleaned:
        cleaned = question.lower().strip().rstrip(" ?.")
    cleaned = re.sub(r"\s+", " ", cleaned)
    condition = cleaned.title()
    if question_type == "overview" and condition.lower().startswith("the "):
        condition = condition[4:]
    return condition or "Unknown Condition"


def build_source_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def _stable_doc_id(dataset_id: str, question: str, answer: str) -> str:
    digest = hashlib.sha1(f"{dataset_id}::{question}::{answer}".encode("utf-8")).hexdigest()
    return digest[:16]


def _iter_raw_records(dataset_id: str, split: str) -> Iterable[dict]:
    dataset = load_dataset(dataset_id, split=split)
    for row in dataset:
        yield row


def load_medical_records(config: BuildConfig) -> list[MedicalRecord]:
    parsed: list[MedicalRecord] = []
    for row in _iter_raw_records(config.dataset_id, config.dataset_split):
        raw_text = row.get("text", "")
        question, answer = parse_instruction_response(raw_text)
        if not question or not answer:
            continue
        question_type = infer_question_type(question)
        condition = normalize_condition(question, question_type)
        doc_id = _stable_doc_id(config.dataset_id, question, answer)
        parsed.append(
            MedicalRecord(
                doc_id=doc_id,
                question=question,
                answer=answer,
                question_type=question_type,
                condition=condition,
                source_text=build_source_text(question, answer),
                source_dataset=config.dataset_id,
            )
        )
    return stratified_sample(parsed, config.max_records, config.sample_seed)


def stratified_sample(
    records: Sequence[MedicalRecord], max_records: int, sample_seed: int
) -> list[MedicalRecord]:
    if max_records <= 0 or len(records) <= max_records:
        return list(records)

    rng = random.Random(sample_seed)
    by_type: dict[str, list[MedicalRecord]] = {}
    for record in records:
        by_type.setdefault(record.question_type, []).append(record)

    for items in by_type.values():
        rng.shuffle(items)

    ordered_types = sorted(by_type)
    sampled: list[MedicalRecord] = []
    while len(sampled) < max_records and ordered_types:
        remaining_types: list[str] = []
        for question_type in ordered_types:
            bucket = by_type[question_type]
            if bucket and len(sampled) < max_records:
                sampled.append(bucket.pop())
            if bucket:
                remaining_types.append(question_type)
        ordered_types = remaining_types

    return sampled


def records_to_documents(records: Sequence[MedicalRecord]) -> list[Document]:
    documents: list[Document] = []
    for record in records:
        documents.append(
            Document(
                text=record.answer,
                doc_id=record.doc_id,
                metadata={
                    "doc_id": record.doc_id,
                    "question": record.question,
                    "question_type": record.question_type,
                    "condition": record.condition,
                    "source_text": record.source_text,
                    "source_dataset": record.source_dataset,
                },
            )
        )
    return documents

