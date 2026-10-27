"""Measure retrieval, document grouping, and extractive group labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (
    STUDENT_MODEL,
    clustering_text,
    matching_doc_text,
    read_jsonl,
    search_query_text,
    write_jsonl,
)

THRESHOLDS = tuple(round(value * 0.05, 2) for value in range(1, 17))
BORING_LABELS = {
    "메모", "기록", "확인", "예정", "일정", "사항", "관련", "내용", "이름", "날짜", "시간",
    "필요", "모두", "사실", "경우", "정보", "완료", "추가", "정리", "가족", "물건", "동료",
    "구성원", "연락", "날씨", "준비",
}


def pair_f1(pairs: list[dict], groups: dict[str, int]) -> dict:
    tp = fp = fn = tn = 0
    for pair in pairs:
        prediction = groups[pair["left_id"]] == groups[pair["right_id"]]
        if pair["related"] and prediction:
            tp += 1
        elif pair["related"]:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def groups_for(documents: list[dict], vectors: np.ndarray, threshold: float) -> dict[str, int]:
    if len(documents) < 2:
        return {doc["doc_id"]: index for index, doc in enumerate(documents)}
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="average",
        metric="cosine",
    ).fit_predict(vectors)
    return {document["doc_id"]: int(label) for document, label in zip(documents, labels)}


def chunk_document(document: dict, tokenizer, max_tokens: int = 420, overlap: int = 64) -> list[dict]:
    tokens = tokenizer.encode(document["text"], add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [document]
    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        piece = tokenizer.decode(tokens[start : start + max_tokens], skip_special_tokens=True)
        chunks.append({**document, "text": piece})
        if start + max_tokens >= len(tokens):
            break
    return chunks


def encode_documents(model, documents: list[dict], task: str, batch_size: int) -> tuple[np.ndarray, list[str]]:
    chunks = [chunk for document in documents for chunk in chunk_document(document, model.tokenizer)]
    texts = [matching_doc_text(chunk) if task == "search" else clustering_text(chunk) for chunk in chunks]
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32), [chunk["doc_id"] for chunk in chunks]


def document_vectors(chunk_vectors: np.ndarray, chunk_ids: list[str], documents: list[dict]) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for vector, doc_id in zip(chunk_vectors, chunk_ids):
        grouped[doc_id].append(vector)
    vectors = np.stack([np.mean(grouped[doc["doc_id"]], axis=0) for doc in documents])
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def retrieval_score(queries: list[dict], ranked_ids: list[list[str]]) -> dict:
    ranks = []
    for query, ids in zip(queries, ranked_ids):
        relevant = set(query["relevant_doc_ids"])
        rank = next((index for index, doc_id in enumerate(ids, start=1) if doc_id in relevant), None)
        ranks.append(rank)
    hits_at_1 = sum(rank == 1 for rank in ranks)
    hits_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    total = len(ranks)
    return {
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "total": total,
        "recall_at_1": hits_at_1 / total if total else 0.0,
        "recall_at_5": hits_at_5 / total if total else 0.0,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total if total else 0.0,
    }


def dense_search(model, documents: list[dict], queries: list[dict], batch_size: int) -> dict:
    vectors, doc_ids = encode_documents(model, documents, "search", batch_size)
    query_vectors = np.asarray(
        model.encode(
            [search_query_text(query["text"]) for query in queries],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    ranked = []
    for query_vector in query_vectors:
        scores: dict[str, float] = {}
        for doc_id, score in zip(doc_ids, vectors @ query_vector):
            scores[doc_id] = max(float(score), scores.get(doc_id, float("-inf")))
        ranked.append(sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id)))
    return retrieval_score(queries, ranked)


def lexical_search(documents: list[dict], queries: list[dict]) -> dict:
    if not queries:
        return retrieval_score([], [])
    texts = [f"{item['title']} {item['text']}" for item in documents]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform([item["text"] for item in queries])
    ranked = []
    for query_vector in query_matrix:
        scores = (matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores)
        ranked.append([documents[index]["doc_id"] for index in order])
    return retrieval_score(queries, ranked)


def extract_labels(documents: list[dict], groups: dict[str, int]) -> list[dict]:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    all_tokens: dict[str, set[str]] = {}
    doc_frequency: Counter[str] = Counter()
    for document in documents:
        person, separator, title = document["title"].partition(" · ")
        content = (title if separator else document["title"]) + " " + document["text"]
        if separator:
            content = content.replace(person, "").replace(person[1:], "")
        tokens = {
            token.form
            for token in kiwi.tokenize(content)
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2 and token.form not in BORING_LABELS
        }
        all_tokens[document["doc_id"]] = tokens
        doc_frequency.update(tokens)
    members: dict[int, list[dict]] = defaultdict(list)
    for document in documents:
        members[groups[document["doc_id"]]].append(document)
    output = []
    for group_id, grouped_docs in sorted(members.items()):
        counts = Counter(token for doc in grouped_docs for token in all_tokens[doc["doc_id"]])
        minimum = max(1, math.ceil(len(grouped_docs) / 2))
        candidates = [token for token, count in counts.items() if count >= minimum]
        candidates.sort(
            key=lambda token: (
                -counts[token],
                -math.log((len(documents) + 1) / (doc_frequency[token] + 1)),
                token,
            )
        )
        label = candidates[0] if candidates else "이름 미정"
        sources = [doc["doc_id"] for doc in grouped_docs if label in doc["title"] + doc["text"]]
        output.append({"group_id": group_id, "label": label, "doc_ids": [doc["doc_id"] for doc in grouped_docs], "source_doc_ids": sources})
    return output


def write_tag_review(path: Path, labels: list[dict], documents: list[dict]) -> None:
    docs = {document["doc_id"]: document for document in documents}
    ordered = sorted(labels, key=lambda item: (-len(item["doc_ids"]), item["group_id"]))
    count = min(30, len(ordered))
    selected = [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)] if count > 1 else ordered
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_id", "tag", "document_count", "source_doc_ids", "example_documents", "accepted", "comment"])
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "group_id": item["group_id"],
                "tag": item["label"],
                "document_count": len(item["doc_ids"]),
                "source_doc_ids": " ".join(item["source_doc_ids"]),
                "example_documents": "\n".join(f"{doc_id}: {docs[doc_id]['text']}" for doc_id in item["doc_ids"][:4]),
            })


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    dataset = Path(args.dataset)
    documents = read_jsonl(dataset / "documents.jsonl")
    queries = read_jsonl(dataset / "queries.jsonl")
    pairs = read_jsonl(dataset / "pairs.jsonl")
    if args.adapter:
        adapter_path = Path(args.adapter)
        training_path = adapter_path / "training.json"
        if not training_path.exists():
            training_path = adapter_path.parent / "training.json"
        if not training_path.exists():
            raise FileNotFoundError(f"Missing training provenance: {training_path}")
        base_revision = json.loads(training_path.read_text(encoding="utf-8"))["base_revision"]
        model = SentenceTransformer(
            STUDENT_MODEL,
            revision=base_revision,
            device="mps",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32},
            default_prompt_name=None,
        )
        model.load_adapter(args.adapter)
        model_id = str(args.adapter)
    else:
        base_revision = args.base_revision or HfApi().model_info(STUDENT_MODEL).sha
        model = SentenceTransformer(
            STUDENT_MODEL,
            revision=base_revision,
            device="mps",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32},
            default_prompt_name=None,
        )
        model_id = STUDENT_MODEL
    model.max_seq_length = 512
    dev_docs = [doc for doc in documents if doc["split"] == "dev"]
    test_docs = [doc for doc in documents if doc["split"] == "test"]
    dev_pairs = [pair for pair in pairs if pair["split"] == "dev"]
    test_pairs = [pair for pair in pairs if pair["split"] == "test"]
    test_queries = [query for query in queries if query["split"] == "test"][:140]
    dev_chunks, dev_chunk_ids = encode_documents(model, dev_docs, "clustering", args.batch_size)
    dev_vectors = document_vectors(dev_chunks, dev_chunk_ids, dev_docs)
    tuning = [
        {"threshold": threshold, **pair_f1(dev_pairs, groups_for(dev_docs, dev_vectors, threshold))}
        for threshold in THRESHOLDS
    ]
    selected = max(tuning, key=lambda row: (row["f1"], -row["threshold"]))
    test_chunks, test_chunk_ids = encode_documents(model, test_docs, "clustering", args.batch_size)
    test_vectors = document_vectors(test_chunks, test_chunk_ids, test_docs)
    groups = groups_for(test_docs, test_vectors, selected["threshold"])
    labels = extract_labels(test_docs, groups)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output.with_suffix(".labels.jsonl"), labels)
    write_tag_review(output.with_suffix(".tag_review.csv"), labels, test_docs)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": manifest["status"],
        "model": model_id,
        "base_revision": base_revision,
        "teacher_id": manifest["teacher_id"],
        "teacher_revision": manifest["teacher_revision"],
        "prompt_version": manifest["prompt_version"],
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "scikit_learn": version("scikit-learn"),
        },
        "dataset_files": manifest["files"],
        "dev_selected_threshold": selected["threshold"],
        "dev_pair_f1": selected["f1"],
        "test_pair": pair_f1(test_pairs, groups),
        "test_dense_search": dense_search(model, test_docs, test_queries, args.batch_size),
        "test_lexical_search": lexical_search(test_docs, test_queries),
        "test_groups": len(labels),
        "test_groups_unnamed": sum(label["label"] == "이름 미정" for label in labels),
        "tag_review_sample_count": min(30, len(labels)),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--adapter")
    parser.add_argument("--base-revision")
    parser.add_argument("--output", default="artifacts/evaluation/base.json")
    parser.add_argument("--batch-size", type=int, default=2)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
