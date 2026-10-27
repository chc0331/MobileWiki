"""Check the synthetic dataset before using it for training or evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from common import DOCUMENTS_PER_EVENT, SPLIT_EVENTS, event_card, read_jsonl, sha256


def validate_dataset(folder: Path, partial: bool = False) -> dict:
    documents = read_jsonl(folder / "documents.jsonl")
    queries = read_jsonl(folder / "queries.jsonl")
    pairs = read_jsonl(folder / "pairs.jsonl")
    triplets = read_jsonl(folder / "triplets.jsonl")
    docs = {item["doc_id"]: item for item in documents}
    q_by_id = {item["query_id"]: item for item in queries}
    if len(docs) != len(documents) or len(q_by_id) != len(queries):
        raise ValueError("Duplicate document or query IDs")
    text_keys = [" ".join(item["text"].split()) for item in documents]
    if len(set(text_keys)) != len(text_keys):
        raise ValueError("Duplicate document text")
    pii = re.compile(r"\b01[016789][- ]?\d{3,4}[- ]?\d{4}\b|\b\d{6}-?[1-4]\d{6}\b|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    checked_texts = text_keys + [item["title"] for item in documents]
    checked_texts += [item["text"] + " " + item["evidence"] for item in queries]
    if any(pii.search(text) for text in checked_texts):
        raise ValueError("Phone number, resident ID, or email-shaped text found")
    if not partial:
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2).fit_transform(text_keys)
        similarity = (vectors @ vectors.T).tocoo()
        for left, right, score in zip(similarity.row, similarity.col, similarity.data):
            if left < right and documents[left]["split"] != documents[right]["split"] and score >= 0.98:
                raise ValueError(
                    f"Near-duplicate across splits: {documents[left]['doc_id']} / {documents[right]['doc_id']} ({score:.3f})"
                )
    family_splits: dict[str, set[str]] = {}
    event_splits: dict[str, set[str]] = {}
    for doc in documents:
        family_splits.setdefault(doc["family_id"], set()).add(doc["split"])
        event_splits.setdefault(doc["event_id"], set()).add(doc["split"])
    if any(len(splits) != 1 for splits in (*family_splits.values(), *event_splits.values())):
        raise ValueError("Family or event crosses splits")
    event_counts = Counter(doc["event_id"] for doc in documents)
    if any(count != DOCUMENTS_PER_EVENT for count in event_counts.values()):
        raise ValueError("An event does not have four documents")
    counts = Counter(item["split"] for item in documents)
    if not partial:
        expected = {split: event_count * DOCUMENTS_PER_EVENT for split, event_count in SPLIT_EVENTS.items()}
        if dict(counts) != expected:
            raise ValueError(f"Unexpected document split counts: {dict(counts)}")
        if Counter(query["split"] for query in queries) != Counter(SPLIT_EVENTS):
            raise ValueError("Unexpected query split counts")
        if Counter(pair["split"] for pair in pairs) != Counter({"dev": 100, "test": 100}):
            raise ValueError("Unexpected pair split counts")
    for query in queries:
        evidence_doc = docs[query["evidence_doc_id"]]
        split, index = query["event_id"].split("-")
        card = event_card(split, int(index))
        if query["evidence"] not in evidence_doc["text"]:
            raise ValueError(f"Missing query evidence: {query['query_id']}")
        if card.fact_value not in query["evidence"] or card.fact_value in query["text"]:
            raise ValueError(f"Ungrounded evidence or answer leakage: {query['query_id']}")
        if len(query["relevant_doc_ids"]) != 1:
            raise ValueError(f"Expected one evidence document: {query['query_id']}")
        if query["relevant_doc_ids"] != [query["evidence_doc_id"]]:
            raise ValueError(f"Relevant document differs from evidence: {query['query_id']}")
        if any(docs[doc_id]["split"] != query["split"] for doc_id in query["relevant_doc_ids"]):
            raise ValueError(f"Query reference crosses splits: {query['query_id']}")
    for pair in pairs:
        left, right = docs[pair["left_id"]], docs[pair["right_id"]]
        if left["split"] != pair["split"] or right["split"] != pair["split"]:
            raise ValueError(f"Pair crosses splits: {pair['pair_id']}")
        if (left["event_id"] == right["event_id"]) != pair["related"]:
            raise ValueError(f"Pair label contradicts event IDs: {pair['pair_id']}")
    for triple in triplets:
        anchor = q_by_id[triple["anchor_id"]] if triple["task"] == "search" else docs[triple["anchor_id"]]
        positive, negative = docs[triple["positive_id"]], docs[triple["negative_id"]]
        if any(item["split"] != "train" for item in (anchor, positive, negative)):
            raise ValueError("Training triplet refers to a non-training split")
        if triple["task"] == "search":
            if negative["doc_id"] in anchor["relevant_doc_ids"]:
                raise ValueError("Search triplet has a false negative")
        elif positive["event_id"] == negative["event_id"]:
            raise ValueError("Clustering triplet has a false negative")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        if sha256(folder / name) != expected_hash:
            raise ValueError(f"Hash mismatch: {name}")
    if sha256(folder / "review.csv") != manifest["review_file_sha256"]:
        raise ValueError("Review sheet hash mismatch")
    return {"documents": dict(counts), "queries": len(queries), "pairs": len(pairs), "triplets": len(triplets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(Path(args.dataset), args.partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
