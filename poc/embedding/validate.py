"""Check the synthetic dataset before using it for training or evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from common import (
    DATE_PATTERN,
    DOCUMENTS_PER_EVENT,
    RELATIVE_DATE_PATTERN,
    RELATIONSHIP_PATTERN,
    SPLIT_EVENTS,
    cards,
    event_card,
    format_event_date,
    normalize_event_dates,
    read_jsonl,
    sha256,
)
from curate import answer_cues


def validate_dataset(folder: Path, partial: bool = False) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") == "context-hardneg-v3":
        from data_v3 import validate_dataset as validate_v3_dataset
        return validate_v3_dataset(folder, partial)
    if manifest.get("dataset_revision") == "shortcut-hardneg-v2-cluster-phase2":
        from phase2 import validate_dataset as validate_phase2_dataset
        return validate_phase2_dataset(folder, partial)
    documents = read_jsonl(folder / "documents.jsonl")
    queries = read_jsonl(folder / "queries.jsonl")
    pairs = read_jsonl(folder / "pairs.jsonl")
    triplets = read_jsonl(folder / "triplets.jsonl")
    docs = {item["doc_id"]: item for item in documents}
    q_by_id = {item["query_id"]: item for item in queries}
    cards_by_event = {card.event_id: card for card in cards()}
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
        if query["answer_doc_index"] != card.answer_doc_index or query["evidence_doc_id"] != f"{card.event_id}-d{card.answer_doc_index}":
            raise ValueError(f"Evidence location differs from event assignment: {query['query_id']}")
        if query["identity_style"] != card.identity_style or query["person"] != card.person:
            raise ValueError(f"Identity metadata differs from event assignment: {query['query_id']}")
        if RELATIVE_DATE_PATTERN.search(query["text"]):
            raise ValueError(f"Relative or approximate date in query: {query['query_id']}")
        if query["date_style"] != card.query_date_style:
            raise ValueError(f"Query date style differs from event assignment: {query['query_id']}")
        query_dates = len(DATE_PATTERN.findall(query["text"]))
        if card.query_date_style == "none":
            if query_dates:
                raise ValueError(f"Unexpected date in query: {query['query_id']}")
        else:
            normalized, count = normalize_event_dates(query["text"], card.date, card.query_date_style)
            if not count or format_event_date(card.date, card.query_date_style) not in normalized:
                raise ValueError(f"Query does not use its assigned absolute date: {query['query_id']}")
        if card.identity_style == "implicit" and (
            card.person in query["text"] or RELATIONSHIP_PATTERN.search(query["text"])
        ):
            raise ValueError(f"Identity shortcut in implicit query: {query['query_id']}")
        if card.identity_style == "named" and card.person not in query["text"]:
            raise ValueError(f"Named query is missing its person: {query['query_id']}")
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
        cues = answer_cues(card.fact_value)
        if any(cue in query["text"] for cue in cues):
            raise ValueError(f"Answer clue in question: {query['query_id']}")
        distractors = [doc for doc in documents if doc["event_id"] == query["event_id"] and doc["doc_id"] != query["evidence_doc_id"]]
        if any(cue in doc["title"] + doc["text"] for doc in distractors for cue in cues):
            raise ValueError(f"Answer clue in distractor: {query['query_id']}")
    for document in documents:
        card = cards_by_event[document["event_id"]]
        doc_index = int(document["doc_id"].rsplit("-d", 1)[1])
        if document["answer_doc_index"] != card.answer_doc_index or document["date_style"] != card.document_date_styles[doc_index]:
            raise ValueError(f"Document metadata differs from event assignment: {document['doc_id']}")
        if card.person in document["title"]:
            raise ValueError(f"Person name in title: {document['doc_id']}")
        if card.identity_style == "implicit" and (
            card.person in document["title"] + document["text"]
            or RELATIONSHIP_PATTERN.search(document["title"] + document["text"])
        ):
            raise ValueError(f"Identity shortcut in implicit event: {document['doc_id']}")
        if RELATIVE_DATE_PATTERN.search(document["title"] + document["text"]):
            raise ValueError(f"Relative or approximate date in document: {document['doc_id']}")
        normalized, count = normalize_event_dates(document["text"], card.date, card.document_date_styles[doc_index])
        if not count or format_event_date(card.date, card.document_date_styles[doc_index]) not in normalized:
            raise ValueError(f"Document does not use its assigned absolute date: {document['doc_id']}")
    for pair in pairs:
        left, right = docs[pair["left_id"]], docs[pair["right_id"]]
        if left["split"] != pair["split"] or right["split"] != pair["split"]:
            raise ValueError(f"Pair crosses splits: {pair['pair_id']}")
        if (left["event_id"] == right["event_id"]) != pair["related"]:
            raise ValueError(f"Pair label contradicts event IDs: {pair['pair_id']}")
    cluster_negative_counts = Counter()
    for triple in triplets:
        anchor = q_by_id[triple["anchor_id"]] if triple["task"] == "search" else docs[triple["anchor_id"]]
        positive, negative = docs[triple["positive_id"]], docs[triple["negative_id"]]
        if any(item["split"] != "train" for item in (anchor, positive, negative)):
            raise ValueError("Training triplet refers to a non-training split")
        if triple["task"] == "search":
            if negative["doc_id"] in anchor["relevant_doc_ids"]:
                raise ValueError("Search triplet has a false negative")
            if positive["doc_id"] != anchor["evidence_doc_id"]:
                raise ValueError("Search triplet positive differs from evidence document")
        elif positive["event_id"] == negative["event_id"]:
            raise ValueError("Clustering triplet has a false negative")
        else:
            kind = triple.get("negative_type")
            cluster_negative_counts[kind] += 1
            if kind == "A":
                if positive["family_id"] != negative["family_id"]:
                    raise ValueError("A negative is not a same-family different event")
            elif kind == "B":
                if positive["category"] != negative["category"] or positive["family_id"] == negative["family_id"]:
                    raise ValueError("B negative is not same-category and different-family")
            elif kind == "C":
                date_close = abs((date.fromisoformat(positive["date"]) - date.fromisoformat(negative["date"])).days) <= 30
                same_fact = positive["fact_value"] == negative["fact_value"]
                if (
                    positive["family_id"] == negative["family_id"]
                    or positive["category"] == negative["category"]
                    or not (date_close or same_fact)
                ):
                    raise ValueError("C negative does not meet its disjoint date/fact criteria")
            elif kind != "partial" or not partial:
                raise ValueError(f"Unknown clustering negative type: {kind}")
    if not partial:
        if Counter(item["task"] for item in triplets) != Counter({"search": 200, "clustering": 800}):
            raise ValueError("Unexpected training triplet counts")
        if cluster_negative_counts != Counter({"A": 400, "B": 240, "C": 160}):
            raise ValueError(f"Unexpected clustering negative composition: {dict(cluster_negative_counts)}")
        for split in SPLIT_EVENTS:
            for category in {card.category for card in cards(split)}:
                members = [card for card in cards(split) if card.category == category]
                for field, values in (
                    ("answer_doc_index", [card.answer_doc_index for card in members]),
                    ("identity_style", [card.identity_style for card in members]),
                ):
                    frequencies = Counter(values)
                    if max(frequencies.values()) - min(frequencies.values()) > 1:
                        raise ValueError(f"Unbalanced {field} assignment: {split}/{category}")
                for doc_index in range(DOCUMENTS_PER_EVENT):
                    frequencies = Counter(card.document_date_styles[doc_index] for card in members)
                    if max(frequencies.values()) - min(frequencies.values()) > 1:
                        raise ValueError(f"Unbalanced document date styles: {split}/{category}/d{doc_index}")
                query_date_counts = Counter(card.query_date_style for card in members)
                no_date_ratio = query_date_counts["none"] / len(members)
                if not 0.4 <= no_date_ratio <= 0.6 or not {"iso", "korean_full"} <= query_date_counts.keys():
                    raise ValueError(f"Unbalanced query date styles: {split}/{category}")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        if sha256(folder / name) != expected_hash:
            raise ValueError(f"Hash mismatch: {name}")
    if sha256(folder / "review.csv") != manifest["review_file_sha256"]:
        raise ValueError("Review sheet hash mismatch")
    return {
        "documents": dict(counts),
        "queries": len(queries),
        "pairs": len(pairs),
        "triplets": len(triplets),
        "clustering_negative_types": dict(cluster_negative_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(Path(args.dataset), args.partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
