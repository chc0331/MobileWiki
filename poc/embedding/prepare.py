"""Compile teacher output into document, query, triplet, and review files."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import (
    DOCUMENTS_PER_EVENT,
    PARAPHRASE_QUERIES,
    SPLIT_EVENTS,
    cards,
    read_jsonl,
    sha256,
    write_jsonl,
)


def compile_rows(raw: list[dict], allow_partial: bool = False) -> dict[str, list[dict]]:
    by_event: dict[str, dict] = {}
    for row in raw:
        event_id = row["event_id"]
        if event_id in by_event:
            raise ValueError(f"Duplicate event: {event_id}")
        by_event[event_id] = row
    expected = {card.event_id for card in cards()}
    if not allow_partial and set(by_event) != expected:
        missing = expected - set(by_event)
        extra = set(by_event) - expected
        raise ValueError(f"Event set incomplete: {len(missing)} missing, {len(extra)} extra")
    if set(by_event) - expected:
        raise ValueError("Unknown event IDs in teacher output")
    snapshots = {(row["teacher_id"], row["teacher_revision"], row["prompt_version"]) for row in raw}
    if len(snapshots) != 1:
        raise ValueError("A dataset must use one teacher snapshot and prompt version")

    documents: list[dict] = []
    queries: list[dict] = []
    event_docs: dict[str, list[dict]] = {}
    by_category: dict[tuple[str, str], list[str]] = defaultdict(list)
    for card in cards():
        row = by_event.get(card.event_id)
        if row is None:
            continue
        if any(row[key] != value for key, value in card.as_dict().items()):
            raise ValueError(f"Card mismatch: {card.event_id}")
        if len(row["documents"]) != DOCUMENTS_PER_EVENT:
            raise ValueError(f"Wrong document count: {card.event_id}")
        event_docs[card.event_id] = []
        by_category[(card.split, card.category)].append(card.event_id)
        for index, item in enumerate(row["documents"]):
            title = item["title"].strip()
            if card.person not in title:
                title = f"{card.person} · {title}"
            document = {
                "doc_id": f"{card.event_id}-d{index}",
                "event_id": card.event_id,
                "family_id": card.family_id,
                "split": card.split,
                "category": card.category,
                "title": title,
                "text": item["text"].strip(),
            }
            documents.append(document)
            event_docs[card.event_id].append(document)
        use_paraphrase = int(card.event_id.split("-")[-1]) % 2 == 0
        question = PARAPHRASE_QUERIES[card.category] if use_paraphrase else row["query"].strip()
        query_source = "paraphrase_template" if use_paraphrase else "teacher"
        if not use_paraphrase and (ord(card.fact_label[-1]) - 0xAC00) % 28 == 0:
            corrected = question.replace(f"{card.fact_label}은", f"{card.fact_label}는")
            if corrected != question:
                question = corrected
                query_source = "teacher_grammar_fixed"
        context = [value for value in (card.person, card.date, card.category) if value not in question]
        if context:
            question = f"{' '.join(context)} 기록에서 {question}"
        relevant = [
            document["doc_id"]
            for document in event_docs[card.event_id]
            if card.fact_value in document["text"]
        ]
        if event_docs[card.event_id][0]["doc_id"] not in relevant:
            raise ValueError(f"Search evidence missing: {card.event_id}")
        if len(relevant) != 1:
            raise ValueError(f"Answer repeated outside evidence document: {card.event_id}")
        queries.append(
            {
                "query_id": f"{card.event_id}-q0",
                "event_id": card.event_id,
                "split": card.split,
                "text": question,
                "query_source": query_source,
                "relevant_doc_ids": relevant,
                "evidence_doc_id": event_docs[card.event_id][0]["doc_id"],
                "evidence": row["evidence"],
                "reviewed": False,
            }
        )

    pairs: list[dict] = []
    triplets: list[dict] = []
    for split in SPLIT_EVENTS:
        split_events = [card.event_id for card in cards(split) if card.event_id in event_docs]
        for event_id in split_events:
            row = by_event[event_id]
            peers = by_category[(split, row["category"])]
            if len(peers) < 2:
                continue
            negative_event = peers[(peers.index(event_id) + 1) % len(peers)]
            if split == "train":
                triplets.append(
                    {
                        "task": "search",
                        "anchor_id": f"{event_id}-q0",
                        "positive_id": f"{event_id}-d0",
                        "negative_id": f"{event_id}-d1",
                    }
                )
                for index in range(DOCUMENTS_PER_EVENT):
                    triplets.append(
                        {
                            "task": "clustering",
                            "anchor_id": f"{event_id}-d{index}",
                            "positive_id": f"{event_id}-d{(index + 1) % DOCUMENTS_PER_EVENT}",
                            "negative_id": f"{negative_event}-d{index}",
                        }
                    )
        if split in ("dev", "test") and len(split_events) >= 50:
            for event_id in split_events[:50]:
                pairs.append(
                    {
                        "pair_id": f"{event_id}-related",
                        "split": split,
                        "left_id": f"{event_id}-d0",
                        "right_id": f"{event_id}-d1",
                        "related": True,
                        "reviewed": False,
                    }
                )
                category = by_event[event_id]["category"]
                peers = by_category[(split, category)]
                other = peers[(peers.index(event_id) + 1) % len(peers)]
                pairs.append(
                    {
                        "pair_id": f"{event_id}-unrelated",
                        "split": split,
                        "left_id": f"{event_id}-d0",
                        "right_id": f"{other}-d0",
                        "related": False,
                        "reviewed": False,
                    }
                )
    return {"documents": documents, "queries": queries, "pairs": pairs, "triplets": triplets}


def write_review(path: Path, rows: dict[str, list[dict]]) -> None:
    docs = {row["doc_id"]: row for row in rows["documents"]}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["kind", "item_id", "question_or_left", "evidence_or_right", "gold", "accepted", "comment"],
        )
        writer.writeheader()
        test_queries = [query for query in rows["queries"] if query["split"] == "test"]
        for query in test_queries[:140]:
            event_documents = [doc for doc in rows["documents"] if doc["event_id"] == query["event_id"]]
            writer.writerow(
                {
                    "kind": "query",
                    "item_id": query["query_id"],
                    "question_or_left": query["text"],
                    "evidence_or_right": "\n".join(f"{doc['doc_id']}: {doc['text']}" for doc in event_documents),
                    "gold": query["evidence"],
                }
            )
        for pair in rows["pairs"]:
            if pair["split"] != "test":
                continue
            writer.writerow(
                {
                    "kind": "pair",
                    "item_id": pair["pair_id"],
                    "question_or_left": docs[pair["left_id"]]["text"],
                    "evidence_or_right": docs[pair["right_id"]]["text"],
                    "gold": str(pair["related"]),
                }
            )


def prepare(args: argparse.Namespace) -> None:
    raw_path, output = Path(args.input), Path(args.output)
    raw = read_jsonl(raw_path)
    rows = compile_rows(raw, allow_partial=args.allow_partial)
    output.mkdir(parents=True, exist_ok=True)
    for name, items in rows.items():
        write_jsonl(output / f"{name}.jsonl", items)
    write_review(output / "review.csv", rows)
    counts = {
        name: dict(Counter(row["split"] for row in items if "split" in row))
        for name, items in rows.items() if name != "triplets"
    }
    counts["triplets"] = len(rows["triplets"])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_pending_human_review",
        "teacher_id": raw[0]["teacher_id"],
        "teacher_revision": raw[0]["teacher_revision"],
        "teacher_runtime": subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
        "prompt_version": raw[0]["prompt_version"],
        "split_policy": "families, events, and source cards stay in one split",
        "counts": counts,
        "repair_counts": dict(Counter(repair for row in raw for repair in row.get("repairs", []))),
        "files": {path.name: sha256(path) for path in output.glob("*.jsonl")},
        "review_file_sha256": sha256(output / "review.csv"),
        "raw_sha256": sha256(raw_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/raw_events.jsonl")
    parser.add_argument("--output", default="artifacts/dataset")
    parser.add_argument("--allow-partial", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
