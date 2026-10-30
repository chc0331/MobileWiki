"""Compile teacher output into document, query, triplet, and review files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from common import (
    DOCUMENTS_PER_EVENT,
    RELATIONSHIP_PATTERN,
    SEED,
    SPLIT_EVENTS,
    cards,
    read_jsonl,
    sha256,
    write_jsonl,
)


def _stable_order(value: str, seed: str) -> bytes:
    return hashlib.sha256(f"{SEED}|{seed}|{value}".encode("utf-8")).digest()


def _negative_pools(card, train_cards: list) -> dict[str, list]:
    pools = {"A": [], "B": [], "C": []}
    for candidate in train_cards:
        if candidate.event_id == card.event_id:
            continue
        if candidate.family_id == card.family_id:
            pools["A"].append(candidate)
        elif candidate.category == card.category:
            pools["B"].append(candidate)
        elif (
            candidate.fact_value == card.fact_value
            or abs((date.fromisoformat(candidate.date) - date.fromisoformat(card.date)).days) <= 30
        ):
            pools["C"].append(candidate)
    return pools


def _cluster_type_schedule(train_cards: list) -> dict[str, list[str]]:
    if len(train_cards) != SPLIT_EVENTS["train"]:
        return {card.event_id: ["partial"] * DOCUMENTS_PER_EVENT for card in train_cards}
    double_b_events: set[str] = set()
    categories: dict[str, list] = defaultdict(list)
    for card in train_cards:
        categories[card.category].append(card)
    for category, members in sorted(categories.items()):
        ordered = sorted(members, key=lambda card: _stable_order(card.event_id, f"double-b-{category}"))
        double_b_events.update(card.event_id for card in ordered[:4])

    output = {}
    for card in train_cards:
        types = ["A", "A", "B", "B"] if card.event_id in double_b_events else ["A", "A", "B", "C"]
        seed = int.from_bytes(_stable_order(card.event_id, "cluster-type-schedule")[:8], "big")
        random.Random(seed).shuffle(types)
        output[card.event_id] = types
    return output


def _pick_negative(pool: list, event_id: str, anchor_index: int, kind: str, used: set[str]):
    ordered = sorted(pool, key=lambda card: _stable_order(card.event_id, f"{event_id}-{anchor_index}-{kind}"))
    candidate = next((card for card in ordered if card.event_id not in used), ordered[0] if ordered else None)
    if candidate is not None:
        used.add(candidate.event_id)
    return candidate


def compile_rows(raw: list[dict], allow_partial: bool = False) -> dict[str, list[dict]]:
    if raw and raw[0].get("card_revision") == "context-hardneg-v3":
        from data_v3 import compile_rows as compile_v3_rows
        return compile_v3_rows(raw, allow_partial)
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
    snapshots = {
        (row["teacher_id"], row["teacher_revision"], row["prompt_version"], row.get("curation_version"))
        for row in raw
    }
    if len(snapshots) != 1:
        raise ValueError("A dataset must use one teacher snapshot, prompt version, and curation version")

    documents: list[dict] = []
    queries: list[dict] = []
    event_docs: dict[str, list[dict]] = {}
    by_category: dict[tuple[str, str], list[str]] = defaultdict(list)
    card_by_event = {card.event_id: card for card in cards()}
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
            text = item["text"].strip()
            if card.person in title:
                raise ValueError(f"Person name in title: {card.event_id}-d{index}")
            if card.identity_style == "implicit" and (
                card.person in title + text or RELATIONSHIP_PATTERN.search(title + text)
            ):
                raise ValueError(f"Identity shortcut in implicit event: {card.event_id}-d{index}")
            document = {
                "doc_id": f"{card.event_id}-d{index}",
                "event_id": card.event_id,
                "family_id": card.family_id,
                "split": card.split,
                "category": card.category,
                "date": card.date,
                "fact_value": card.fact_value,
                "person": card.person,
                "identity_style": card.identity_style,
                "answer_doc_index": card.answer_doc_index,
                "date_style": card.document_date_styles[index],
                "title": title,
                "text": text,
            }
            documents.append(document)
            event_docs[card.event_id].append(document)
        question = row["query"].strip()
        query_source = "curated_paraphrase" if "query: replaced answer clue with paraphrase" in row.get("curation_repairs", []) else "teacher"
        if card.identity_style == "implicit" and (
            card.person in question or RELATIONSHIP_PATTERN.search(question)
        ):
            raise ValueError(f"Identity shortcut in implicit query: {card.event_id}")
        if card.identity_style == "named" and card.person not in question:
            raise ValueError(f"Missing person name in named query: {card.event_id}")
        relevant = [document["doc_id"] for document in event_docs[card.event_id] if card.fact_value in document["text"]]
        evidence_doc_id = f"{card.event_id}-d{card.answer_doc_index}"
        if evidence_doc_id not in relevant:
            raise ValueError(f"Search evidence missing from configured document: {card.event_id}")
        if relevant != [evidence_doc_id]:
            raise ValueError(f"Answer repeated outside evidence document: {card.event_id}")
        queries.append(
            {
                "query_id": f"{card.event_id}-q0",
                "event_id": card.event_id,
                "split": card.split,
                "person": card.person,
                "identity_style": card.identity_style,
                "answer_doc_index": card.answer_doc_index,
                "date_style": card.query_date_style,
                "text": question,
                "query_source": query_source,
                "relevant_doc_ids": relevant,
                "evidence_doc_id": evidence_doc_id,
                "evidence": row["evidence"],
                "reviewed": False,
            }
        )

    pairs: list[dict] = []
    triplets: list[dict] = []
    train_cards = [card_by_event[event_id] for event_id in event_docs if card_by_event[event_id].split == "train"]
    train_type_schedule = _cluster_type_schedule(train_cards)
    negative_usage: dict[tuple[str, str], set[str]] = defaultdict(set)
    pools_by_event = {card.event_id: _negative_pools(card, train_cards) for card in train_cards}
    for split in SPLIT_EVENTS:
        split_events = [card.event_id for card in cards(split) if card.event_id in event_docs]
        for event_id in split_events:
            row = by_event[event_id]
            peers = by_category[(split, row["category"])]
            if len(peers) < 2:
                continue
            if split == "train":
                card = card_by_event[event_id]
                fact_tokens = [token for token in row["fact_value"].split() if len(token) >= 2]
                search_negative_index = min(
                    (index for index in range(DOCUMENTS_PER_EVENT) if index != card.answer_doc_index),
                    key=lambda index: (
                        sum(token in event_docs[event_id][index]["text"] for token in fact_tokens),
                        index,
                    ),
                )
                triplets.append(
                    {
                        "task": "search",
                        "anchor_id": f"{event_id}-q0",
                        "positive_id": f"{event_id}-d{card.answer_doc_index}",
                        "negative_id": f"{event_id}-d{search_negative_index}",
                    }
                )
                for index, negative_type in enumerate(train_type_schedule[event_id]):
                    pool = pools_by_event[event_id].get(negative_type, [])
                    negative = _pick_negative(
                        pool,
                        event_id,
                        index,
                        negative_type,
                        negative_usage[(event_id, negative_type)],
                    )
                    if negative is None and allow_partial:
                        fallback = [candidate for candidate in train_cards if candidate.event_id != event_id]
                        negative = _pick_negative(
                            fallback,
                            event_id,
                            index,
                            "partial",
                            negative_usage[(event_id, "partial")],
                        )
                        negative_type = "partial"
                    if negative is None:
                        raise ValueError(f"No {negative_type} negative candidate for {event_id}")
                    triplets.append(
                        {
                            "task": "clustering",
                            "anchor_id": f"{event_id}-d{index}",
                            "positive_id": f"{event_id}-d{(index + 1) % DOCUMENTS_PER_EVENT}",
                            "negative_id": f"{negative.event_id}-d{index}",
                            "negative_type": negative_type,
                        }
                    )
        if split in ("dev", "test") and len(split_events) >= 50:
            for event_id in split_events[:50]:
                slot = int.from_bytes(_stable_order(event_id, f"pair-slot-{split}")[:4], "big") % DOCUMENTS_PER_EVENT
                pairs.append(
                    {
                        "pair_id": f"{event_id}-related",
                        "split": split,
                        "left_id": f"{event_id}-d{slot}",
                        "right_id": f"{event_id}-d{(slot + 1) % DOCUMENTS_PER_EVENT}",
                        "related": True,
                        "reviewed": False,
                    }
                )
                category = by_event[event_id]["category"]
                same_topic_peers = by_category[(split, category)]
                other = same_topic_peers[(same_topic_peers.index(event_id) + 1) % len(same_topic_peers)]
                pairs.append(
                    {
                        "pair_id": f"{event_id}-unrelated",
                        "split": split,
                        "left_id": f"{event_id}-d{slot}",
                        "right_id": f"{other}-d{slot}",
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
    counts["triplets_by_task"] = dict(Counter(item["task"] for item in rows["triplets"]))
    counts["clustering_negative_types"] = dict(
        Counter(item.get("negative_type", "unknown") for item in rows["triplets"] if item["task"] == "clustering")
    )
    curation_path = raw_path.parent / "curation.json"
    curation = None
    if curation_path.exists():
        candidate = json.loads(curation_path.read_text(encoding="utf-8"))
        if candidate["curated_sha256"] == sha256(raw_path):
            curation = candidate
        elif not args.allow_partial:
            raise ValueError("Curation manifest differs from prepared source")
    if not args.allow_partial and curation is None:
        raise ValueError("Full dataset requires a matching curation manifest")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_pending_human_review",
        "teacher_id": raw[0]["teacher_id"],
        "teacher_revision": raw[0]["teacher_revision"],
        "teacher_runtime": subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
        "prompt_version": raw[0]["prompt_version"],
        "code_files": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("common.py", "generate.py", "repair_duplicates.py", "curate.py", "prepare.py", "validate.py", "train.py", "evaluate.py", *( ("data_v3.py",) if raw[0].get("card_revision") == "context-hardneg-v3" else () ))
        },
        "dataset_revision": raw[0].get("card_revision", "shortcut-hardneg-v2-balanced"),
        "data_seed": SEED,
        "split_policy": "families, events, and source cards stay in one split",
        "answer_position_policy": "balanced by split and category across d0-d3",
        "identity_style_policy": "balanced named/implicit by split and category; no person-name title prefix",
        "date_style_policy": "ISO, exact Korean full date, or exact Korean month-day; relative and approximate dates are excluded",
        "search_negative_policy": "one within-event evidence negative and one same-family other-event negative per query" if raw[0].get("card_revision") == "context-hardneg-v3" else "same-event non-answer document with least fact-token overlap",
        "clustering_negative_policy": "A 50% same-family other-event; B 30% same-category other-family different-fact, same-date prioritized; C 20% same-fact other-family" if raw[0].get("card_revision") == "context-hardneg-v3" else "A 50% same-family other-event; B 30% same-category other-family; C 20% other-family other-category with shared fact or within 30 days",
        "counts": counts,
        "repair_counts": dict(Counter(repair for row in raw for repair in row.get("repairs", []))),
        "curation_repair_counts": dict(Counter(repair for row in raw for repair in row.get("curation_repairs", []))),
        "files": {path.name: sha256(path) for path in output.glob("*.jsonl")},
        "review_file_sha256": sha256(output / "review.csv"),
        "prepared_source_sha256": sha256(raw_path),
        "curation": curation,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/curated_events.jsonl")
    parser.add_argument("--output", default="artifacts/dataset")
    parser.add_argument("--allow-partial", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
