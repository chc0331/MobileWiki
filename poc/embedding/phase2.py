"""Preserve fixed v2 data and derive the phase-2 clustering training recipe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, sha256, verified_dataset_manifest, write_jsonl

REVISION = "shortcut-hardneg-v2-cluster-phase2"
SOURCE_REVISION = "shortcut-hardneg-v2-balanced"
TRAINING_POLICY = "cluster-hardneg-v3-on-v2"
PRESERVED_FILES = ("documents.jsonl", "queries.jsonl", "pairs.jsonl", "review.csv")
SOURCE_FILES = ("source.manifest.json", "triplets.source.jsonl")


def _rank(*values) -> str:
    return hashlib.sha256(json.dumps([42, *values], ensure_ascii=False).encode()).hexdigest()


def _contains_fact(document: dict) -> bool:
    return bool(document["fact_value"]) and document["fact_value"] in document["title"] + "\n" + document["text"]


def make_triplets(documents: list[dict], queries: list[dict]) -> list[dict]:
    events = defaultdict(list)
    for document in documents:
        if document["split"] == "train":
            events[document["event_id"]].append(document)
    query_map = {query["event_id"]: query for query in queries if query["split"] == "train"}
    if len(events) != 200 or len(query_map) != 200 or set(events) != set(query_map):
        raise ValueError("Phase 2 requires all 200 fixed v2 training events and queries")
    answers = {}
    for event_id, members in events.items():
        members.sort(key=lambda doc: doc["doc_id"])
        if len(members) != 4 or len({doc["doc_id"] for doc in members}) != 4:
            raise ValueError("Phase 2 requires four distinct documents per training event")
        answer_id = query_map[event_id]["evidence_doc_id"]
        answers[event_id] = next(doc for doc in members if doc["doc_id"] == answer_id)
        if not _contains_fact(answers[event_id]):
            raise ValueError("The assigned answer document must contain its actual fact")
    pools, categories = {}, defaultdict(list)
    for event_id, answer in answers.items():
        categories[answer["category"]].append(event_id)
        pools[event_id] = {
            "A": [other for other in answers.values() if other["event_id"] != event_id and other["family_id"] == answer["family_id"]],
            "B": [other for other in answers.values() if other["family_id"] != answer["family_id"] and other["category"] == answer["category"] and other["fact_value"] != answer["fact_value"]],
            "C": [other for other in answers.values() if other["family_id"] != answer["family_id"] and other["fact_value"] == answer["fact_value"] and _contains_fact(other)],
        }
        for kind, candidates in pools[event_id].items():
            candidates.sort(key=lambda other: (kind == "B" and other["date"] != answer["date"], _rank(event_id, kind, other["event_id"])))
        if not pools[event_id]["A"] or not pools[event_id]["B"]:
            raise ValueError(f"No valid A/B negative pool for {event_id}")
    c_events = set()
    if len(categories) != 10 or any(len(members) != 20 for members in categories.values()):
        raise ValueError("Phase 2 requires twenty training events in each of ten topics")
    for category, members in categories.items():
        eligible = sorted((event_id for event_id in members if pools[event_id]["C"]), key=lambda event_id: _rank(category, "C-events", event_id))
        if len(eligible) < 16:
            raise ValueError(f"Fewer than sixteen observed-fact C events in {category}")
        c_events.update(eligible[:16])
    triplets = []
    for event_id in sorted(events):
        answer, members, query = answers[event_id], events[event_id], query_map[event_id]
        within = next(doc for doc in members if doc["doc_id"] != answer["doc_id"])
        for kind, negative in (("within_event", within), ("same_family_other_event", pools[event_id]["A"][0])):
            triplets.append({"task": "search", "anchor_id": query["query_id"], "positive_id": answer["doc_id"], "negative_id": negative["doc_id"], "negative_type": kind})
        kinds = ["A", "A", "B", "C"] if event_id in c_events else ["A", "A", "B", "B"]
        random.Random(_rank(event_id, "slots")).shuffle(kinds)
        if event_id in c_events:
            answer_slot = next(index for index, doc in enumerate(members) if doc["doc_id"] == answer["doc_id"])
            c_slot = kinds.index("C")
            kinds[c_slot], kinds[answer_slot] = kinds[answer_slot], kinds[c_slot]
        used = Counter()
        for slot, kind in enumerate(kinds):
            candidates = pools[event_id][kind]
            other = candidates[used[kind] % len(candidates)]
            used[kind] += 1
            negative = other if kind == "C" else events[other["event_id"]][slot]
            triplets.append({"task": "clustering", "anchor_id": members[slot]["doc_id"], "positive_id": members[(slot + 1) % 4]["doc_id"], "negative_id": negative["doc_id"], "negative_type": kind})
    return triplets


def _triplet_hash(rows: list[dict]) -> str:
    return hashlib.sha256("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()).hexdigest()


def validate_triplets(triplets: list[dict], documents: list[dict], queries: list[dict], expected: list[dict]) -> None:
    docs = {doc["doc_id"]: doc for doc in documents}
    query_map = {query["query_id"]: query for query in queries}
    if Counter(row["task"] for row in triplets) != Counter(search=400, clustering=800):
        raise ValueError("Phase-2 training requires 400 search and 800 clustering triplets")
    if Counter(row["negative_type"] for row in triplets if row["task"] == "clustering") != Counter(A=400, B=240, C=160):
        raise ValueError("Phase-2 clustering requires A/B/C counts 400/240/160")
    anchors = Counter(row["anchor_id"] for row in triplets if row["task"] == "clustering")
    if anchors != Counter(doc["doc_id"] for doc in documents if doc["split"] == "train"):
        raise ValueError("Every training document must be a clustering anchor exactly once")
    for row, original in zip(triplets, expected):
        if any(row.get(key) != original[key] for key in ("task", "anchor_id", "positive_id", "negative_type")):
            raise ValueError("Phase-2 anchor, positive, or type differs from its fixed recipe")
        positive, negative = docs[row["positive_id"]], docs[row["negative_id"]]
        anchor = query_map[row["anchor_id"]] if row["task"] == "search" else docs[row["anchor_id"]]
        if any(item["split"] != "train" for item in (anchor, positive, negative)):
            raise ValueError("Phase-2 triplets must use training rows only")
        kind = row["negative_type"]
        if row["task"] == "search":
            if positive["doc_id"] != anchor["evidence_doc_id"] or anchor["relevant_doc_ids"] != [positive["doc_id"]] or negative["doc_id"] in anchor["relevant_doc_ids"]:
                raise ValueError("Phase-2 search evidence or negative is invalid")
            correct = (negative["event_id"] == positive["event_id"] if kind == "within_event" else
                       negative["event_id"] != positive["event_id"] and negative["family_id"] == positive["family_id"])
        else:
            if anchor["event_id"] != positive["event_id"] or anchor["doc_id"] == positive["doc_id"] or negative["event_id"] == anchor["event_id"]:
                raise ValueError("Phase-2 clustering positive/negative event relation is invalid")
            correct = {"A": negative["family_id"] == anchor["family_id"],
                       "B": negative["family_id"] != anchor["family_id"] and negative["category"] == anchor["category"] and negative["fact_value"] != anchor["fact_value"],
                       "C": negative["family_id"] != anchor["family_id"] and negative["fact_value"] == anchor["fact_value"] and _contains_fact(anchor) and _contains_fact(negative)}[kind]
        if not correct:
            raise ValueError(f"Phase-2 negative does not satisfy type {kind}")


def validate_dataset(folder: Path, partial: bool = False) -> dict:
    if partial:
        raise ValueError("Phase 2 requires the complete fixed v2 dataset")
    manifest = verified_dataset_manifest(folder)
    if manifest.get("dataset_revision") != REVISION or manifest.get("training_policy") != TRAINING_POLICY or manifest.get("clustering_deidentify") is not True:
        raise ValueError("Phase-2 revision, training policy, and explicit name masking are required")
    if not set((*PRESERVED_FILES, *SOURCE_FILES)) <= manifest["files"].keys():
        raise ValueError("Phase-2 manifest is missing preserved/source file hashes")
    source = json.loads((folder / "source.manifest.json").read_text())
    proof = manifest["source_proof"]
    if source.get("dataset_revision") != SOURCE_REVISION or proof["manifest_sha256"] != sha256(folder / "source.manifest.json") or proof["files"] != source["files"] or proof["review_file_sha256"] != source["review_file_sha256"]:
        raise ValueError("Phase-2 original manifest proof differs from the preserved source")
    if proof["recipe_file"] != "phase2.py" or proof["recipe_sha256"] != sha256(Path(__file__)):
        raise ValueError("Phase-2 recipe file hash differs from its recorded version")
    for name in PRESERVED_FILES:
        expected_hash = source["review_file_sha256"] if name == "review.csv" else source["files"][name]
        if sha256(folder / name) != expected_hash:
            raise ValueError(f"Phase-2 source bytes changed: {name}")
    if sha256(folder / "triplets.source.jsonl") != source["files"]["triplets.jsonl"]:
        raise ValueError("Phase-2 original triplet backup differs from the source")
    if manifest["review_file_sha256"] != source["review_file_sha256"] or manifest["source_status"] != source.get("status"):
        raise ValueError("Phase-2 source review/status provenance differs")
    # Reuse every original v2 content/pair check in a temporary, unchanged source view.
    from validate import validate_dataset as validate_original
    with tempfile.TemporaryDirectory(prefix="embedding-phase2-source-") as temporary:
        view = Path(temporary)
        for name in PRESERVED_FILES:
            shutil.copyfile(folder / name, view / name)
        shutil.copyfile(folder / "source.manifest.json", view / "manifest.json")
        shutil.copyfile(folder / "triplets.source.jsonl", view / "triplets.jsonl")
        summary = validate_original(view)
    documents, queries = read_jsonl(folder / "documents.jsonl"), read_jsonl(folder / "queries.jsonl")
    expected = make_triplets(documents, queries)
    initial_hash = _triplet_hash(expected)
    if proof["initial_triplets_sha256"] != initial_hash:
        raise ValueError("Phase-2 initial triplet hash differs from the fixed recipe")
    counts = manifest.get("counts", {})
    if counts.get("triplets") != 1200 or counts.get("triplets_by_task") != {"search": 400, "clustering": 800} or counts.get("clustering_negative_types") != {"A": 400, "B": 240, "C": 160}:
        raise ValueError("Phase-2 manifest training counts differ from the fixed recipe")
    mining = manifest.get("mining")
    if mining:
        if "triplets.initial.jsonl" not in manifest["files"] or sha256(folder / "triplets.initial.jsonl") != initial_hash or mining.get("source_triplets_sha256") != initial_hash:
            raise ValueError("Phase-2 mining initial triplets differ from the fixed recipe")
        if mining.get("scope") != "train_only" or mining.get("clustering_deidentify") is not True:
            raise ValueError("Phase-2 mining must preserve train-only scope and name masking")
    elif sha256(folder / "triplets.jsonl") != initial_hash:
        raise ValueError("Unmined phase-2 triplets differ from the fixed recipe")
    triplets = read_jsonl(folder / "triplets.jsonl")
    validate_triplets(triplets, documents, queries, expected)
    summary.update(triplets=len(triplets), clustering_negative_types={"A": 400, "B": 240, "C": 160}, training_policy=TRAINING_POLICY)
    return summary


def build(args: argparse.Namespace) -> None:
    source, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Phase-2 output already exists: {output}")
    original = verified_dataset_manifest(source)
    if original.get("dataset_revision") != SOURCE_REVISION:
        raise ValueError("Phase 2 requires the original fixed shortcut-hardneg-v2-balanced dataset")
    source_manifest_hash = sha256(source / "manifest.json")
    triplets = make_triplets(read_jsonl(source / "documents.jsonl"), read_jsonl(source / "queries.jsonl"))
    output.mkdir(parents=True)
    for name in PRESERVED_FILES:
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "manifest.json", output / "source.manifest.json")
    shutil.copyfile(source / "triplets.jsonl", output / "triplets.source.jsonl")
    write_jsonl(output / "triplets.jsonl", triplets)
    manifest = copy.deepcopy(original)
    manifest.update(dataset_revision=REVISION, training_policy=TRAINING_POLICY, clustering_deidentify=True,
                    created_at=datetime.now(timezone.utc).isoformat(), status="provisional_pending_human_review",
                    source_status=original.get("status"),
                    search_negative_policy="one within-event and one same-family other-event negative per query",
                    clustering_negative_policy="A 400 same-family other-event; B 240 same-topic other-family different-fact; C 160 other-family same observed fact")
    manifest["source_proof"] = {"dataset_path": str(source), "manifest_sha256": source_manifest_hash,
                                "files": original["files"], "review_file_sha256": original["review_file_sha256"],
                                "recipe_file": "phase2.py", "recipe_sha256": sha256(Path(__file__)),
                                "initial_triplets_sha256": sha256(output / "triplets.jsonl")}
    manifest["counts"].update(triplets=1200, triplets_by_task={"search": 400, "clustering": 800}, clustering_negative_types={"A": 400, "B": 240, "C": 160})
    manifest["files"] = {name: sha256(output / name) for name in (*PRESERVED_FILES, *SOURCE_FILES, "triplets.jsonl")}
    manifest.setdefault("code_files", {})["phase2.py"] = manifest["source_proof"]["recipe_sha256"]
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if sha256(source / "manifest.json") != source_manifest_hash or verified_dataset_manifest(source)["files"] != original["files"]:
        raise ValueError("Original v2 dataset changed during phase-2 preparation")
    print(json.dumps(validate_dataset(output), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Original fixed v2 dataset directory")
    parser.add_argument("--output", required=True, help="New derived dataset directory; never overwritten")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
