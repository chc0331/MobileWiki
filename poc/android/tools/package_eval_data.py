#!/usr/bin/env python3
"""Package fixed test data and isolated demo notes for the Android POC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "poc/embedding/artifacts/dataset"
TRAINING = ROOT / "poc/embedding/artifacts/trained/training.json"
OUTPUT = ROOT / "poc/android/app/src/main/assets"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    dataset_manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    documents = [
        {key: row[key] for key in ("doc_id", "event_id", "title", "text")}
        for row in read_jsonl(DATASET / "documents.jsonl")
        if row["split"] == "test"
    ]
    queries = [
        {key: row[key] for key in ("query_id", "text", "relevant_doc_ids")}
        for row in read_jsonl(DATASET / "queries.jsonl")
        if row["split"] == "test"
    ][:140]
    pairs = [
        {key: row[key] for key in ("left_id", "right_id", "related")}
        for row in read_jsonl(DATASET / "pairs.jsonl")
        if row["split"] == "test"
    ]
    if (len(documents), len(queries), len(pairs)) != (1000, 140, 100):
        raise SystemExit(f"Unexpected fixed test set size: {len(documents)}, {len(queries)}, {len(pairs)}")

    selected = training.get("development_selection", {}).get("selected")
    if not selected or selected.get("checkpoint") != "checkpoint-250":
        raise SystemExit("Development-selected adapter checkpoint is missing")

    event_ids = sorted({row["event_id"] for row in read_jsonl(DATASET / "documents.jsonl") if row["split"] == "train"})[:4]
    sample_docs = [
        {"doc_id": row["doc_id"].replace("train-", "sample-"),
         "event_id": row["event_id"].replace("train-", "sample-"),
         "title": row["title"], "text": row["text"]}
        for row in read_jsonl(DATASET / "documents.jsonl")
        if row["split"] == "train" and row["event_id"] in event_ids
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    publish = json.loads((ROOT / "poc/embedding/artifacts/hub_publish.json").read_text(encoding="utf-8"))
    eval_manifest = {
        "status": dataset_manifest["status"],
        "dataset_id": publish["dataset_repo"],
        "dataset_revision": publish["dataset_revision"],
        "source_manifest_sha256": digest(DATASET / "manifest.json"),
        "source_file_sha256": dataset_manifest["files"],
        "counts": {"documents": len(documents), "queries": len(queries), "pairs": len(pairs)},
        "query_limit": 140,
        "cluster_threshold": selected["threshold"],
        "threshold_selection": "development full-event pair F1; ties prefer lower threshold",
        "selected_checkpoint": selected["checkpoint"],
        "human_review": "pending; all automatic metrics are provisional",
    }
    (OUTPUT / "evaluation.json").write_text(
        json.dumps({"manifest": eval_manifest, "documents": documents, "queries": queries, "pairs": pairs}, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "sample_notes.json").write_text(
        json.dumps(sample_docs, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    asset_manifest = {
        **eval_manifest,
        "evaluation_json_sha256": digest(OUTPUT / "evaluation.json"),
        "sample_notes_sha256": digest(OUTPUT / "sample_notes.json"),
    }
    (OUTPUT / "evaluation_manifest.json").write_text(
        json.dumps(asset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asset_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
