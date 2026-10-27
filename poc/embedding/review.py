"""Apply completed human review of the fixed search and pairwise gold labels."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from common import read_jsonl, sha256, write_jsonl
from prepare import write_review
from validate import validate_dataset


def apply_review(folder: Path) -> None:
    review_path = folder / "review.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 240 or len({row["item_id"] for row in rows}) != 240:
        raise ValueError("Expected 140 search questions and 100 pairs in the review sheet")
    queries = read_jsonl(folder / "queries.jsonl")
    pairs = read_jsonl(folder / "pairs.jsonl")
    with tempfile.TemporaryDirectory(dir=folder.parent) as temporary:
        expected_path = Path(temporary) / "review.csv"
        write_review(expected_path, {
            "documents": read_jsonl(folder / "documents.jsonl"),
            "queries": queries,
            "pairs": pairs,
        })
        with expected_path.open(encoding="utf-8", newline="") as stream:
            expected = {row["item_id"]: row for row in csv.DictReader(stream)}
    fixed_fields = ("kind", "item_id", "question_or_left", "evidence_or_right", "gold")
    if any(
        row["item_id"] not in expected
        or any(row[field] != expected[row["item_id"]][field] for field in fixed_fields)
        for row in rows
    ):
        raise ValueError("Review sheet content differs from the dataset; fix source data and regenerate the sheet")
    incomplete = [row["item_id"] for row in rows if row["accepted"].strip().lower() not in {"yes", "y"}]
    if incomplete:
        raise ValueError(f"Review is incomplete or rejected for {len(incomplete)} items")
    reviewed_ids = {row["item_id"] for row in rows}
    test_queries = [row for row in queries if row["split"] == "test"]
    expected_ids = {row["query_id"] for row in test_queries[:140]}
    expected_ids |= {row["pair_id"] for row in pairs if row["split"] == "test"}
    if reviewed_ids != expected_ids:
        raise ValueError("Review IDs do not match the fixed evaluation items")
    for query in queries:
        if query["query_id"] in reviewed_ids:
            query["reviewed"] = True
    for pair in pairs:
        if pair["pair_id"] in reviewed_ids:
            pair["reviewed"] = True
    write_jsonl(folder / "queries.jsonl", queries)
    write_jsonl(folder / "pairs.jsonl", pairs)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "human_reviewed"
    manifest["files"] = {path.name: sha256(path) for path in folder.glob("*.jsonl")}
    manifest["review_file_sha256"] = sha256(review_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_dataset(folder)
    print("Review applied to 140 queries and 100 pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    args = parser.parse_args()
    apply_review(Path(args.dataset))


if __name__ == "__main__":
    main()
