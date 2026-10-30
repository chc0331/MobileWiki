"""Measure a train-fitted character TF-IDF event-clustering baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from clustering import all_event_pair_f1
from common import (
    clustering_text,
    code_fingerprint,
    dataset_clustering_deidentify,
    json_sha256,
    read_jsonl,
    sha256,
    verified_dataset_manifest,
)

THRESHOLDS = tuple(round(value * 0.05, 2) for value in range(1, 20))
LINKAGES = ("average", "complete")
FEATURE_CONFIG = {"analyzer": "char", "ngram_range": [2, 5], "dtype": "float32", "norm": "l2"}


def feature_texts(documents: list[dict], deidentify: bool | None = None) -> list[str]:
    """Use the same textual clustering input, without event or family labels."""
    return [clustering_text(document, deidentify=deidentify) for document in documents]


def fit_vectorizer(train_documents: list[dict], deidentify: bool | None = None) -> TfidfVectorizer:
    if not train_documents or any(document["split"] != "train" for document in train_documents):
        raise ValueError("The lexical vocabulary and IDF require training documents only")
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), dtype=np.float32, norm="l2").fit(
        feature_texts(train_documents, deidentify)
    )


def lexical_distances(vectorizer: TfidfVectorizer, documents: list[dict], deidentify: bool | None = None) -> np.ndarray:
    if not documents:
        return np.empty((0, 0), dtype=np.float32)
    # Keep the document-by-feature matrix sparse; only pairwise distances are dense.
    vectors = vectorizer.transform(feature_texts(documents, deidentify))
    distances = cosine_distances(vectors)
    distances = np.clip((distances + distances.T) / 2, 0, 2)
    np.fill_diagonal(distances, 0)
    return distances


def groups_at_threshold(documents: list[dict], distances: np.ndarray, linkage: str, threshold: float) -> dict[str, int]:
    if len(documents) < 2:
        return {document["doc_id"]: 0 for document in documents}
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="precomputed", linkage=linkage,
    ).fit_predict(distances)
    return {document["doc_id"]: int(label) for document, label in zip(documents, labels)}


def select_development(dev_documents: list[dict], distances: np.ndarray) -> tuple[dict, list[dict]]:
    if not dev_documents or any(document["split"] != "dev" for document in dev_documents):
        raise ValueError("Lexical clustering selection requires development documents only")
    candidates = []
    for linkage in LINKAGES:
        for threshold in THRESHOLDS:
            groups = groups_at_threshold(dev_documents, distances, linkage, threshold)
            candidates.append({
                "order": len(candidates), "linkage": linkage, "threshold": threshold,
                **all_event_pair_f1(dev_documents, groups),
            })
    selected = max(candidates, key=lambda row: (row["f1"], -row["fp"], -row["order"]))
    return selected, candidates


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    dataset = Path(args.dataset).resolve()
    manifest = verified_dataset_manifest(dataset)
    deidentify = dataset_clustering_deidentify(manifest)
    documents = read_jsonl(dataset / "documents.jsonl")
    splits = {split: [document for document in documents if document["split"] == split] for split in ("train", "dev", "test")}
    if any(not rows for rows in splits.values()):
        raise ValueError("Train, development, and test documents are required")
    vectorizer = fit_vectorizer(splits["train"], deidentify)
    selected, candidates = select_development(splits["dev"], lexical_distances(vectorizer, splits["dev"], deidentify))
    test_groups = groups_at_threshold(
        splits["test"], lexical_distances(vectorizer, splits["test"], deidentify), selected["linkage"], selected["threshold"],
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_pending_human_review" if manifest.get("status") != "human_reviewed" else "human_reviewed",
        "model": "character_tfidf_lexical_clustering_baseline",
        "purpose": "Measure literal character overlap; a high score does not establish semantic generalization",
        "dataset_path": str(dataset),
        "dataset_revision": manifest.get("dataset_revision"),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "dataset_files": manifest["files"],
        "feature_config": FEATURE_CONFIG,
        "fit_scope": "train_only",
        "fit_rows": len(splits["train"]),
        "training_input_sha256": json_sha256(feature_texts(splits["train"], deidentify)),
        "training_row_order_sha256": json_sha256([document["doc_id"] for document in splits["train"]]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "vocabulary_sha256": json_sha256(vectorizer.get_feature_names_out().tolist()),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f4").tobytes()).hexdigest(),
        "deidentify": "dataset_policy",
        "clustering_deidentify": deidentify,
        "selection_scope": "development_only",
        "selection_tie_break": ["higher_f1", "fewer_false_merges", "earlier_grid_order"],
        "dev_selected": selected,
        "dev_candidates": candidates,
        "test_all_event_pairs": all_event_pair_f1(splits["test"], test_groups),
        "test_group_count": len(set(test_groups.values())),
        "test_assignments": test_groups,
        "code_files": code_fingerprint(Path(__file__).resolve().parent, ("lexical_baseline.py", "common.py", "clustering.py", "uv.lock")),
        "environment": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "dev_selected": selected, "test_all_event_pairs": report["test_all_event_pairs"]}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
