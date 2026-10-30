"""Mine difficult negatives from training documents without reading held-out vectors."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    STUDENT_MODEL, code_fingerprint, dataset_clustering_deidentify, elapsed_budget_seconds, generation_budget_path,
    json_sha256, read_jsonl, search_query_text, sha256, verified_dataset_manifest, write_jsonl,
)
from train import PINNED_BASE_REVISION
from budget import WallBudget

MINING_CODE_FILES = ("mine.py", "common.py", "evaluate.py", "clustering.py", "train.py", "uv.lock")


def _contains_fact(document: dict) -> bool:
    fact = document["fact_value"]
    return bool(fact) and fact in document["title"] + "\n" + document["text"]


def candidate_documents(triple: dict, documents: dict[str, dict], queries: dict[str, dict]) -> list[dict]:
    positive = documents[triple["positive_id"]]
    anchor = queries[triple["anchor_id"]] if triple["task"] == "search" else documents[triple["anchor_id"]]
    if anchor["split"] != "train" or positive["split"] != "train":
        raise ValueError("Mining anchors and positives must be training examples")
    pool = [row for row in documents.values() if row["split"] == "train"]
    if triple["task"] == "search":
        original_negative = documents[triple["negative_id"]]
        if original_negative["split"] != "train":
            raise ValueError("Mining search negatives must be training examples")
        if original_negative["event_id"] == positive["event_id"]:
            return [row for row in pool if row["event_id"] == positive["event_id"] and row["doc_id"] not in anchor["relevant_doc_ids"]]
        return [row for row in pool if row["family_id"] == positive["family_id"] and row["event_id"] != positive["event_id"]]
    pool = [row for row in pool if row["event_id"] != positive["event_id"]]
    kind = triple["negative_type"]
    if kind == "A":
        pool = [row for row in pool if row["family_id"] == positive["family_id"]]
        same_topic = [row for row in pool if row["category"] == positive["category"]]
        return same_topic or pool
    if kind == "B":
        pool = [row for row in pool if row["family_id"] != positive["family_id"] and row["category"] == positive["category"] and row["fact_value"] != positive["fact_value"]]
        same_date = [row for row in pool if row["date"] == positive["date"]]
        return same_date or pool
    if kind == "C":
        if anchor.get("card_revision") == "context-hardneg-v3" and not _contains_fact(anchor):
            raise ValueError("A v3 C mining anchor must contain its fact in the actual title or text")
        return [row for row in pool if row["family_id"] != positive["family_id"]
                and row["fact_value"] == positive["fact_value"] and _contains_fact(row)]
    raise ValueError(f"Unknown mining negative type: {kind}")


def mine_triplets(documents: list[dict], queries: list[dict], triples: list[dict], cluster_vectors: np.ndarray, search_vectors: np.ndarray, query_vectors: np.ndarray) -> list[dict]:
    if any(row["split"] != "train" for row in documents + queries):
        raise ValueError("Only training rows may be passed to the miner")
    docs = {row["doc_id"]: row for row in documents}
    query_map = {row["query_id"]: row for row in queries}
    doc_positions = {row["doc_id"]: index for index, row in enumerate(documents)}
    query_positions = {row["query_id"]: index for index, row in enumerate(queries)}
    used_events: dict[tuple[str, str], set[str]] = defaultdict(set)
    output = []
    for triple in triples:
        candidates = candidate_documents(triple, docs, query_map)
        if not candidates:
            raise ValueError(f"No valid negative pool: {triple}")
        if triple["task"] == "search":
            anchor_vector = query_vectors[query_positions[triple["anchor_id"]]]
            vectors = search_vectors
        else:
            anchor_vector = cluster_vectors[doc_positions[triple["anchor_id"]]]
            vectors = cluster_vectors
            key = (docs[triple["anchor_id"]]["event_id"], triple["negative_type"])
            unused = [row for row in candidates if row["event_id"] not in used_events[key]]
            candidates = unused or candidates
        scores = {row["doc_id"]: float(vectors[doc_positions[row["doc_id"]]] @ anchor_vector) for row in candidates}
        chosen = min(candidates, key=lambda row: (-scores[row["doc_id"]], row["doc_id"]))
        if triple["task"] == "clustering":
            used_events[key].add(chosen["event_id"])
        output.append({**triple, "negative_id": chosen["doc_id"], "mining_cosine": scores[chosen["doc_id"]], "mining_pool_documents": len(candidates)})
    return output


def _mine(args: argparse.Namespace, budget: WallBudget, source_manifest: dict, generation_path: Path | None) -> None:
    dataset, output = Path(args.dataset), Path(args.output)
    if verified_dataset_manifest(dataset)["files"] != source_manifest["files"]:
        raise ValueError("Mining source files changed before model loading")
    if output.exists():
        raise FileExistsError(f"Mining output already exists: {output}")
    project = Path(__file__).resolve().parent
    source_files = code_fingerprint(project, MINING_CODE_FILES)
    deidentify = dataset_clustering_deidentify(source_manifest)
    import torch
    from sentence_transformers import SentenceTransformer
    from evaluate import document_vectors, encode_documents

    if args.adapter:
        training = json.loads((Path(args.adapter).parent / "training.json").read_text(encoding="utf-8"))
        if training["base_revision"] != args.base_revision:
            raise ValueError("Mining adapter and pinned base revisions differ")
    docs = [row for row in read_jsonl(dataset / "documents.jsonl") if row["split"] == "train"]
    queries = [row for row in read_jsonl(dataset / "queries.jsonl") if row["split"] == "train"]
    model = SentenceTransformer(STUDENT_MODEL, revision=args.base_revision, device="mps", config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"dtype": torch.float32}, default_prompt_name=None)
    if args.adapter:
        model.load_adapter(args.adapter)
    model.max_seq_length = 512
    cluster_chunks, cluster_ids = encode_documents(model, docs, "clustering", args.batch_size, None, None, deidentify)
    cluster_vectors = document_vectors(cluster_chunks, cluster_ids, docs)
    search_chunks, search_ids = encode_documents(model, docs, "search", args.batch_size)
    search_vectors = document_vectors(search_chunks, search_ids, docs)
    query_vectors = np.asarray(model.encode([search_query_text(row["text"]) for row in queries], batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    triples = mine_triplets(docs, queries, read_jsonl(dataset / "triplets.jsonl"), cluster_vectors, search_vectors, query_vectors)
    if verified_dataset_manifest(dataset)["files"] != source_manifest["files"] or code_fingerprint(project, MINING_CODE_FILES) != source_files:
        raise ValueError("Mining source files or code changed during inference")
    shutil.copytree(dataset, output)
    shutil.copyfile(dataset / "triplets.jsonl", output / "triplets.initial.jsonl")
    initial_hash = sha256(output / "triplets.initial.jsonl")
    if initial_hash != source_manifest["files"]["triplets.jsonl"]:
        raise ValueError("Original mining triplet backup differs from the verified source")
    write_jsonl(output / "triplets.jsonl", triples)
    provenance = {
        "scope": "train_only", "base_model": STUDENT_MODEL, "base_revision": args.base_revision,
        "adapter": args.adapter, "source_triplets_sha256": source_manifest["files"]["triplets.jsonl"],
        "code_sha256": sha256(Path(__file__)), "examples": len(triples),
        "code_files": source_files, "code_fingerprint": json_sha256(source_files),
        "source_dataset_path": str(dataset.resolve()),
        "source_dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "generation_budget_path": str(generation_path) if generation_path else None,
        "clustering_deidentify": deidentify,
        "types": dict(Counter(row.get("negative_type", "search") for row in triples)),
        "elapsed_seconds": budget.elapsed_seconds, "budget_path": str(budget.path.resolve()),
    }
    if args.adapter:
        provenance["adapter_sha256"] = sha256(Path(args.adapter) / "adapter_model.safetensors")
    source_manifest["mining"] = provenance
    source_manifest["files"]["triplets.initial.jsonl"] = initial_hash
    source_manifest["files"]["triplets.jsonl"] = sha256(output / "triplets.jsonl")
    source_manifest["code_files"]["mine.py"] = provenance["code_sha256"]
    (output / "manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


def mine(args: argparse.Namespace) -> None:
    dataset, output = Path(args.dataset), Path(args.output)
    source_manifest = verified_dataset_manifest(dataset)
    generation_path = generation_budget_path(dataset, source_manifest, args.generation_budget)
    documents = {row["doc_id"]: row for row in read_jsonl(dataset / "documents.jsonl")}
    queries = {row["query_id"]: row for row in read_jsonl(dataset / "queries.jsonl")}
    for triple in read_jsonl(dataset / "triplets.jsonl"):
        if not candidate_documents(triple, documents, queries):
            raise ValueError(f"No valid negative pool: {triple}")
    prior = elapsed_budget_seconds(generation_path) if generation_path else 0
    with WallBudget(output.parent / f"{output.name}.mining_budget.json", args.max_hours, prior) as budget:
        _mine(args, budget, source_manifest, generation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--base-revision", default=PINNED_BASE_REVISION)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--generation-budget")
    mine(parser.parse_args())


if __name__ == "__main__":
    main()
