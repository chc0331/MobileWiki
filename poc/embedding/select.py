"""Select a LoRA epoch using development data, without reading test labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from common import STUDENT_MODEL, dataset_clustering_deidentify, read_jsonl, verified_dataset_manifest
from clustering import ClusterConfig, choose_checkpoint, cluster_groups, load_profile, save_profile, select_profile
from evaluate import dense_search, document_vectors, encode_documents, groups_for, model_fingerprint, pair_f1, select_cluster_threshold


def select(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    manifest = verified_dataset_manifest(dataset)
    run = Path(args.run)
    provenance = run / "training.json"
    training = json.loads(provenance.read_text(encoding="utf-8"))
    all_documents = read_jsonl(dataset / "documents.jsonl")
    train_documents = [row for row in all_documents if row["split"] == "train"]
    documents = [row for row in all_documents if row["split"] == "dev"]
    queries = [row for row in read_jsonl(dataset / "queries.jsonl") if row["split"] == "dev"]
    pairs = [row for row in read_jsonl(dataset / "pairs.jsonl") if row["split"] == "dev"]
    if not documents or not pairs or not queries:
        raise ValueError("Development data is incomplete")
    checkpoints = sorted((run / "checkpoints").glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    if not checkpoints:
        raise ValueError("No training checkpoints found")
    results = []
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else dataset.parent / "embedding_cache"
    deidentify_arg = getattr(args, "deidentify", "auto")
    deidentify = deidentify_arg == "on" if deidentify_arg != "auto" else dataset_clustering_deidentify(manifest)
    for checkpoint in checkpoints:
        model = SentenceTransformer(
            STUDENT_MODEL,
            revision=training["base_revision"],
            device="mps",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32},
            default_prompt_name=None,
        )
        model.load_adapter(str(checkpoint))
        model.max_seq_length = training["settings"]["max_length"]
        model_key = model_fingerprint(training["base_revision"], checkpoint)
        chunks, chunk_ids = encode_documents(model, documents, "clustering", args.batch_size, cache_dir, model_key, deidentify)
        vectors = document_vectors(chunks, chunk_ids, documents)
        profile_path = None
        if getattr(args, "hybrid", False):
            train_chunks, train_chunk_ids = encode_documents(model, train_documents, "clustering", args.batch_size, cache_dir, model_key, deidentify)
            train_vectors = document_vectors(train_chunks, train_chunk_ids, train_documents)
            profile, threshold_candidates = select_profile(train_vectors, documents, vectors)
            best_cluster = profile["selection"]
            selected_groups = cluster_groups(documents, vectors, ClusterConfig.from_dict(profile["config"]), profile["fitted"])
            profile_path = run / "profiles" / f"{checkpoint.name}.json"
            save_profile(profile_path, profile, {
                "base_revision": training["base_revision"], "model_key": model_key,
                "clustering_deidentify": deidentify, "dataset_files": manifest["files"],
                "fit_split": "train", "selection_split": "dev",
                "training_document_ids_sha256": hashlib.sha256(json.dumps([doc["doc_id"] for doc in train_documents]).encode()).hexdigest(),
            })
        else:
            best_cluster, threshold_candidates = select_cluster_threshold(documents, vectors)
            selected_groups = groups_for(documents, vectors, best_cluster["threshold"])
        sampled_pair = pair_f1(pairs, selected_groups)
        search = dense_search(model, documents, queries, args.batch_size, cache_dir, model_key)
        results.append(
            {
                "checkpoint": checkpoint.name,
                "threshold": best_cluster["threshold"],
                "all_event_pair_f1": best_cluster["f1"],
                "all_event_pair_precision": best_cluster["precision"],
                "all_event_pair_recall": best_cluster["recall"],
                "all_event_pair_false_merges": best_cluster["fp"],
                "fp": best_cluster["fp"],
                "sampled_pair_f1": sampled_pair["f1"],
                "threshold_candidates": threshold_candidates,
                "search_recall_at_5": search["recall_at_5"],
                "search_mrr": search["mrr"],
                "profile_path": str(profile_path.resolve()) if profile_path else None,
                "model_key": model_key,
                "clustering_deidentify": deidentify,
            }
        )
        del model
        gc.collect()
        torch.mps.empty_cache()
    chosen = choose_checkpoint(results)
    selected_path = run / "checkpoints" / chosen["checkpoint"]
    adapter_path = run / "adapter"
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    shutil.copytree(
        selected_path,
        adapter_path,
        ignore=shutil.ignore_patterns("trainer_state.json", "training_args.bin", "scheduler.pt", "optimizer.pt", "rng_state.pth"),
    )
    if chosen["profile_path"]:
        frozen_profile = load_profile(chosen["profile_path"])
        final_profile_path = run / "cluster_profile.json"
        save_profile(final_profile_path, frozen_profile)
        chosen["profile_path"] = str(final_profile_path.resolve())
    training["development_selection"] = {
        "policy": "dev Recall@5 >= 0.90, then full event-pair F1, fewer false merges, MRR, earlier checkpoint; unmet candidates remain provisional",
        "candidates": results, "selected": chosen,
    }
    provenance.write_text(json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "training_report": str(provenance.resolve()),
        "cluster_profile": chosen["profile_path"],
        "cluster_config": frozen_profile["config"] if chosen["profile_path"] else ClusterConfig(threshold=chosen["threshold"]).__dict__,
        "selection_status": chosen["selection_status"],
        "selected": {key: value for key, value in chosen.items() if key != "threshold_candidates"},
        "candidates": [
            {key: value for key, value in row.items() if key != "threshold_candidates"}
            for row in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="artifacts/trained")
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hybrid", action="store_true", help="Select a train-fitted hybrid cluster profile on development data")
    parser.add_argument("--deidentify", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cache-dir")
    select(parser.parse_args())


if __name__ == "__main__":
    main()
