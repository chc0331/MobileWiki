"""Select a LoRA epoch using development data, without reading test labels."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from common import STUDENT_MODEL, read_jsonl
from evaluate import THRESHOLDS, dense_search, document_vectors, encode_documents, groups_for, pair_f1


def select(args: argparse.Namespace) -> None:
    run = Path(args.run)
    provenance = run / "training.json"
    training = json.loads(provenance.read_text(encoding="utf-8"))
    dataset = Path(args.dataset)
    documents = [row for row in read_jsonl(dataset / "documents.jsonl") if row["split"] == "dev"]
    queries = [row for row in read_jsonl(dataset / "queries.jsonl") if row["split"] == "dev"]
    pairs = [row for row in read_jsonl(dataset / "pairs.jsonl") if row["split"] == "dev"]
    if not documents or not pairs or not queries:
        raise ValueError("Development data is incomplete")
    checkpoints = sorted((run / "checkpoints").glob("checkpoint-*"))
    if not checkpoints:
        raise ValueError("No training checkpoints found")
    results = []
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
        chunks, chunk_ids = encode_documents(model, documents, "clustering", args.batch_size)
        vectors = document_vectors(chunks, chunk_ids, documents)
        pair_results = [
            {"threshold": threshold, **pair_f1(pairs, groups_for(documents, vectors, threshold))}
            for threshold in THRESHOLDS
        ]
        best_pair = max(pair_results, key=lambda row: (row["f1"], -row["threshold"]))
        search = dense_search(model, documents, queries, args.batch_size)
        results.append(
            {
                "checkpoint": checkpoint.name,
                "threshold": best_pair["threshold"],
                "pair_f1": best_pair["f1"],
                "search_recall_at_5": search["recall_at_5"],
                "search_mrr": search["mrr"],
                "selection_score": best_pair["f1"] + search["mrr"],
            }
        )
        del model
        gc.collect()
        torch.mps.empty_cache()
    chosen = max(results, key=lambda row: (row["selection_score"], -int(row["checkpoint"].split("-")[-1])))
    selected_path = run / "checkpoints" / chosen["checkpoint"]
    adapter_path = run / "adapter"
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    shutil.copytree(
        selected_path,
        adapter_path,
        ignore=shutil.ignore_patterns("trainer_state.json", "training_args.bin", "scheduler.pt", "optimizer.pt", "rng_state.pth"),
    )
    training["development_selection"] = {"candidates": results, "selected": chosen}
    provenance.write_text(json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(training["development_selection"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="artifacts/trained")
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--batch-size", type=int, default=2)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
