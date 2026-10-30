"""Fine-tune the EmbeddingGemma 2 text encoder with local MPS LoRA."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

from common import (
    SEED,
    STUDENT_MODEL,
    clustering_text,
    code_fingerprint,
    dataset_clustering_deidentify,
    generation_budget_path,
    json_sha256,
    matching_doc_text,
    read_jsonl,
    search_query_text,
    sha256,
    verified_dataset_manifest,
)

PINNED_BASE_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
LORA_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
TRAIN_CODE_FILES = ("train.py", "common.py", "uv.lock")


def lora_targets(value: str) -> list[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets or len(set(targets)) != len(targets) or set(targets) - LORA_MODULES:
        raise argparse.ArgumentTypeError("Use unique q_proj,k_proj,v_proj,o_proj module names")
    return targets


def training_examples(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    manifest_path = dataset_dir / "manifest.json"
    deidentify = dataset_clustering_deidentify(json.loads(manifest_path.read_text(encoding="utf-8"))) if manifest_path.exists() else None
    documents = {row["doc_id"]: row for row in read_jsonl(dataset_dir / "documents.jsonl")}
    queries = {row["query_id"]: row for row in read_jsonl(dataset_dir / "queries.jsonl")}
    output = []
    for item in read_jsonl(dataset_dir / "triplets.jsonl")[:limit]:
        if item["task"] == "search":
            source = queries[item["anchor_id"]]
            anchor = search_query_text(source["text"])
            positive = matching_doc_text(documents[item["positive_id"]])
            negative = matching_doc_text(documents[item["negative_id"]])
        elif item["task"] == "clustering":
            source = documents[item["anchor_id"]]
            anchor = clustering_text(source, deidentify=deidentify)
            positive = clustering_text(documents[item["positive_id"]], deidentify=deidentify)
            negative = clustering_text(documents[item["negative_id"]], deidentify=deidentify)
        else:
            raise ValueError(f"Unknown training task: {item['task']}")
        if any(row["split"] != "train" for row in (
            source, documents[item["positive_id"]], documents[item["negative_id"]]
        )):
            raise ValueError("Training examples must not include development or test documents")
        output.append({"anchor": anchor, "positive": positive, "negative": negative})
    if not output:
        raise ValueError("No training triplets found")
    return output


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset)
    dataset_manifest = verified_dataset_manifest(dataset_dir)
    deidentify = dataset_clustering_deidentify(dataset_manifest)
    generation_path = generation_budget_path(dataset_dir, dataset_manifest, getattr(args, "generation_budget", None))
    example_rows = training_examples(dataset_dir, args.limit)
    source_files = code_fingerprint(Path(__file__).resolve().parent, TRAIN_CODE_FILES)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import TripletDistanceMetric, TripletLoss
    from transformers import TrainerCallback

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this experiment requires the local M2 GPU")
    torch.manual_seed(SEED)
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_revision = args.base_revision
    model = SentenceTransformer(
        STUDENT_MODEL,
        revision=base_revision,
        device="mps",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32},
        default_prompt_name=None,
    )
    model.max_seq_length = args.max_length
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=args.lora_targets,
    )
    model.add_adapter(lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable}", flush=True)
    examples = Dataset.from_list(example_rows)
    loss = TripletLoss(
        model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=args.triplet_margin,
    )

    class TimeBudget(TrainerCallback):
        hit = False

        def on_step_end(self, training_args, state, control, **kwargs):
            if time.monotonic() - start >= args.max_hours * 3600:
                self.hit = True
                control.should_training_stop = True
                control.should_save = True
            return control

    settings = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=0.1,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        fp16=False,
        bf16=False,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )
    budget_callback = TimeBudget()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=settings,
        train_dataset=examples,
        loss=loss,
        callbacks=[budget_callback],
    )
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    report = {
        "status": "time_budget_reached" if budget_callback.hit else "trained",
        "started_at": started_at,
        "base_model": STUDENT_MODEL,
        "base_revision": base_revision,
        "dataset_manifest": dataset_manifest,
        "code_sha256": sha256(Path(__file__)),
        "training_code_files": source_files,
        "training_code_fingerprint": json_sha256(source_files),
        "training_input_sha256": json_sha256(example_rows),
        "generation_budget_path": str(generation_path) if generation_path else None,
        "trainable_parameters": trainable,
        "steps": result.global_step,
        "training_loss": result.training_loss,
        "elapsed_seconds": time.monotonic() - start,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "settings": {
            "dtype": "float32",
            "device": "mps",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": args.lora_targets,
            "triplet_margin": args.triplet_margin,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "limit": args.limit,
            "clustering_deidentify": deidentify,
        },
    }
    (output_dir / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter: {adapter_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--output", default="artifacts/trained")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--base-revision", default=PINNED_BASE_REVISION)
    parser.add_argument("--generation-budget", help="Generation ledger override; v3 otherwise discovers it from the dataset")
    parser.add_argument("--triplet-margin", type=float, default=0.1)
    parser.add_argument("--lora-targets", type=lora_targets, default=lora_targets("q_proj,v_proj"))
    args = parser.parse_args()
    if not 0 < args.triplet_margin <= 2:
        parser.error("--triplet-margin must be between 0 and 2")
    train(args)


if __name__ == "__main__":
    main()
