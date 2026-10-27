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
    matching_doc_text,
    read_jsonl,
    search_query_text,
)


def training_examples(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    documents = {row["doc_id"]: row for row in read_jsonl(dataset_dir / "documents.jsonl")}
    queries = {row["query_id"]: row for row in read_jsonl(dataset_dir / "queries.jsonl")}
    output = []
    for item in read_jsonl(dataset_dir / "triplets.jsonl")[:limit]:
        if item["task"] == "search":
            anchor = search_query_text(queries[item["anchor_id"]]["text"])
            positive = matching_doc_text(documents[item["positive_id"]])
            negative = matching_doc_text(documents[item["negative_id"]])
        else:
            anchor = clustering_text(documents[item["anchor_id"]])
            positive = clustering_text(documents[item["positive_id"]])
            negative = clustering_text(documents[item["negative_id"]])
        output.append({"anchor": anchor, "positive": positive, "negative": negative})
    if not output:
        raise ValueError("No training triplets found")
    return output


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
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
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_revision = HfApi().model_info(STUDENT_MODEL).sha
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
        target_modules=["q_proj", "v_proj"],
    )
    model.add_adapter(lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable}", flush=True)
    examples = Dataset.from_list(training_examples(dataset_dir, args.limit))
    loss = TripletLoss(
        model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=0.1,
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
        "dataset_manifest": json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8")),
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
            "target_modules": ["q_proj", "v_proj"],
            "triplet_margin": 0.1,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
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
    train(parser.parse_args())


if __name__ == "__main__":
    main()
