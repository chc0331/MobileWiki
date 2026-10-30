"""Run the four predeclared LoRA candidates and select them on development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import code_fingerprint, elapsed_budget_seconds, generation_budget_path, json_sha256, sha256, verified_dataset_manifest
from train import PINNED_BASE_REVISION, TRAIN_CODE_FILES, training_examples

CANDIDATES = (
    ("qv-margin-01", "q_proj,v_proj", 0.1),
    ("qkvo-margin-01", "q_proj,k_proj,v_proj,o_proj", 0.1),
    ("qkvo-margin-02", "q_proj,k_proj,v_proj,o_proj", 0.2),
    ("qkvo-margin-03", "q_proj,k_proj,v_proj,o_proj", 0.3),
)
SELECTION_CODE_FILES = ("select.py", "evaluate.py", "clustering.py", "common.py", "uv.lock")


def choose_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No completed development candidates")
    eligible = [row for row in candidates if row["search_recall_at_5"] >= 0.90]
    def quality(row):
        return (row["all_event_pair_f1"], -row["all_event_pair_false_merges"], row["search_mrr"], -row["steps"], -row["order"])
    chosen = max(eligible, key=quality) if eligible else max(candidates, key=lambda row: (row["search_recall_at_5"], *quality(row)))
    return {**chosen, "eligible_search": bool(eligible), "selection_status": "eligible" if eligible else "search_target_missed"}


def run(args: argparse.Namespace) -> None:
    project = Path(__file__).resolve().parent
    dataset, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    manifest = verified_dataset_manifest(dataset)
    generation_path = generation_budget_path(dataset, manifest, args.generation_budget)
    training_files = code_fingerprint(project, TRAIN_CODE_FILES)
    selection_files = code_fingerprint(project, SELECTION_CODE_FILES)
    training_fingerprint = json_sha256(training_files)
    selection_fingerprint = json_sha256(selection_files)
    training_input_hash = json_sha256(training_examples(dataset))
    previous_seconds = float(manifest.get("mining", {}).get("elapsed_seconds", 0))
    mining_budget = manifest.get("mining", {}).get("budget_path")
    if mining_budget and Path(mining_budget).exists():
        previous_seconds = elapsed_budget_seconds(Path(mining_budget))
    if generation_path:
        previous_seconds += elapsed_budget_seconds(generation_path)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "experiment.json"
    previous_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    prior_run_seconds = previous_report.get("elapsed_seconds", 0)
    started = time.monotonic()
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "base_revision": args.base_revision,
              "dataset_manifest_sha256": sha256(dataset / "manifest.json"), "prior_generation_and_mining_seconds": previous_seconds,
              "generation_budget_path": str(generation_path) if generation_path else None,
              "training_code_files": training_files, "selection_code_files": selection_files,
              "selection_scope": "development_only", "dataset": str(dataset), "candidates": [],
              "evaluations": previous_report.get("evaluations", {})}
    def save():
        report["elapsed_seconds"] = prior_run_seconds + time.monotonic() - started
        report["total_compute_seconds"] = previous_seconds + report["elapsed_seconds"]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def check_sources():
        if verified_dataset_manifest(dataset)["files"] != manifest["files"]:
            raise ValueError("Dataset files changed during the experiment")
        if code_fingerprint(project, TRAIN_CODE_FILES) != training_files or code_fingerprint(project, SELECTION_CODE_FILES) != selection_files:
            raise ValueError("Training or selection code changed during the experiment")
    def execute(command: list[str], phase: str):
        check_sources()
        report["current_stage"] = phase
        save()
        remaining = args.max_hours * 3600 - previous_seconds - prior_run_seconds - (time.monotonic() - started)
        try:
            if remaining <= 0:
                raise TimeoutError("The combined experiment budget is exhausted")
            subprocess.run(command, check=True, cwd=project, timeout=remaining)
            check_sources()
        except BaseException as error:
            report["status"] = "time_budget_reached" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "interrupted_or_failed"
            raise
        finally:
            save()
    save()
    for order, (name, modules, margin) in enumerate(CANDIDATES):
        remaining = args.max_hours * 3600 - previous_seconds - prior_run_seconds - (time.monotonic() - started)
        if remaining <= 0:
            report["status"] = "time_budget_reached"
            save()
            raise TimeoutError("The combined generation, mining and training budget is exhausted")
        run_dir = output / name
        report["current_candidate"] = name
        training_path = run_dir / "training.json"
        training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else None
        if training and training.get("status") == "trained":
            if (training["base_revision"] != args.base_revision or training["dataset_manifest"]["files"] != manifest["files"]
                or training["settings"]["target_modules"] != modules.split(",") or training["settings"]["triplet_margin"] != margin
                or training["settings"]["epochs"] != args.epochs or training["settings"]["max_length"] != 512
                or training["settings"]["max_steps"] != -1 or training["settings"].get("limit") is not None
                or training.get("code_sha256") != sha256(project / "train.py")
                or training.get("training_code_fingerprint") != training_fingerprint
                or training.get("training_input_sha256") != training_input_hash):
                report["status"] = "provenance_mismatch"
                save()
                raise ValueError(f"Existing candidate has different provenance: {run_dir}")
        else:
            print(f"Training {name}", flush=True)
            training_command = [sys.executable, str(project / "train.py"), "--dataset", str(dataset), "--output", str(run_dir),
                                "--base-revision", args.base_revision, "--lora-targets", modules, "--triplet-margin", str(margin),
                                "--epochs", str(args.epochs), "--max-hours", str(remaining / 3600)]
            if generation_path:
                training_command.extend(("--generation-budget", str(generation_path)))
            execute(training_command, "training")
            training = json.loads(training_path.read_text(encoding="utf-8"))
            if training["status"] != "trained":
                report["status"] = "time_budget_reached"
                save()
                raise TimeoutError(f"Candidate did not finish: {name}")
            if training.get("training_code_fingerprint") != training_fingerprint or training.get("training_input_sha256") != training_input_hash:
                report["status"] = "provenance_mismatch"
                save()
                raise ValueError(f"New candidate used different training inputs or code: {run_dir}")
        if ("development_selection" not in training or "profile_path" not in training["development_selection"]["selected"]
            or training.get("development_selection_code_fingerprint") != selection_fingerprint):
            print(f"Selecting development checkpoint and profile for {name}", flush=True)
            remaining = args.max_hours * 3600 - previous_seconds - prior_run_seconds - (time.monotonic() - started)
            if remaining <= 0:
                report["status"] = "time_budget_reached"
                save()
                raise TimeoutError("No time remains for development selection")
            execute([sys.executable, str(project / "select.py"), "--run", str(run_dir), "--dataset", str(dataset), "--hybrid"], "development_selection")
            training = json.loads(training_path.read_text(encoding="utf-8"))
            training["development_selection_code_sha256"] = sha256(project / "select.py")
            training["development_selection_code_files"] = selection_files
            training["development_selection_code_fingerprint"] = selection_fingerprint
            training_path.write_text(json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        selected = training["development_selection"]["selected"]
        selected_step = int(selected["checkpoint"].split("-")[-1])
        summary = {key: value for key, value in selected.items() if key != "threshold_candidates"}
        report["candidates"].append({**summary, "name": name, "order": order, "run": str(run_dir), "adapter": str(run_dir / "adapter"), "steps": selected_step, "total_training_steps": training["steps"]})
        save()
    report["selected"] = choose_candidate(report["candidates"])
    report["status"] = "selected_pending_test"
    save()
    if getattr(args, "evaluate", False):
        selected = report["selected"]
        evaluation_dir = output / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        jobs = (
            ("base", ["--base-revision", args.base_revision]),
            ("base_hybrid", ["--base-revision", args.base_revision, "--hybrid"]),
            ("tuned", ["--adapter", selected["adapter"], "--cluster-profile", selected["profile_path"]]),
        )
        for name, flags in jobs:
            target = evaluation_dir / f"{name}.json"
            command = [sys.executable, str(project / "evaluate.py"), "--dataset", str(dataset), *flags, "--output", str(target)]
            fingerprint_data = {"command": command, "dataset_files": manifest["files"],
                                "code": {name: sha256(project / name) for name in ("evaluate.py", "common.py", "clustering.py")}}
            if name == "tuned":
                profile_file = Path(selected["profile_path"])
                profile_json = json.loads(profile_file.read_text(encoding="utf-8"))
                fingerprint_data["profile"] = sha256(profile_file)
                fingerprint_data["transform"] = sha256(profile_file.parent / profile_json["transform_file"])
                fingerprint_data["weights"] = sha256(Path(selected["adapter"]) / "adapter_model.safetensors")
            fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
            existing = report["evaluations"].get(name)
            if target.exists():
                if not existing or existing["fingerprint"] != fingerprint or existing["sha256"] != sha256(target):
                    raise ValueError(f"Frozen test result has different provenance: {target}")
            else:
                execute(command, f"test_evaluation:{name}")
                report["evaluations"][name] = {"path": str(target), "fingerprint": fingerprint, "sha256": sha256(target)}
                save()
            if getattr(args, "visualize", False):
                execute([sys.executable, str(project / "visualize.py"), "export", "--dataset", str(dataset),
                         "--evaluation", str(target), "--output", str(output / "visualization" / name)], f"visualization:{name}")
        tuned = json.loads((evaluation_dir / "tuned.json").read_text(encoding="utf-8"))
        report["acceptance"] = {
            "search_recall_at_5": tuned["test_dense_search"]["recall_at_5"], "search_goal": 0.90,
            "all_event_pair_f1": tuned["test_all_event_pairs"]["f1"], "clustering_goal": 0.80,
            "automatic_goals_met": tuned["test_dense_search"]["recall_at_5"] >= 0.90 and tuned["test_all_event_pairs"]["f1"] >= 0.80,
            "human_review_complete": manifest.get("status") == "human_reviewed",
        }
        report["status"] = "evaluated_provisional" if not report["acceptance"]["human_review_complete"] else "evaluated"
        save()
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-budget")
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base-revision", default=PINNED_BASE_REVISION)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only after the development choice is frozen")
    parser.add_argument("--visualize", action="store_true", help="Export caches for the completed evaluations (requires --evaluate)")
    args = parser.parse_args()
    if args.visualize and not args.evaluate:
        parser.error("--visualize requires --evaluate")
    run(args)


if __name__ == "__main__":
    main()
