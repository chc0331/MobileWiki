"""Upload reviewed synthetic data and the trained adapter to private Hub repos."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import sha256
from validate import validate_dataset


def publish(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, snapshot_download

    dataset = Path(args.dataset)
    adapter = Path(args.adapter)
    raw = Path(args.raw)
    base_eval = Path(args.base_eval)
    tuned_eval = Path(args.tuned_eval)
    validate_dataset(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if sha256(raw) != manifest["raw_sha256"]:
        raise ValueError("Raw teacher output does not match the dataset manifest")
    if manifest["status"] != "human_reviewed":
        raise ValueError("Human review must be completed before Hub upload")
    if not all(path.exists() for path in (adapter / "modules.json", raw, base_eval, tuned_eval)):
        raise FileNotFoundError("Adapter, raw events, or evaluation reports are missing")
    training = json.loads((adapter.parent / "training.json").read_text(encoding="utf-8"))
    base_metrics = json.loads(base_eval.read_text(encoding="utf-8"))
    tuned_metrics = json.loads(tuned_eval.read_text(encoding="utf-8"))
    if base_metrics["dataset_files"] != manifest["files"] or tuned_metrics["dataset_files"] != manifest["files"]:
        raise ValueError("Evaluation reports do not match the reviewed dataset")
    api = HfApi()
    user = api.whoami()["name"]
    dataset_repo = args.dataset_repo or f"{user}/slm-wiki-embeddinggemma2-text-synthetic-v1"
    model_repo = args.model_repo or f"{user}/slm-wiki-embeddinggemma2-ko-lora-v1"
    for repo_id, repo_type in ((dataset_repo, "dataset"), (model_repo, "model")):
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        if not api.repo_info(repo_id, repo_type=repo_type).private:
            raise ValueError(f"Refusing to upload to a public repository: {repo_id}")

    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    dataset_stage = stage / "dataset"
    model_stage = stage / "model"
    if dataset_stage.exists():
        shutil.rmtree(dataset_stage)
    if model_stage.exists():
        shutil.rmtree(model_stage)
    shutil.copytree(dataset, dataset_stage)
    shutil.copy2(raw, dataset_stage / "raw_events.jsonl")
    (dataset_stage / "README.md").write_text(
        f"""---
language: ko
pretty_name: SLM Wiki synthetic Korean text notes
---

# SLM Wiki 합성 텍스트 데이터

실제 사용자 기록을 쓰지 않았다. 로컬 `{manifest['teacher_id']}` 모델의
`{manifest['teacher_revision']}` digest로 만든 가상 생활 메모다.
학습·개발·평가 분할과 근거 문서 ID는 `manifest.json`에 기록했다.
고정 평가 자료의 질문 140개와 문서 쌍 100개는 원문 대조 검수 뒤 올렸다.
이 자료로 얻은 결과를 실제 개인정보나 모든 한국어 메모에 일반화하지 않는다.
""",
        encoding="utf-8",
    )
    api.upload_folder(
        repo_id=dataset_repo,
        repo_type="dataset",
        folder_path=dataset_stage,
        commit_message="Add reviewed synthetic Korean text embedding dataset",
    )
    dataset_revision = api.repo_info(dataset_repo, repo_type="dataset").sha
    downloaded_dataset = Path(snapshot_download(dataset_repo, repo_type="dataset", revision=dataset_revision))
    for name, expected in manifest["files"].items():
        if sha256(downloaded_dataset / name) != expected:
            raise ValueError(f"Hub dataset hash mismatch: {name}")

    shutil.copytree(adapter, model_stage)
    shutil.copy2(adapter.parent / "training.json", model_stage / "training.json")
    shutil.copy2(base_eval, model_stage / "base_evaluation.json")
    shutil.copy2(tuned_eval, model_stage / "tuned_evaluation.json")
    (model_stage / "README.md").write_text(
        f"""---
language: ko
base_model: google/embeddinggemma-2
datasets:
- {dataset_repo}
---

# SLM Wiki EmbeddingGemma 2 한국어 텍스트 어댑터

기반 모델: `google/embeddinggemma-2` @ `{training['base_revision']}`.
학습 데이터: `{dataset_repo}` @ `{dataset_revision}`.
이 저장소에는 텍스트 인코더의 LoRA 어댑터와 평가 기록을 둔다.
Android 변환·속도·메모리 기준은 아직 시험하지 않았다.

평가 상태: `{tuned_metrics['status']}`.
검색 Recall@5: {tuned_metrics['test_dense_search']['recall_at_5']:.3f}.
묶음 문서 쌍 F1: {tuned_metrics['test_pair']['f1']:.3f}.
""",
        encoding="utf-8",
    )
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=model_stage,
        commit_message="Add EmbeddingGemma 2 Korean text LoRA adapter and evaluation",
    )
    model_revision = api.repo_info(model_repo, repo_type="model").sha
    downloaded_model = Path(snapshot_download(model_repo, revision=model_revision))
    if not (downloaded_model / "modules.json").exists():
        raise ValueError("Hub model reload check failed")
    from evaluate import evaluate

    hub_evaluation_path = stage / "hub_evaluation.json"
    evaluate(
        argparse.Namespace(
            dataset=str(downloaded_dataset),
            adapter=str(downloaded_model),
            base_revision=None,
            output=str(hub_evaluation_path),
            batch_size=2,
        )
    )
    hub_metrics = json.loads(hub_evaluation_path.read_text(encoding="utf-8"))
    for key in ("test_dense_search", "test_pair"):
        metric = "recall_at_5" if key == "test_dense_search" else "f1"
        if abs(hub_metrics[key][metric] - tuned_metrics[key][metric]) > 1e-6:
            raise ValueError(f"Hub reload changed {key}.{metric}")
    if abs(hub_metrics["test_dense_search"]["mrr"] - tuned_metrics["test_dense_search"]["mrr"]) > 1e-6:
        raise ValueError("Hub reload changed search MRR")
    result = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "dataset_url": f"https://huggingface.co/datasets/{dataset_repo}/tree/{dataset_revision}",
        "model_url": f"https://huggingface.co/{model_repo}/tree/{model_revision}",
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--raw", default="artifacts/raw_events.jsonl")
    parser.add_argument("--adapter", default="artifacts/trained/adapter")
    parser.add_argument("--base-eval", default="artifacts/evaluation/base.json")
    parser.add_argument("--tuned-eval", default="artifacts/evaluation/tuned.json")
    parser.add_argument("--dataset-repo")
    parser.add_argument("--model-repo")
    parser.add_argument("--stage", default="artifacts/publish_stage")
    parser.add_argument("--output", default="artifacts/hub_publish.json")
    publish(parser.parse_args())


if __name__ == "__main__":
    main()
