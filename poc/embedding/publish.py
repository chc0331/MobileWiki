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
    curated = Path(args.curated)
    teacher_raw = Path(args.teacher_raw)
    base_eval = Path(args.base_eval)
    tuned_eval = Path(args.tuned_eval)
    validate_dataset(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if sha256(curated) != manifest["prepared_source_sha256"]:
        raise ValueError("Curated source does not match the dataset manifest")
    if not manifest["curation"] or sha256(teacher_raw) != manifest["curation"]["teacher_raw_sha256"]:
        raise ValueError("Teacher output does not match the curation manifest")
    if manifest["status"] != "human_reviewed":
        raise ValueError("Human review must be completed before Hub upload")
    if not all(path.exists() for path in (adapter / "modules.json", curated, teacher_raw, base_eval, tuned_eval)):
        raise FileNotFoundError("Adapter, source events, or evaluation reports are missing")
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
    shutil.copy2(curated, dataset_stage / "curated_events.jsonl")
    shutil.copy2(teacher_raw, dataset_stage / "teacher_raw_events.jsonl")
    (dataset_stage / "README.md").write_text(
        f"""---
language: ko
pretty_name: SLM Wiki synthetic Korean text notes
---

# SLM Wiki 합성 텍스트 데이터

실제 사용자 기록을 쓰지 않았다. 로컬 `{manifest['teacher_id']}` 모델의
`{manifest['teacher_revision']}` digest로 만든 가상 생활 메모다.
`documents.jsonl`에는 원문 2,100건, `queries.jsonl`에는 검색 질문과 근거,
`triplets.jsonl`에는 학습용 세 쌍, `pairs.jsonl`에는 문서 묶기 평가 쌍이 있다.
학습 800건, 개발 300건, 고정 평가 1,000건을 가족·사건 단위로 분리했다.
근거 문서 ID, 교사 모델·프롬프트 버전, 생성 코드와 데이터 파일의 해시는 `manifest.json`에 남겼다.
고정 평가 질문 140개와 문서 쌍 100개의 정답은 원문 대조 검수했다.
생성 단계의 1차 보정본은 `teacher_raw_events.jsonl`, 답 단서 제거본은 `curated_events.jsonl`이다.
자동 수정 내역은 두 파일의 `repairs`와 `curation_repairs`에 남겼다.
교사 모델의 보정 전 응답 전문은 저장하지 않았다.
나머지 모든 문장을 사람이 검토한 것은 아니다.
10가지 주제와 일부 정답 표현은 분할 사이에 반복된다. 새로운 주제의 성능은 이 자료로 알 수 없다.
실제 개인정보나 모든 한국어 메모에 성능을 일반화하지 않는다.
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
    validate_dataset(downloaded_dataset)
    if sha256(downloaded_dataset / "curated_events.jsonl") != manifest["prepared_source_sha256"]:
        raise ValueError("Hub curated source hash mismatch")
    if sha256(downloaded_dataset / "teacher_raw_events.jsonl") != manifest["curation"]["teacher_raw_sha256"]:
        raise ValueError("Hub teacher output hash mismatch")

    shutil.copytree(adapter, model_stage)
    shutil.copy2(adapter.parent / "training.json", model_stage / "training.json")
    shutil.copy2(base_eval, model_stage / "base_evaluation.json")
    shutil.copy2(tuned_eval, model_stage / "tuned_evaluation.json")
    tuned_labels = tuned_eval.with_suffix(".labels.jsonl")
    tag_review = tuned_eval.with_suffix(".tag_review.csv")
    if not tuned_labels.exists() or not tag_review.exists():
        raise FileNotFoundError("Tuned tag labels or review sheet are missing")
    shutil.copy2(tuned_labels, model_stage / "tag_labels.jsonl")
    shutil.copy2(tag_review, model_stage / "tag_review.csv")
    (model_stage / "README.md").write_text(
        f"""---
language: ko
base_model: google/embeddinggemma-2
datasets:
- {dataset_repo}
license: apache-2.0
---

# SLM Wiki EmbeddingGemma 2 한국어 텍스트 어댑터

기반 모델: `google/embeddinggemma-2` @ `{training['base_revision']}`.
학습 데이터: `{dataset_repo}` @ `{dataset_revision}`.
이 저장소에는 텍스트 인코더의 LoRA 어댑터와 평가 기록을 둔다.
실행할 때는 위 revision의 텍스트 전용 기반 모델을 먼저 열고 이 어댑터를 붙인다.
기반 모델까지 합친 독립 실행 파일은 아니다.
Android 변환·속도·메모리 기준은 아직 시험하지 않았다.

평가 상태: `{tuned_metrics['status']}`.
검색 Recall@5: {tuned_metrics['test_dense_search']['recall_at_5']:.3f}.
검수 표본 문서 쌍 100개 F1: {tuned_metrics['test_pair']['f1']:.3f}.
전체 합성 사건의 모든 문서 쌍 F1: {tuned_metrics['test_all_event_pairs']['f1']:.3f}.
전체 쌍 점수는 사건 카드의 자동 정답에 따른 진단이며 모든 쌍을 사람이 확인하지 않았다.
묶음 태그와 근거는 `tag_labels.jsonl`에 기록했다. 태그 적합성은
`tag_review.csv`에서 별도로 확인해야 한다.
""",
        encoding="utf-8",
    )
    model_files = {
        path.relative_to(model_stage).as_posix(): sha256(path)
        for path in model_stage.rglob("*") if path.is_file()
    }
    (model_stage / "model_files.json").write_text(
        json.dumps(model_files, ensure_ascii=False, indent=2) + "\n",
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
    for name, expected in model_files.items():
        if sha256(downloaded_model / name) != expected:
            raise ValueError(f"Hub model hash mismatch: {name}")
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
    for key in ("test_dense_search", "test_pair", "test_all_event_pairs"):
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
    parser.add_argument("--curated", default="artifacts/curated_events.jsonl")
    parser.add_argument("--teacher-raw", default="artifacts/raw_events.jsonl")
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
