"""Export immutable clustering bundles and inspect them without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import STUDENT_MODEL, clustering_text, read_jsonl, sha256, write_jsonl

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "clustering-visualization"
BASE_DIR = Path(__file__).resolve().parent


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def document_key(dataset_hash: str, doc_id: str) -> str:
    return f"{dataset_hash}:{doc_id}"


def evaluation_run_id(evaluation_path: Path) -> str:
    path = Path(evaluation_path)
    try:
        path = path.relative_to(BASE_DIR / "artifacts")
    except ValueError:
        path = Path(*path.parts[-3:])
    return path.with_suffix("").as_posix()


def bundle_display_name(manifest: dict) -> str:
    source = evaluation_run_id(Path(manifest["evaluation_path"])) if manifest.get("evaluation_path") else manifest["run_id"]
    name = manifest["run_id"] if manifest.get("run_id_explicit") else source
    if manifest.get("run_id_explicit") and name != source:
        name = f"{name} · {source}"
    return f"{name} · {manifest['dataset_revision']}"


def profile_fit_scope(profile: dict, train_vectors: np.ndarray, dataset_documents_sha256: str) -> tuple[str, str]:
    """Validate local fits while allowing a frozen foreign fit for regression data."""
    from clustering import vector_hash

    provenance = profile.get("provenance", {})
    source_hash = (
        provenance.get("fit_dataset_documents_sha256")
        or provenance.get("dataset_files", {}).get("documents.jsonl")
        or provenance.get("dataset_documents_sha256")
    )
    if source_hash and source_hash != dataset_documents_sha256:
        return "foreign_training_dataset", source_hash
    if profile["fitted"].get("training_vector_sha256") != vector_hash(train_vectors):
        raise ValueError("Cluster postprocessor was fitted on different training vectors without matching foreign-fit provenance")
    return "current_training_dataset", dataset_documents_sha256


def assignments_from_labels(documents: list[dict], labels: list[dict]) -> dict[str, int]:
    expected = {document["doc_id"] for document in documents}
    if len(expected) != len(documents):
        raise ValueError("Duplicate document IDs in visualization input")
    groups = {}
    for label in labels:
        for doc_id in label["doc_ids"]:
            if doc_id in groups:
                raise ValueError(f"Document occurs in multiple groups: {doc_id}")
            groups[doc_id] = int(label["group_id"])
    if set(groups) != expected:
        raise ValueError("Clustering assignments do not match the split document IDs")
    return groups


def same_dataset(left: dict, right: dict) -> bool:
    return left["dataset_documents_sha256"] == right["dataset_documents_sha256"]


def selected_doc_ids(selection, allowed_ids: set[str]) -> list[str]:
    """Use customdata IDs, never trace-local Plotly point indices."""
    if not selection:
        return []
    state = selection.get("selection", selection)
    selected = []
    for point in state.get("points", []):
        value = point.get("customdata")
        doc_id = value[0] if isinstance(value, (list, tuple)) and value else value
        if isinstance(doc_id, str) and doc_id in allowed_ids and doc_id not in selected:
            selected.append(doc_id)
    return selected


def _same_metadata(left: dict, right: dict, key: str):
    if not left.get(key) or not right.get(key):
        return None
    return left[key] == right[key]


def is_answer_document(document: dict):
    if "answer_doc_index" not in document:
        return None
    return document["doc_id"].endswith(f"-d{document['answer_doc_index']}")


def error_pairs(documents: list[dict], groups: dict[str, int]) -> list[dict]:
    """Enumerate only false merges/splits, rather than every unrelated pair."""
    by_group: dict[int, list[dict]] = defaultdict(list)
    by_event: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_group[groups[document["doc_id"]]].append(document)
        if document.get("event_id"):
            by_event[document["event_id"]].append(document)
    output = []
    for error, buckets in (("오결합", by_group.values()), ("과분할", by_event.values())):
        for members in buckets:
            for left, right in combinations(members, 2):
                if not left.get("event_id") or not right.get("event_id"):
                    continue
                same_event = left["event_id"] == right["event_id"]
                same_group = groups[left["doc_id"]] == groups[right["doc_id"]]
                if (error == "오결합" and same_event) or (error == "과분할" and same_group):
                    continue
                left_answer, right_answer = is_answer_document(left), is_answer_document(right)
                output.append({
                    "error": error,
                    "left_id": left["doc_id"],
                    "right_id": right["doc_id"],
                    "left_group": groups[left["doc_id"]],
                    "right_group": groups[right["doc_id"]],
                    "same_category": _same_metadata(left, right, "category"),
                    "same_date": _same_metadata(left, right, "date"),
                    "same_family": _same_metadata(left, right, "family_id"),
                    "same_fact": _same_metadata(left, right, "fact_value"),
                    "both_answerless": None if left_answer is None or right_answer is None else not left_answer and not right_answer,
                })
    return sorted(output, key=lambda row: (row["error"], row["left_id"], row["right_id"]))


def group_summaries(documents: list[dict], groups: dict[str, int], labels: list[dict]) -> list[dict]:
    by_group: dict[int, list[dict]] = defaultdict(list)
    label_by_id = {int(label["group_id"]): label for label in labels}
    for document in documents:
        by_group[groups[document["doc_id"]]].append(document)
    output = []
    for group_id, members in by_group.items():
        events = Counter(document.get("event_id") for document in members if document.get("event_id"))
        total = len(members) * (len(members) - 1) // 2
        correct = sum(count * (count - 1) // 2 for count in events.values())
        label = label_by_id.get(group_id, {})
        output.append({
            "group_id": group_id,
            "label": label.get("label", "이름 미정"),
            "documents": len(members),
            "events": len(events),
            "false_merge_pairs": total - correct,
            "label_stage": label.get("label_stage", "기존 규칙"),
            "doc_ids": sorted(document["doc_id"] for document in members),
        })
    return sorted(output, key=lambda row: (-row["false_merge_pairs"], -row["documents"], row["group_id"]))


def projection(vectors: np.ndarray, method: str = "pca", train_vectors: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if method == "tsne" and len(vectors) >= 3:
        dimensions = min(50, vectors.shape[1], len(vectors) - 1)
        reduced = PCA(n_components=dimensions, random_state=42).fit_transform(vectors)
        return TSNE(
            n_components=2, perplexity=min(30.0, max(1.0, (len(vectors) - 1) / 3)),
            init="pca", learning_rate="auto", random_state=42, max_iter=1000,
        ).fit_transform(reduced).astype(np.float32)
    if method not in {"pca", "tsne"}:
        raise ValueError(f"Unknown projection: {method}")
    fit_vectors = vectors if train_vectors is None or len(train_vectors) < 2 else np.asarray(train_vectors)
    count = min(2, fit_vectors.shape[0], fit_vectors.shape[1])
    projected = PCA(n_components=count, random_state=42).fit(fit_vectors).transform(vectors)
    return np.pad(projected, ((0, 0), (0, 2 - count))).astype(np.float32)


def _resolve_path(value: str, evaluation_path: Path) -> Path:
    path = Path(value).expanduser()
    candidates = (path, BASE_DIR / path, evaluation_path.parent / path, evaluation_path.parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Artifact path does not exist: {value}")


def discover_bundles(root: Path) -> list[Path]:
    output = []
    if root.exists():
        for path in root.rglob("manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if manifest.get("artifact_type") == ARTIFACT_TYPE:
                output.append(path.parent)
    return sorted(output)


def load_bundle(folder: Path) -> dict:
    from clustering import load_profile

    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != ARTIFACT_TYPE or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported visualization bundle schema")
    required = {"run_id", "dataset_revision", "dataset_documents_sha256", "base_revision", "status", "splits", "files"}
    if required - manifest.keys():
        raise ValueError(f"Incomplete visualization manifest: {sorted(required - manifest.keys())}")
    for name, expected in manifest["files"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Visualization artifact hash mismatch: {name}")
    bundle = {
        "root": folder,
        "manifest": manifest,
        "evaluation": json.loads((folder / "evaluation.json").read_text(encoding="utf-8")),
        "profile": load_profile(folder / "cluster_profile.json"),
        "train_vectors": np.load(folder / "train.npz", allow_pickle=False)["raw"],
        "splits": {},
    }
    for split, info in manifest["splits"].items():
        documents = read_jsonl(folder / info["documents"])
        ids = [document["doc_id"] for document in documents]
        if ids != info["doc_ids"] or content_hash(ids) != info["row_order_sha256"]:
            raise ValueError(f"Visualization vector row order mismatch: {split}")
        labels = read_jsonl(folder / info["labels"])
        arrays = dict(np.load(folder / info["vectors"], allow_pickle=False))
        if not {"raw", "corrected", "pca", "tsne"} <= arrays.keys():
            raise ValueError(f"Missing visualization vector arrays: {split}")
        if any(values.ndim != 2 or len(values) != len(documents) or not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError(f"Visualization vector row count mismatch: {split}")
        if arrays["raw"].shape != arrays["corrected"].shape or any(arrays[name].shape[1] != 2 for name in ("pca", "tsne")):
            raise ValueError(f"Invalid visualization vector dimensions: {split}")
        bundle["splits"][split] = {
            "documents": documents, "labels": labels,
            "groups": assignments_from_labels(documents, labels), **arrays,
        }
    return bundle


def recluster_development(bundle: dict, split: str, config, labeler=None) -> dict:
    """Return a session-only dev view; frozen input bundles are never modified."""
    from dataclasses import asdict
    from clustering import cluster_groups, fit_postprocessor, transform_vectors

    if split != "dev":
        raise ValueError("Only the development split permits clustering experiments")
    if labeler is None:
        from evaluate import extract_labels

        labeler = extract_labels
    original = bundle["splits"][split]
    fitted = fit_postprocessor(bundle["train_vectors"], config.postprocess, config.remove_components)
    corrected = transform_vectors(original["raw"], fitted)
    groups = cluster_groups(original["documents"], original["raw"], config, fitted)
    return {
        **original, "groups": groups, "labels": labeler(original["documents"], groups),
        "corrected": corrected,
        "pca": projection(corrected, train_vectors=transform_vectors(bundle["train_vectors"], fitted)),
        "tsne": projection(corrected, "tsne"), "exploratory": True,
        "profile": {"config": asdict(config), "fitted": fitted},
    }


def export_bundle(args: argparse.Namespace) -> None:
    """Load the model once here; the dashboard only imports bundle helpers."""
    import torch
    from sentence_transformers import SentenceTransformer
    from clustering import ClusterConfig, cluster_groups, fit_postprocessor, load_profile, save_profile, transform_vectors
    from evaluate import document_vectors, encode_documents, extract_labels, model_fingerprint

    dataset = Path(args.dataset).resolve()
    evaluation_path = Path(args.evaluation).resolve()
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    document_hash = sha256(dataset / "documents.jsonl")
    if report.get("dataset_files", {}).get("documents.jsonl") != document_hash:
        raise ValueError("Evaluation and visualization dataset hashes differ")
    output = Path(args.output).resolve()
    if (output / "manifest.json").exists() and not args.force:
        existing = load_bundle(output)
        if (
            existing["manifest"]["dataset_documents_sha256"] == document_hash
            and existing["manifest"]["evaluation_sha256"] == sha256(evaluation_path)
        ):
            print(f"Verified existing visualization bundle: {output}")
            return
        raise ValueError("Output contains another run; select a new output directory or pass --force")
    documents = read_jsonl(dataset / "documents.jsonl")
    by_split = {split: [doc for doc in documents if doc["split"] == split] for split in ("train", "dev", "test")}
    if any(not members for members in by_split.values()):
        raise ValueError("Visualization export requires train/dev/test documents")
    adapter = _resolve_path(report["model"], evaluation_path) if report["model"] != STUDENT_MODEL else None
    deidentify = report.get("clustering_deidentify", False)
    model_key = model_fingerprint(report["base_revision"], adapter)
    if report.get("model_key") and report["model_key"] != model_key:
        raise ValueError("Model weights differ from the frozen evaluation fingerprint")
    profile_value = args.cluster_profile or report.get("cluster_profile_path") or report.get("cluster_profile")
    if isinstance(profile_value, dict):
        profile_value = profile_value.get("path")
    profile = load_profile(_resolve_path(str(profile_value), evaluation_path)) if profile_value else None
    if profile:
        saved = profile.get("provenance", {})
        if saved.get("model_key") and saved["model_key"] != model_key:
            raise ValueError("Cluster profile was fitted for another model or adapter")
        if saved.get("clustering_deidentify") is not None and saved["clustering_deidentify"] != deidentify:
            raise ValueError("Input de-identification differs from frozen cluster profile")
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(
        STUDENT_MODEL, revision=report["base_revision"], device=device,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32}, default_prompt_name=None,
    )
    adapter_hash = None
    if adapter:
        model.load_adapter(str(adapter))
        adapter_hash = content_hash({path.name: sha256(path) for path in sorted(adapter.glob("*.safetensors"))})
    model.max_seq_length = 512
    cache_dir = dataset.parent / "embedding_cache"
    raw = {}
    for split, members in by_split.items():
        print(f"Encoding visualization split: {split} ({len(members)} documents)", flush=True)
        chunks, chunk_ids = encode_documents(
            model, members, "clustering", args.batch_size, cache_dir, model_key, deidentify
        )
        raw[split] = document_vectors(chunks, chunk_ids, members)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    if profile is None:
        config = ClusterConfig(threshold=report["dev_selected_threshold"], linkage="average", postprocess="none")
        profile = {"config": vars(config), "fitted": fit_postprocessor(raw["train"]), "selection": {"source": "frozen_legacy_evaluation"}}
    config = ClusterConfig.from_dict(profile["config"])
    fitted = profile["fitted"]
    fit_scope, fit_hash = profile_fit_scope(profile, raw["train"], document_hash)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        **profile.get("provenance", {}),
        "fit_dataset_documents_sha256": fit_hash,
        "visualization_dataset_documents_sha256": document_hash, "evaluation_sha256": sha256(evaluation_path),
        "base_revision": report["base_revision"], "adapter_sha256": adapter_hash,
        "model_key": model_key, "clustering_deidentify": deidentify,
    }
    save_profile(output / "cluster_profile.json", profile, provenance=provenance)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "train.npz", raw=raw["train"])
    split_info = {}
    fitted_projection = transform_vectors(raw["train"], fitted)
    for split in ("dev", "test"):
        members = by_split[split]
        groups = cluster_groups(members, raw[split], config, fitted)
        frozen_labels_path = evaluation_path.with_suffix(".labels.jsonl")
        if split == "test" and frozen_labels_path.exists():
            labels = read_jsonl(frozen_labels_path)
            frozen_groups = assignments_from_labels(members, labels)
            partition = lambda mapping: sorted(sorted(doc_id for doc_id, group_id in mapping.items() if group_id == group) for group in set(mapping.values()))
            if partition(groups) != partition(frozen_groups):
                raise ValueError("Frozen evaluation assignments differ from the supplied model/profile")
            groups = frozen_groups
        else:
            labels = extract_labels(members, groups)
        corrected = transform_vectors(raw[split], fitted)
        ids = [doc["doc_id"] for doc in members]
        write_jsonl(output / f"{split}.documents.jsonl", members)
        write_jsonl(output / f"{split}.labels.jsonl", labels)
        np.savez_compressed(
            output / f"{split}.npz", raw=raw[split], corrected=corrected,
            pca=projection(corrected, train_vectors=fitted_projection),
            tsne=projection(corrected, "tsne"),
        )
        split_info[split] = {
            "documents": f"{split}.documents.jsonl", "labels": f"{split}.labels.jsonl",
            "vectors": f"{split}.npz", "doc_ids": ids, "row_order_sha256": content_hash(ids),
        }
    files = {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id or evaluation_run_id(evaluation_path), "run_id_explicit": bool(args.run_id),
        "dataset_revision": dataset_manifest.get("dataset_revision", "legacy-v1"),
        "dataset_documents_sha256": document_hash,
        "dataset_dir": str(dataset), "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256(evaluation_path),
        "base_revision": report["base_revision"], "adapter_sha256": adapter_hash,
        "clustering_input_sha256": content_hash([clustering_text(doc, deidentify=deidentify) for doc in documents]),
        "clustering_deidentify": deidentify,
        "fit_scope": fit_scope, "fit_dataset_documents_sha256": fit_hash,
        "status": report.get("status", "unknown"), "projection_seed": 42,
        "projection_note": "PCA is fitted on training vectors; t-SNE is a display-only split projection.",
        "splits": split_info, "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_bundle(output)
    print(f"Saved visualization bundle: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Encode and export one frozen evaluation for the dashboard")
    export.add_argument("--dataset", required=True)
    export.add_argument("--evaluation", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--cluster-profile")
    export.add_argument("--run-id")
    export.add_argument("--batch-size", type=int, default=2)
    export.add_argument("--device", choices=("cpu", "mps"))
    export.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_bundle(args)


if __name__ == "__main__":
    main()
