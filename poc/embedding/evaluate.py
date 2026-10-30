"""Measure retrieval, document grouping, and extractive group labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from clustering import (
    ClusterConfig,
    all_event_pair_f1,
    cluster_groups,
    error_diagnostics,
    geometry_diagnostics,
    load_profile,
    save_profile,
    select_profile,
    transform_vectors,
)

from common import (
    STUDENT_MODEL,
    clustering_text,
    dataset_clustering_deidentify,
    matching_doc_text,
    read_jsonl,
    search_query_text,
    verified_dataset_manifest,
    write_jsonl,
)

THRESHOLDS = tuple(round(value * 0.05, 2) for value in range(1, 17))
BORING_LABELS = {
    "메모", "기록", "확인", "예정", "일정", "사항", "관련", "내용", "이름", "날짜", "시간",
    "필요", "모두", "사실", "경우", "정보", "완료", "추가", "정리", "가족", "물건", "동료",
    "구성원", "연락", "날씨", "준비",
}


def pair_f1(pairs: list[dict], groups: dict[str, int]) -> dict:
    tp = fp = fn = tn = 0
    for pair in pairs:
        prediction = groups[pair["left_id"]] == groups[pair["right_id"]]
        if pair["related"] and prediction:
            tp += 1
        elif pair["related"]:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def groups_for(documents: list[dict], vectors: np.ndarray, threshold: float) -> dict[str, int]:
    return cluster_groups(documents, vectors, ClusterConfig(threshold=threshold))


def select_cluster_threshold(
    documents: list[dict],
    vectors: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> tuple[dict, list[dict]]:
    candidates = [
        {"threshold": threshold, **all_event_pair_f1(documents, groups_for(documents, vectors, threshold))}
        for threshold in thresholds
    ]
    selected = max(candidates, key=lambda row: (row["f1"], -row["threshold"]))
    return selected, candidates


def chunk_document(document: dict, tokenizer, max_tokens: int = 420, overlap: int = 64) -> list[dict]:
    tokens = tokenizer.encode(document["text"], add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [document]
    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        piece = tokenizer.decode(tokens[start : start + max_tokens], skip_special_tokens=True)
        chunks.append({**document, "text": piece})
        if start + max_tokens >= len(tokens):
            break
    return chunks


def model_fingerprint(base_revision: str, adapter: str | Path | None = None) -> str:
    digest = hashlib.sha256(base_revision.encode())
    if adapter:
        path = Path(adapter)
        for file in sorted(path.glob("adapter*.json")) + sorted(path.glob("adapter*.safetensors")):
            digest.update(file.name.encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def encode_texts(model, texts: list[str], batch_size: int, cache_dir: Path | None = None, model_key: str | None = None, task: str = "text") -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    cache = None
    if cache_dir is not None and model_key:
        payload = json.dumps({"model": model_key, "task": task, "max_length": model.max_seq_length, "texts": texts}, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{key}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as value:
                vectors = value["vectors"]
            if len(vectors) != len(texts) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid embedding cache: {cache}")
            return vectors
    vectors = np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    if cache is not None:
        temporary = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, vectors=vectors)
        temporary.replace(cache)
    return vectors


def encode_documents(model, documents: list[dict], task: str, batch_size: int, cache_dir: Path | None = None, model_key: str | None = None, deidentify: bool | None = None) -> tuple[np.ndarray, list[str]]:
    chunks = [chunk for document in documents for chunk in chunk_document(document, model.tokenizer)]
    texts = [
        matching_doc_text(chunk) if task == "search" else (
            clustering_text(chunk) if deidentify is None else clustering_text(chunk, deidentify=deidentify)
        )
        for chunk in chunks
    ]
    return encode_texts(model, texts, batch_size, cache_dir, model_key, task), [chunk["doc_id"] for chunk in chunks]


def document_vectors(chunk_vectors: np.ndarray, chunk_ids: list[str], documents: list[dict]) -> np.ndarray:
    if not documents:
        return np.empty((0, chunk_vectors.shape[1]), dtype=np.float32)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for vector, doc_id in zip(chunk_vectors, chunk_ids):
        grouped[doc_id].append(vector)
    vectors = np.stack([np.mean(grouped[doc["doc_id"]], axis=0) for doc in documents])
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def retrieval_score(queries: list[dict], ranked_ids: list[list[str]]) -> dict:
    ranks = []
    for query, ids in zip(queries, ranked_ids):
        relevant = set(query["relevant_doc_ids"])
        rank = next((index for index, doc_id in enumerate(ids, start=1) if doc_id in relevant), None)
        ranks.append(rank)
    hits_at_1 = sum(rank == 1 for rank in ranks)
    hits_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    total = len(ranks)
    return {
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "total": total,
        "recall_at_1": hits_at_1 / total if total else 0.0,
        "recall_at_5": hits_at_5 / total if total else 0.0,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total if total else 0.0,
    }


def dense_search(model, documents: list[dict], queries: list[dict], batch_size: int, cache_dir: Path | None = None, model_key: str | None = None) -> dict:
    if not documents or not queries:
        return retrieval_score(queries, [[] for _ in queries])
    vectors, doc_ids = encode_documents(model, documents, "search", batch_size, cache_dir, model_key)
    query_vectors = encode_texts(model, [search_query_text(query["text"]) for query in queries], batch_size, cache_dir, model_key, "search-query")
    ranked = []
    for query_vector in query_vectors:
        scores: dict[str, float] = {}
        for doc_id, score in zip(doc_ids, vectors @ query_vector):
            scores[doc_id] = max(float(score), scores.get(doc_id, float("-inf")))
        ranked.append(sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id)))
    return retrieval_score(queries, ranked)


def lexical_search(documents: list[dict], queries: list[dict]) -> dict:
    if not queries:
        return retrieval_score([], [])
    texts = [f"{item['title']} {item['text']}" for item in documents]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform([item["text"] for item in queries])
    ranked = []
    for query_vector in query_matrix:
        scores = (matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores)
        ranked.append([documents[index]["doc_id"] for index in order])
    return retrieval_score(queries, ranked)


def shortcut_baselines(documents: list[dict], queries: list[dict]) -> dict:
    """Report name-only and fixed-first-slot retrieval on the applicable query slices."""
    names = {doc["person"] for doc in documents if doc.get("person")}
    all_ids = sorted(doc["doc_id"] for doc in documents)
    first_slot_ids = sorted(doc["doc_id"] for doc in documents if doc["doc_id"].endswith("-d0"))
    first_slot = first_slot_ids + [doc_id for doc_id in all_ids if doc_id not in set(first_slot_ids)]
    named_queries = [query for query in queries if query.get("identity_style") == "named"]
    person_only_rankings = []
    person_first_rankings = []
    for query in named_queries:
        matched = [person for person in names if person in query["text"]]
        if len(matched) != 1:
            raise ValueError(f"Expected one synthetic person in named query {query['query_id']}")
        person = matched[0]
        matching = sorted(doc["doc_id"] for doc in documents if doc.get("person") == person)
        first_notes = [doc_id for doc_id in matching if doc_id.endswith("-d0")]
        person_only_rankings.append(matching + [doc_id for doc_id in all_ids if doc_id not in set(matching)])
        person_first_rankings.append(first_notes + [doc_id for doc_id in all_ids if doc_id not in set(first_notes)])
    return {
        "name_only_named_queries": retrieval_score(named_queries, person_only_rankings),
        "name_and_d0_named_queries": retrieval_score(named_queries, person_first_rankings),
        "d0_all_queries": retrieval_score(queries, [first_slot for _ in queries]),
    }


def extract_labels(documents: list[dict], groups: dict[str, int], tokenizer=None) -> list[dict]:
    if tokenizer is None:
        from kiwipiepy import Kiwi

        tokenizer = Kiwi().tokenize
    all_tokens: dict[str, Counter] = {}
    for document in documents:
        content = document["title"] + " " + document["text"]
        person = document.get("person", "")
        if person:
            content = content.replace(person, "")
        aliases = {person, person[1:]} if len(person) >= 3 else {person}
        tokens = Counter(
            token.form
            for token in tokenizer(content)
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2
            and token.form not in BORING_LABELS and token.form not in aliases
            and not re.search(r"\d", token.form)
        )
        all_tokens[document["doc_id"]] = tokens
    members: dict[int, list[dict]] = defaultdict(list)
    for document in documents:
        members[groups[document["doc_id"]]].append(document)
    class_counts = {
        group_id: sum((all_tokens[document["doc_id"]] for document in items), Counter())
        for group_id, items in members.items()
    }
    all_counts = sum(class_counts.values(), Counter())
    average_length = sum(sum(counts.values()) for counts in class_counts.values()) / max(1, len(members))
    output = []
    for group_id, grouped_docs in sorted(members.items()):
        counts = class_counts[group_id]
        support = Counter(token for doc in grouped_docs for token in all_tokens[doc["doc_id"]])
        candidates = [token for token, count in support.items() if count > len(grouped_docs) / 2]
        stage = "singleton" if len(grouped_docs) == 1 else "majority"
        representative = min(grouped_docs, key=lambda doc: (-sum(all_tokens[doc["doc_id"]].values()), doc["doc_id"]))
        if not candidates:
            candidates = [token for token, count in support.items() if count >= 2]
            stage = "two_documents"
        if not candidates:
            candidates = list(all_tokens[representative["doc_id"]])
            stage = "representative_only"
        scores = {
            token: counts[token] / max(1, sum(counts.values())) * math.log1p(average_length / all_counts[token])
            for token in candidates
        }
        candidates = [token for token in candidates if any(token in doc["title"] + doc["text"] for doc in grouped_docs)]
        candidates.sort(key=lambda token: (-scores[token], -support[token], token))
        label = candidates[0] if candidates else "이름 미정"
        sources = [doc["doc_id"] for doc in grouped_docs if label in all_tokens[doc["doc_id"]] and label in doc["title"] + doc["text"]]
        output.append({
            "group_id": group_id, "label": label,
            "doc_ids": [doc["doc_id"] for doc in grouped_docs], "source_doc_ids": sources,
            "support_count": len(sources), "label_stage": stage if candidates else "none",
            "ctfidf_score": scores[label] if candidates else 0.0,
            "representative_doc_id": representative["doc_id"],
        })
    return output


def write_tag_review(path: Path, labels: list[dict], documents: list[dict]) -> None:
    docs = {document["doc_id"]: document for document in documents}
    ordered = sorted(labels, key=lambda item: (-len(item["doc_ids"]), item["group_id"]))
    count = min(30, len(ordered))
    selected = [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)] if count > 1 else ordered
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_id", "tag", "document_count", "source_doc_ids", "example_documents", "accepted", "comment"])
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "group_id": item["group_id"],
                "tag": item["label"],
                "document_count": len(item["doc_ids"]),
                "source_doc_ids": " ".join(item["source_doc_ids"]),
                "example_documents": "\n".join(f"{doc_id}: {docs[doc_id]['text']}" for doc_id in item["doc_ids"][:4]),
            })


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    dataset = Path(args.dataset)
    manifest = verified_dataset_manifest(dataset)
    deidentify_arg = getattr(args, "deidentify", "auto")
    deidentify = deidentify_arg == "on" if deidentify_arg != "auto" else dataset_clustering_deidentify(manifest)
    documents = read_jsonl(dataset / "documents.jsonl")
    queries = read_jsonl(dataset / "queries.jsonl")
    pairs = read_jsonl(dataset / "pairs.jsonl")
    training = {}
    if args.adapter:
        adapter_path = Path(args.adapter)
        training_path = adapter_path / "training.json"
        if not training_path.exists():
            training_path = adapter_path.parent / "training.json"
        if not training_path.exists():
            raise FileNotFoundError(f"Missing training provenance: {training_path}")
        training = json.loads(training_path.read_text(encoding="utf-8"))
        base_revision = training["base_revision"]
        model = SentenceTransformer(
            STUDENT_MODEL,
            revision=base_revision,
            device="mps",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32},
            default_prompt_name=None,
        )
        model.load_adapter(args.adapter)
        model_id = str(args.adapter)
    else:
        base_revision = args.base_revision or HfApi().model_info(STUDENT_MODEL).sha
        model = SentenceTransformer(
            STUDENT_MODEL,
            revision=base_revision,
            device="mps",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32},
            default_prompt_name=None,
        )
        model_id = STUDENT_MODEL
    model.max_seq_length = training.get("settings", {}).get("max_length", 512)
    model_key = model_fingerprint(base_revision, args.adapter)
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else dataset.parent / "embedding_cache"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.cluster_profile) if getattr(args, "cluster_profile", None) else None
    profile = load_profile(profile_path) if profile_path else None
    if profile:
        saved_key = profile.get("provenance", {}).get("model_key")
        if saved_key and saved_key != model_key:
            raise ValueError("Cluster profile was fitted for another model or adapter")
        saved_input = profile.get("provenance", {}).get("clustering_deidentify")
        if saved_input is not None:
            if deidentify_arg != "auto" and deidentify != saved_input:
                raise ValueError("Input de-identification differs from frozen cluster profile")
            deidentify = saved_input
    train_docs = [doc for doc in documents if doc["split"] == "train"]
    dev_docs = [doc for doc in documents if doc["split"] == "dev"]
    test_docs = [doc for doc in documents if doc["split"] == "test"]
    dev_pairs = [pair for pair in pairs if pair["split"] == "dev"]
    test_pairs = [pair for pair in pairs if pair["split"] == "test"]
    dev_queries = [query for query in queries if query["split"] == "dev"]
    test_queries = [query for query in queries if query["split"] == "test"][:140]
    dev_chunks, dev_chunk_ids = encode_documents(model, dev_docs, "clustering", args.batch_size, cache_dir, model_key, deidentify)
    dev_vectors = document_vectors(dev_chunks, dev_chunk_ids, dev_docs)
    if profile:
        config = ClusterConfig.from_dict(profile["config"])
        selected_groups = cluster_groups(dev_docs, dev_vectors, config, profile["fitted"])
        selected = {"threshold": config.threshold, **all_event_pair_f1(dev_docs, selected_groups)}
        threshold_candidates = []
        selection_mode = "frozen_profile"
    elif getattr(args, "hybrid", False):
        train_chunks, train_chunk_ids = encode_documents(model, train_docs, "clustering", args.batch_size, cache_dir, model_key, deidentify)
        train_vectors = document_vectors(train_chunks, train_chunk_ids, train_docs)
        profile, threshold_candidates = select_profile(train_vectors, dev_docs, dev_vectors)
        config = ClusterConfig.from_dict(profile["config"])
        selected_groups = cluster_groups(dev_docs, dev_vectors, config, profile["fitted"])
        selected = {"threshold": config.threshold, **all_event_pair_f1(dev_docs, selected_groups)}
        profile_path = output.with_suffix(".cluster_profile.json")
        save_profile(profile_path, profile, {
            "base_revision": base_revision, "model_key": model_key,
            "clustering_deidentify": deidentify, "dataset_files": manifest["files"],
            "fit_split": "train", "selection_split": "dev",
            "training_document_ids_sha256": hashlib.sha256(json.dumps([doc["doc_id"] for doc in train_docs]).encode()).hexdigest(),
        })
        selection_mode = "development_hybrid"
    else:
        selected, threshold_candidates = select_cluster_threshold(dev_docs, dev_vectors)
        selected_groups = groups_for(dev_docs, dev_vectors, selected["threshold"])
        config = ClusterConfig(threshold=selected["threshold"])
        selection_mode = "development_legacy"
    test_chunks, test_chunk_ids = encode_documents(model, test_docs, "clustering", args.batch_size, cache_dir, model_key, deidentify)
    test_vectors = document_vectors(test_chunks, test_chunk_ids, test_docs)
    groups = cluster_groups(test_docs, test_vectors, config, profile["fitted"] if profile else None)
    labels = extract_labels(test_docs, groups)
    write_jsonl(output.with_suffix(".labels.jsonl"), labels)
    write_jsonl(output.with_suffix(".dev.labels.jsonl"), extract_labels(dev_docs, selected_groups))
    write_tag_review(output.with_suffix(".tag_review.csv"), labels, test_docs)
    test_raw_geometry = geometry_diagnostics(test_docs, test_vectors)
    dev_raw_geometry = geometry_diagnostics(dev_docs, dev_vectors)
    if profile and config.postprocess != "none":
        test_corrected_geometry = geometry_diagnostics(test_docs, transform_vectors(test_vectors, profile["fitted"]))
        dev_corrected_geometry = geometry_diagnostics(dev_docs, transform_vectors(dev_vectors, profile["fitted"]))
    else:
        test_corrected_geometry, dev_corrected_geometry = test_raw_geometry, dev_raw_geometry
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": manifest["status"],
        "model": model_id,
        "base_revision": base_revision,
        "teacher_id": manifest["teacher_id"],
        "teacher_revision": manifest["teacher_revision"],
        "prompt_version": manifest["prompt_version"],
        "dataset_revision": manifest.get("dataset_revision", "legacy"),
        "dataset_path": str(dataset.resolve()),
        "model_key": model_key,
        "clustering_deidentify": deidentify,
        "cluster_profile": str(profile_path.resolve()) if profile_path else None,
        "cluster_config": profile["config"] if profile else None,
        "cluster_selection_mode": selection_mode,
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "scikit_learn": version("scikit-learn"),
        },
        "dataset_files": manifest["files"],
        "dev_selected_threshold": selected["threshold"],
        "dev_threshold_candidates": threshold_candidates,
        "dev_all_event_pairs": selected,
        "dev_sampled_pairs": pair_f1(dev_pairs, selected_groups),
        "dev_dense_search": dense_search(model, dev_docs, dev_queries, args.batch_size, cache_dir, model_key),
        "test_pair": pair_f1(test_pairs, groups),
        "test_all_event_pairs": all_event_pair_f1(test_docs, groups),
        "test_dense_search": dense_search(model, test_docs, test_queries, args.batch_size, cache_dir, model_key),
        "test_lexical_search": lexical_search(test_docs, test_queries),
        "test_shortcut_baselines": shortcut_baselines(test_docs, test_queries),
        "test_groups": len(labels),
        "test_groups_unnamed": sum(label["label"] == "이름 미정" for label in labels),
        "tag_review_sample_count": min(30, len(labels)),
        "test_error_diagnostics": error_diagnostics(test_docs, groups),
        "test_geometry": {"raw": test_raw_geometry, "corrected": test_corrected_geometry},
        "dev_geometry": {"raw": dev_raw_geometry, "corrected": dev_corrected_geometry},
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(output.resolve()),
        "status": report["status"],
        "cluster_profile": report["cluster_profile"],
        "cluster_config": config.__dict__,
        "selection_status": selection_mode,
        **{key: report[key] for key in (
            "dev_all_event_pairs", "dev_dense_search", "test_pair",
            "test_all_event_pairs", "test_dense_search", "test_groups", "test_groups_unnamed",
        )},
        "test_targets_met": report["test_dense_search"]["recall_at_5"] >= 0.90 and report["test_all_event_pairs"]["f1"] >= 0.80,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/dataset")
    parser.add_argument("--adapter")
    parser.add_argument("--base-revision")
    parser.add_argument("--output", default="artifacts/evaluation/base.json")
    parser.add_argument("--batch-size", type=int, default=2)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument("--cluster-profile", help="Load a frozen train-fitted, dev-selected cluster profile")
    profile_group.add_argument("--hybrid", action="store_true", help="Fit on train, select the hybrid profile on dev, then freeze before test")
    parser.add_argument("--deidentify", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cache-dir")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
