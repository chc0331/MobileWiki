"""Train-fitted embedding correction and reproducible event clustering."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ClusterConfig:
    postprocess: str = "none"
    remove_components: int = 0
    linkage: str = "average"
    threshold: float = 0.15
    date_window_days: int = 7
    date_penalty: float = 0.0
    mutual_k: int = 0
    neighbor_penalty: float = 0.0
    resplit: bool = False
    max_cluster_size: int = 16
    split_ratio: float = 0.75
    max_split_depth: int = 2

    def __post_init__(self):
        if self.postprocess not in {"none", "center", "abtt"}:
            raise ValueError("Unknown embedding postprocessor")
        if self.linkage not in {"average", "complete"}:
            raise ValueError("Only average and complete linkage are supported")
        if self.remove_components < 0 or (self.postprocess != "abtt" and self.remove_components):
            raise ValueError("Principal components may only be removed in abtt mode")
        if not 0 < self.threshold <= 2 or self.date_window_days < 0:
            raise ValueError("Invalid distance threshold or date window")
        if self.mutual_k < 0 or min(self.date_penalty, self.neighbor_penalty) < 0:
            raise ValueError("Distance penalties and neighbor count cannot be negative")
        if self.max_cluster_size < 2 or not 0 < self.split_ratio < 1 or self.max_split_depth < 0:
            raise ValueError("Invalid re-splitting settings")

    @classmethod
    def from_dict(cls, value: dict) -> "ClusterConfig":
        return cls(**value)


def _vectors(value: np.ndarray) -> np.ndarray:
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must be a finite two-dimensional array")
    return vectors


def _normalize(value: np.ndarray) -> np.ndarray:
    vectors = _vectors(value)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def vector_hash(vectors: np.ndarray) -> str:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return hashlib.sha256(str(vectors.shape).encode() + vectors.tobytes()).hexdigest()


def fit_postprocessor(train_vectors: np.ndarray, mode: str = "none", remove_components: int = 0) -> dict:
    """Fit only on the explicitly supplied training vectors; no labels are read."""
    vectors = _vectors(train_vectors)
    if not len(vectors):
        raise ValueError("Training vectors are required to fit the postprocessor")
    if mode not in {"none", "center", "abtt"} or remove_components < 0:
        raise ValueError("Invalid postprocessor")
    if mode != "abtt" and remove_components:
        raise ValueError("Only abtt removes principal components")
    mean = vectors.mean(axis=0) if mode != "none" else np.zeros(vectors.shape[1])
    components = np.empty((0, vectors.shape[1]))
    if mode == "abtt" and remove_components:
        _, _, directions = np.linalg.svd(vectors - mean, full_matrices=False)
        components = directions[: min(remove_components, len(directions))]
    return {
        "mode": mode,
        "remove_components": remove_components,
        "fitted_rows": len(vectors),
        "training_vector_sha256": vector_hash(vectors),
        "mean": np.asarray(mean, dtype=np.float32),
        "components": np.asarray(components, dtype=np.float32),
    }


def transform_vectors(vectors: np.ndarray, fitted: dict) -> np.ndarray:
    values = _vectors(vectors)
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    components = np.asarray(fitted["components"], dtype=np.float64)
    if mean.shape != (values.shape[1],) or components.ndim != 2 or components.shape[1] != values.shape[1]:
        raise ValueError("Postprocessor and embedding dimensions differ")
    corrected = values - mean
    if len(components):
        corrected -= (corrected @ components.T) @ components
    return np.asarray(_normalize(corrected), dtype=np.float32)


_ABSOLUTE_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{4}년\s*\d{1,2}월\s*\d{1,2}일")


def text_date(document: dict) -> date | None:
    """Read a single unambiguous full date from text, never synthetic metadata."""
    found = set()
    for match in _ABSOLUTE_DATE.finditer(document.get("title", "") + " " + document.get("text", "")):
        numbers = [int(value) for value in re.findall(r"\d+", match.group())]
        try:
            found.add(date(*numbers))
        except ValueError:
            continue
    return next(iter(found)) if len(found) == 1 else None


def distance_matrix(documents: list[dict], corrected_vectors: np.ndarray, config: ClusterConfig) -> np.ndarray:
    vectors = _normalize(corrected_vectors)
    if len(documents) != len(vectors):
        raise ValueError("Documents and vectors have different row counts")
    base = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    distances = base.copy()
    if config.date_penalty:
        dates = [text_date(document) for document in documents]
        for left, left_date in enumerate(dates):
            if left_date is None:
                continue
            for right in range(left + 1, len(dates)):
                right_date = dates[right]
                if right_date is not None and abs((left_date - right_date).days) > config.date_window_days:
                    distances[left, right] += config.date_penalty
                    distances[right, left] += config.date_penalty
    if config.mutual_k and config.neighbor_penalty and len(documents) > 1:
        neighbors = np.zeros(base.shape, dtype=bool)
        doc_ids = np.asarray([document["doc_id"] for document in documents])
        for index in range(len(documents)):
            order = np.lexsort((doc_ids, base[index]))
            order = order[order != index][: min(config.mutual_k, len(documents) - 1)]
            neighbors[index, order] = True
        distances += config.neighbor_penalty * ~(neighbors & neighbors.T)
    distances = np.clip((distances + distances.T) / 2, 0, 2)
    np.fill_diagonal(distances, 0.0)
    return np.asarray(distances, dtype=np.float64)


def _tree(distances: np.ndarray, linkage: str) -> tuple[np.ndarray, np.ndarray]:
    if len(distances) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0, metric="precomputed", linkage=linkage,
        compute_full_tree=True, compute_distances=True,
    ).fit(distances)
    return model.children_, model.distances_


def _cut_tree(children: np.ndarray, merge_distances: np.ndarray, count: int, threshold: float) -> np.ndarray:
    parents = np.arange(max(1, 2 * count - 1))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for offset, ((left, right), distance) in enumerate(zip(children, merge_distances)):
        if distance >= threshold:
            break
        merged = count + offset
        parents[root(left)] = parents[root(right)] = merged
    labels: dict[int, int] = {}
    return np.asarray([labels.setdefault(root(index), len(labels)) for index in range(count)], dtype=int)


def _split_labels(distances: np.ndarray, semantic: np.ndarray, labels: np.ndarray, config: ClusterConfig) -> np.ndarray:
    members = [np.flatnonzero(labels == label) for label in sorted(set(labels))]

    def split(indices, threshold, depth):
        if len(indices) < 2 or depth >= config.max_split_depth:
            return [indices]
        local_semantic = semantic[np.ix_(indices, indices)]
        triangle = local_semantic[np.triu_indices(len(indices), 1)]
        dispersed = triangle.mean() > 0.6 * threshold and triangle.max() > 0.9 * threshold
        if len(indices) <= config.max_cluster_size and not dispersed:
            return [indices]
        local = distances[np.ix_(indices, indices)]
        children, heights = _tree(local, "complete")
        smaller = threshold * config.split_ratio
        local_labels = _cut_tree(children, heights, len(indices), smaller)
        if len(set(local_labels)) == 1:
            return [indices]
        return [part for label in sorted(set(local_labels)) for part in split(indices[local_labels == label], smaller, depth + 1)]

    output = np.empty(len(labels), dtype=int)
    groups = [part for member in members for part in split(member, config.threshold, 0)]
    for label, indices in enumerate(groups):
        output[indices] = label
    return output


def cluster_groups(documents: list[dict], raw_vectors: np.ndarray, config: ClusterConfig, fitted: dict | None = None) -> dict[str, int]:
    if not documents:
        return {}
    if fitted is None:
        if config.postprocess != "none":
            raise ValueError("A train-fitted postprocessor is required")
        fitted = fit_postprocessor(raw_vectors, "none")
    if fitted["mode"] != config.postprocess or fitted.get("remove_components", 0) != config.remove_components:
        raise ValueError("Cluster configuration and fitted postprocessor differ")
    corrected = transform_vectors(raw_vectors, fitted)
    distances = distance_matrix(documents, corrected, config)
    children, heights = _tree(distances, config.linkage)
    labels = _cut_tree(children, heights, len(documents), config.threshold)
    if config.resplit:
        semantic = np.clip(1.0 - corrected.astype(np.float64) @ corrected.astype(np.float64).T, 0, 2)
        labels = _split_labels(distances, semantic, labels, config)
    return {document["doc_id"]: int(label) for document, label in zip(documents, labels)}


def all_event_pair_f1(documents: list[dict], groups: dict[str, int]) -> dict:
    event_counts = Counter(document["event_id"] for document in documents)
    group_counts = Counter(groups[document["doc_id"]] for document in documents)
    joints = Counter((groups[document["doc_id"]], document["event_id"]) for document in documents)
    choose_two = lambda value: value * (value - 1) // 2
    true_pairs = sum(map(choose_two, event_counts.values()))
    predicted_pairs = sum(map(choose_two, group_counts.values()))
    tp = sum(map(choose_two, joints.values()))
    fp, fn = predicted_pairs - tp, true_pairs - tp
    precision = tp / predicted_pairs if predicted_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _candidate_key(row: dict) -> tuple:
    config = ClusterConfig.from_dict(row["config"])
    complexity = (config.postprocess != "none") + (config.postprocess == "abtt") + bool(config.date_penalty) + bool(config.neighbor_penalty) + config.resplit
    return row["f1"], -row["fp"], row["precision"], -complexity, -config.threshold


def select_profile(train_vectors: np.ndarray, dev_documents: list[dict], dev_vectors: np.ndarray) -> tuple[dict, list[dict]]:
    """Search the predefined grid using development labels, with train-only correction."""
    if not dev_documents:
        raise ValueError("Development documents are required for profile selection")
    fitted_cache = {}
    corrected_cache = {}
    candidates = []

    def fitted_for(config):
        key = config.postprocess, config.remove_components
        if key not in fitted_cache:
            fitted_cache[key] = fit_postprocessor(train_vectors, *key)
            corrected_cache[key] = transform_vectors(dev_vectors, fitted_cache[key])
        return fitted_cache[key], corrected_cache[key]

    def search(config, stage):
        _, corrected = fitted_for(config)
        distances = distance_matrix(dev_documents, corrected, config)
        children, heights = _tree(distances, config.linkage)

        def measure(threshold):
            labels = _cut_tree(children, heights, len(dev_documents), threshold)
            groups = {document["doc_id"]: int(label) for document, label in zip(dev_documents, labels)}
            return {"threshold": threshold, **all_event_pair_f1(dev_documents, groups)}

        scores = [measure(round(value * 0.05, 2)) for value in range(1, 31)]
        best = max(scores, key=lambda row: (row["f1"], -row["fp"], -row["threshold"]))
        near = {round(best["threshold"] + offset * 0.01, 2) for offset in range(-4, 5)}
        scores.extend(measure(value) for value in sorted(near) if 0 < value <= 1.5)
        best = max(scores, key=lambda row: (row["f1"], -row["fp"], -row["threshold"]))
        selected_config = replace(config, threshold=best["threshold"])
        row = {"stage": stage, **best, "config": asdict(selected_config), "threshold_candidates": scores}
        candidates.append(row)
        return row

    stage_one = [
        search(ClusterConfig(postprocess=mode, remove_components=count, linkage=linkage), "geometry")
        for mode, count in (("none", 0), ("center", 0), ("abtt", 1), ("abtt", 3))
        for linkage in ("average", "complete")
    ]
    stage_two = []
    seen = set()
    for base in sorted(stage_one, key=_candidate_key, reverse=True)[:3]:
        original = ClusterConfig.from_dict(base["config"])
        for window, date_penalty in ((7, 0.0), (7, 0.15), (7, 0.30), (30, 0.15), (30, 0.30)):
            for mutual_k, penalty in ((0, 0.0), (5, 0.15), (10, 0.15), (20, 0.15)):
                config = replace(original, date_window_days=window, date_penalty=date_penalty, mutual_k=mutual_k, neighbor_penalty=penalty)
                key = json.dumps(asdict(replace(config, threshold=0.15)), sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    stage_two.append(search(config, "hybrid"))
    for base in sorted(stage_two, key=_candidate_key, reverse=True)[:3]:
        for limit in (16, 32):
            config = replace(ClusterConfig.from_dict(base["config"]), resplit=True, max_cluster_size=limit)
            fitted, _ = fitted_for(config)
            metrics = all_event_pair_f1(dev_documents, cluster_groups(dev_documents, dev_vectors, config, fitted))
            candidates.append({"stage": "resplit", "threshold": config.threshold, **metrics, "config": asdict(config)})
    selected = max(candidates, key=_candidate_key)
    config = ClusterConfig.from_dict(selected["config"])
    fitted, _ = fitted_for(config)
    return {"config": asdict(config), "fitted": fitted, "selection": selected}, candidates


def save_profile(path: Path | str, profile: dict, provenance: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = profile["fitted"]
    vectors_path = path.with_suffix(".npz")
    np.savez_compressed(vectors_path, mean=stats["mean"], components=stats["components"])
    value = {
        "schema_version": 1,
        "config": profile["config"],
        "fitted": {key: item for key, item in stats.items() if key not in {"mean", "components"}},
        "transform_file": vectors_path.name,
        "transform_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "selection": profile.get("selection"),
        "provenance": provenance or profile.get("provenance", {}),
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_profile(path: Path | str) -> dict:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported cluster profile schema")
    ClusterConfig.from_dict(value["config"])
    transform_path = path.parent / value["transform_file"]
    if hashlib.sha256(transform_path.read_bytes()).hexdigest() != value["transform_sha256"]:
        raise ValueError("Cluster transform hash mismatch")
    with np.load(transform_path, allow_pickle=False) as arrays:
        value["fitted"] = {**value["fitted"], "mean": arrays["mean"], "components": arrays["components"]}
    return value


def choose_checkpoint(candidates: list[dict], search_target: float = 0.90) -> dict:
    if not candidates:
        raise ValueError("No checkpoint candidates")
    eligible = [row for row in candidates if row["search_recall_at_5"] >= search_target]

    def key(row):
        step = int(row["checkpoint"].rsplit("-", 1)[-1])
        clustering = (row["all_event_pair_f1"], -row["all_event_pair_false_merges"], row["search_mrr"], -step)
        return clustering if eligible else (row["search_recall_at_5"], *clustering)

    chosen = dict(max(eligible or candidates, key=key))
    chosen["eligible_search"] = bool(eligible)
    chosen["selection_status"] = "eligible" if eligible else "search_target_unmet"
    return chosen


def error_diagnostics(documents: list[dict], groups: dict[str, int]) -> dict:
    """Use gold metadata only after prediction, for human-readable error slices."""
    members = defaultdict(list)
    for document in documents:
        members[groups[document["doc_id"]]].append(document)
    slices = Counter()
    largest = []
    for group_id, items in members.items():
        false_merges = 0
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                if left["event_id"] == right["event_id"]:
                    continue
                false_merges += 1
                for field, label in (("family_id", "same_family"), ("category", "same_category"), ("date", "same_date"), ("fact_value", "same_fact")):
                    if left.get(field) is not None and left.get(field) == right.get(field):
                        slices[label] += 1
                if left.get("date") and left.get("date") == right.get("date") and left.get("category") == right.get("category"):
                    slices["same_category_and_date"] += 1
                if all(item.get("answer_doc_index") is not None and not item["doc_id"].endswith(f"-d{item['answer_doc_index']}") for item in (left, right)):
                    slices["both_non_answer"] += 1
        largest.append({"group_id": group_id, "documents": len(items), "events": len({item["event_id"] for item in items}), "false_merges": false_merges})
    return {"false_merge_slices": dict(slices), "largest_error_groups": sorted(largest, key=lambda row: (-row["false_merges"], -row["documents"], row["group_id"]))[:20]}


def geometry_diagnostics(documents: list[dict], vectors: np.ndarray, k: int = 10) -> dict:
    """Measure geometry without using gold event labels or fitting any transform."""
    values = _normalize(vectors)
    if len(values) != len(documents):
        raise ValueError("Documents and vectors have different row counts")
    if not len(values):
        return {"documents": 0, "mean_pair_cosine": None, "top_pc_variance_ratio": [], "neighbor_k": 0, "indegree_max": 0, "indegree_p95": 0, "top_hubs": []}
    similarity = values @ values.T
    mean_cosine = float((similarity.sum() - np.trace(similarity)) / (len(values) * (len(values) - 1))) if len(values) > 1 else None
    centered = values - values.mean(axis=0)
    variance = []
    if len(values) > 1 and np.square(centered).sum() > 1e-12:
        pca = PCA(n_components=min(3, *values.shape), svd_solver="randomized", random_state=42)
        pca.fit(values)
        variance = [float(value) for value in pca.explained_variance_ratio_]
    neighbor_k = max(0, min(k, len(values) - 1))
    indegree = np.zeros(len(values), dtype=int)
    doc_ids = np.asarray([document["doc_id"] for document in documents])
    for index in range(len(values)):
        order = np.lexsort((doc_ids, -similarity[index]))
        neighbors = order[order != index][:neighbor_k]
        indegree[neighbors] += 1
    hub_order = sorted(range(len(values)), key=lambda index: (-indegree[index], doc_ids[index]))[:10]
    return {
        "documents": len(values), "mean_pair_cosine": mean_cosine,
        "top_pc_variance_ratio": variance, "variance_method": "randomized PCA, seed 42",
        "neighbor_k": neighbor_k, "indegree_max": int(indegree.max()),
        "indegree_p95": float(np.percentile(indegree, 95)),
        "top_hubs": [{"doc_id": str(doc_ids[index]), "indegree": int(indegree[index])} for index in hub_order],
    }
