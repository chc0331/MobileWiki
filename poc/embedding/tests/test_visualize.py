import json
from dataclasses import asdict

import numpy as np
import pytest

from clustering import ClusterConfig, fit_postprocessor, save_profile
from common import sha256, write_jsonl
from visualize import (
    ARTIFACT_TYPE, BASE_DIR, SCHEMA_VERSION, assignments_from_labels, bundle_display_name, content_hash,
    document_key, error_pairs, export_bundle, group_summaries, load_bundle, projection,
    profile_fit_scope, recluster_development, same_dataset, selected_doc_ids,
)


def documents():
    return [
        {"doc_id": "a-d0", "event_id": "a", "title": "첫 메모", "text": "2026-01-01 첫 사건", "split": "dev", "category": "같은 주제", "date": "2026-01-01", "family_id": "family-a", "fact_value": "답", "answer_doc_index": 0},
        {"doc_id": "a-d1", "event_id": "a", "title": "둘째 메모", "text": "2026-01-01 첫 사건 준비", "split": "dev", "category": "같은 주제", "date": "2026-01-01", "family_id": "family-a", "fact_value": "답", "answer_doc_index": 0},
        {"doc_id": "b-d0", "event_id": "b", "title": "다른 사건", "text": "2026-01-01 둘째 사건", "split": "dev", "category": "같은 주제", "date": "2026-01-01", "family_id": "family-b", "fact_value": "답", "answer_doc_index": 0},
        {"doc_id": "b-d1", "event_id": "b", "title": "다른 준비", "text": "2026-01-01 둘째 사건 준비", "split": "dev", "category": "같은 주제", "date": "2026-01-01", "family_id": "family-b", "fact_value": "답", "answer_doc_index": 0},
    ]


def test_plotly_selection_uses_custom_id_across_traces_and_filters():
    state = {"selection": {"points": [
        {"point_index": 0, "curve_number": 1, "customdata": ["b-d1"]},
        {"point_index": 0, "curve_number": 0, "customdata": ["a-d0"]},
        {"customdata": ["not-in-split"]}, {"customdata": ["b-d1"]},
    ]}}
    assert selected_doc_ids(state, {"a-d0", "b-d1"}) == ["b-d1", "a-d0"]
    assert selected_doc_ids({"selection": {"point_indices": [0]}}, {"a-d0"}) == []


def test_document_identity_requires_matching_dataset_hash():
    left = {"dataset_documents_sha256": "v2"}
    right = {"dataset_documents_sha256": "v3"}
    assert not same_dataset(left, right)
    assert document_key("v2", "test-0000-d0") != document_key("v3", "test-0000-d0")
    assert same_dataset(left, dict(left))


def test_old_duplicate_run_ids_display_distinct_evaluation_sources():
    common = {"run_id": "shortcut-hardneg-v2-balanced/tuned", "dataset_revision": "shortcut-hardneg-v2-balanced"}
    original = {**common, "evaluation_path": str(BASE_DIR / "artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.json")}
    corrected = {**common, "evaluation_path": str(BASE_DIR / "artifacts/cluster-v3-phase1/evaluation/tuned.json")}
    assert bundle_display_name(original) != bundle_display_name(corrected)
    assert "cluster-v3-phase1/evaluation/tuned" in bundle_display_name(corrected)


def test_foreign_frozen_profile_preserves_fit_origin_for_regressions():
    train = np.array([[1, 0], [0, 1]], dtype=np.float32)
    other_train = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    profile = {"fitted": fit_postprocessor(train, "center"), "provenance": {"dataset_files": {"documents.jsonl": "v3-documents"}}}
    assert profile_fit_scope(profile, other_train, "v2-documents") == ("foreign_training_dataset", "v3-documents")
    np.testing.assert_array_equal(profile["fitted"]["mean"], train.mean(axis=0))
    with pytest.raises(ValueError, match="different training vectors"):
        profile_fit_scope(profile, other_train, "v3-documents")
    assert profile_fit_scope(profile, train, "v3-documents") == ("current_training_dataset", "v3-documents")


def export_inputs(tmp_path, saved_provenance, explicit=True):
    from argparse import Namespace
    from common import STUDENT_MODEL
    from evaluate import model_fingerprint

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    docs = [{**doc, "doc_id": f"{split}-{doc['doc_id']}", "split": split}
            for split in ("train", "dev", "test") for doc in documents()]
    write_jsonl(dataset / "documents.jsonl", docs)
    (dataset / "manifest.json").write_text(json.dumps({"dataset_revision": "legacy-v1"}))
    profile_path = tmp_path / "frozen.json"
    profile = {"config": asdict(ClusterConfig(threshold=0.1)),
               "fitted": fit_postprocessor(np.array([[0.3, 0.7], [0.8, 0.2]], dtype=np.float32)),
               "provenance": {"model_key": model_fingerprint("fixed"), "clustering_deidentify": True,
                              "dataset_files": {"documents.jsonl": "v3-documents"}, **saved_provenance}}
    save_profile(profile_path, profile)
    report = {"model": STUDENT_MODEL, "base_revision": "fixed", "model_key": model_fingerprint("fixed"),
              "clustering_deidentify": True, "dataset_files": {"documents.jsonl": sha256(dataset / "documents.jsonl")}}
    if not explicit:
        report["cluster_profile_path"] = str(profile_path)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(report))
    return Namespace(dataset=str(dataset), evaluation=str(evaluation), output=str(tmp_path / "bundle"),
                     cluster_profile=str(profile_path) if explicit else None, device="cpu", batch_size=2,
                     force=False, run_id=None)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("saved, message", [
    ({"model_key": "another-adapter"}, "another model or adapter"),
    ({"clustering_deidentify": False}, "de-identification differs"),
])
def test_export_rejects_profile_input_mismatches_before_model_loading(tmp_path, monkeypatch, explicit, saved, message):
    import sentence_transformers

    args = export_inputs(tmp_path, saved, explicit)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: pytest.fail("An incompatible profile must fail before inference"))
    with pytest.raises(ValueError, match=message):
        export_bundle(args)
    assert not (tmp_path / "bundle").exists()


def test_export_reuses_same_model_profile_on_foreign_regression_dataset(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evaluate
    import sentence_transformers
    import visualize

    args = export_inputs(tmp_path, {})
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: SimpleNamespace())
    vectors = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=np.float32)
    monkeypatch.setattr(evaluate, "encode_documents", lambda model, docs, *args: (vectors, [doc["doc_id"] for doc in docs]))
    monkeypatch.setattr(visualize, "projection", lambda values, *args, **kwargs: np.zeros((len(values), 2)))
    export_bundle(args)
    bundle = load_bundle(tmp_path / "bundle")
    assert bundle["manifest"]["fit_scope"] == "foreign_training_dataset"
    assert bundle["manifest"]["fit_dataset_documents_sha256"] == "v3-documents"
    assert bundle["profile"]["provenance"]["dataset_files"]["documents.jsonl"] == "v3-documents"
    assert bundle["profile"]["provenance"]["clustering_deidentify"] is True


def test_assignments_reject_duplicate_or_missing_document_rows():
    docs = documents()
    labels = [{"group_id": 0, "doc_ids": [doc["doc_id"] for doc in docs]}]
    assert len(assignments_from_labels(docs, labels)) == 4
    with pytest.raises(ValueError, match="multiple groups"):
        assignments_from_labels(docs, labels + [{"group_id": 1, "doc_ids": ["a-d0"]}])
    with pytest.raises(ValueError, match="do not match"):
        assignments_from_labels(docs, [{"group_id": 0, "doc_ids": ["a-d0"]}])


def test_error_diagnostics_count_false_merges_and_splits_without_full_pair_table():
    docs = documents()
    groups = {"a-d0": 0, "a-d1": 1, "b-d0": 0, "b-d1": 0}
    rows = error_pairs(docs, groups)
    assert len([row for row in rows if row["error"] == "오결합"]) == 2
    assert len([row for row in rows if row["error"] == "과분할"]) == 1
    assert all(row["same_date"] and row["same_category"] for row in rows)
    assert sum(row["same_family"] is True for row in rows) == 1
    summaries = group_summaries(docs, groups, [])
    assert summaries[0]["false_merge_pairs"] == 2
    assert summaries[0]["events"] == 2
    assert sum(row["false_merge_pairs"] for row in summaries) == 2
    legacy = [{key: value for key, value in doc.items() if key not in {"date", "family_id", "fact_value", "answer_doc_index"}} for doc in docs]
    assert error_pairs(legacy, groups)[0]["same_date"] is None
    assert error_pairs(legacy, groups)[0]["both_answerless"] is None


def test_projection_handles_small_inputs_and_fits_pca_on_train_only():
    train = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 1]], dtype=np.float32)
    target = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.2]], dtype=np.float32)
    first = projection(target, train_vectors=train)
    extra = projection(np.vstack((target, [100, -100, 100])), train_vectors=train)
    np.testing.assert_allclose(first, extra[:2])
    assert projection(np.empty((0, 3))).shape == (0, 2)
    np.testing.assert_array_equal(projection(train[:1]), np.zeros((1, 2)))


def make_bundle(tmp_path):
    docs = documents()
    vectors = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=np.float32)
    config = ClusterConfig(threshold=0.1)
    profile = {"config": asdict(config), "fitted": fit_postprocessor(vectors)}
    save_profile(tmp_path / "cluster_profile.json", profile)
    np.savez_compressed(tmp_path / "train.npz", raw=vectors)
    (tmp_path / "evaluation.json").write_text(json.dumps({"status": "provisional_pending_human_review"}))
    ids = [doc["doc_id"] for doc in docs]
    labels = [{"group_id": 0, "label": "첫 사건", "doc_ids": ids[:2]}, {"group_id": 1, "label": "둘째 사건", "doc_ids": ids[2:]}]
    write_jsonl(tmp_path / "dev.documents.jsonl", docs)
    write_jsonl(tmp_path / "dev.labels.jsonl", labels)
    np.savez_compressed(tmp_path / "dev.npz", raw=vectors, corrected=vectors, pca=np.zeros((4, 2)), tsne=np.zeros((4, 2)))
    manifest = {
        "schema_version": SCHEMA_VERSION, "artifact_type": ARTIFACT_TYPE,
        "dataset_documents_sha256": "same-documents", "run_id": "test", "dataset_revision": "test",
        "base_revision": "test-base", "status": "provisional_pending_human_review",
        "splits": {"dev": {"documents": "dev.documents.jsonl", "labels": "dev.labels.jsonl", "vectors": "dev.npz", "doc_ids": ids, "row_order_sha256": content_hash(ids)}},
        "files": {path.name: sha256(path) for path in tmp_path.iterdir()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_bundle_roundtrip_checks_hashes_and_vector_row_order(tmp_path):
    manifest = make_bundle(tmp_path)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    loaded = load_bundle(tmp_path)
    assert loaded["splits"]["dev"]["groups"]["a-d1"] == 0
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
    manifest["splits"]["dev"]["doc_ids"] = list(reversed(manifest["splits"]["dev"]["doc_ids"]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row order"):
        load_bundle(tmp_path)


def test_bundle_rejects_changed_data_even_if_ids_still_exist(tmp_path):
    make_bundle(tmp_path)
    path = tmp_path / "dev.documents.jsonl"
    path.write_text(path.read_text().replace("첫 사건", "고친 사건"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(tmp_path)


def test_dev_experiment_cannot_recluster_test_or_mutate_frozen_bundle(tmp_path):
    make_bundle(tmp_path)
    bundle = load_bundle(tmp_path)
    with pytest.raises(ValueError, match="Only the development"):
        recluster_development(bundle, "test", ClusterConfig())
    original_groups = dict(bundle["splits"]["dev"]["groups"])
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    labeler = lambda docs, groups: [{"group_id": group, "doc_ids": [doc_id for doc_id, value in groups.items() if value == group]} for group in sorted(set(groups.values()))]
    view = recluster_development(bundle, "dev", ClusterConfig(threshold=0.2, postprocess="center"), labeler=labeler)
    assert view["exploratory"] is True
    assert view["profile"]["fitted"]["fitted_rows"] == len(bundle["train_vectors"])
    assert bundle["splits"]["dev"]["groups"] == original_groups
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files
