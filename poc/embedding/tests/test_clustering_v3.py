from types import SimpleNamespace

import numpy as np
import pytest

from clustering import (
    ClusterConfig,
    choose_checkpoint,
    cluster_groups,
    distance_matrix,
    fit_postprocessor,
    geometry_diagnostics,
    load_profile,
    save_profile,
    select_profile,
    text_date,
    transform_vectors,
    vector_hash,
)
from evaluate import encode_texts, error_diagnostics, extract_labels


def documents(count):
    return [{"doc_id": f"d{index}", "event_id": f"e{index // 2}", "title": "", "text": ""} for index in range(count)]


def test_postprocessing_fits_training_rows_and_restores_saved_transform(tmp_path):
    train = np.asarray([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]], dtype=np.float32)
    heldout = np.asarray([[9, 2, 3]], dtype=np.float32)
    fitted = fit_postprocessor(train, "abtt", 1)
    assert fitted["mean"].tolist() == [1, 0, 0]
    assert fitted["fitted_rows"] == 4
    assert fitted["training_vector_sha256"] == vector_hash(train)
    before = transform_vectors(heldout, fitted)
    assert np.linalg.norm(before[0]) == pytest.approx(1)
    profile = {"config": ClusterConfig(postprocess="abtt", remove_components=1).__dict__, "fitted": fitted}
    path = tmp_path / "profile.json"
    save_profile(path, profile, {"fit_split": "train"})
    loaded = load_profile(path)
    np.testing.assert_allclose(transform_vectors(heldout, loaded["fitted"]), before)
    assert loaded["provenance"]["fit_split"] == "train"
    (tmp_path / "profile.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_profile(path)


def test_centering_handles_zero_vectors_and_requires_explicit_training_fit():
    vectors = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
    fitted = fit_postprocessor(vectors, "center")
    assert np.isfinite(transform_vectors(vectors, fitted)).all()
    with pytest.raises(ValueError, match="train-fitted"):
        cluster_groups(documents(2), vectors, ClusterConfig(postprocess="center"))
    with pytest.raises(ValueError, match="finite"):
        fit_postprocessor(np.asarray([[np.nan, 1]]))
    with pytest.raises(ValueError, match="differ"):
        cluster_groups(documents(2), vectors, ClusterConfig(), fitted)


def test_dates_use_only_unambiguous_full_dates_in_text():
    assert text_date({"text": "예약은 2026년 3월 15일이다.", "date": "2000-01-01"}).isoformat() == "2026-03-15"
    assert text_date({"text": "3월 15일 예약", "date": "2026-03-15"}) is None
    assert text_date({"text": "2026-03-15 확인 후 2026-04-15 방문"}) is None
    rows = documents(3)
    rows[0].update(text="2026-03-01 예약", date="2030-01-01")
    rows[1].update(text="2026년 3월 20일 방문", date="2030-01-01")
    rows[2].update(text="3월 20일 방문", date="1999-01-01")
    config = ClusterConfig(date_window_days=7, date_penalty=0.3)
    matrix = distance_matrix(rows, np.ones((3, 2)), config)
    assert matrix[0, 1] == pytest.approx(0.3)
    assert matrix[0, 2] == pytest.approx(0)


def test_mutual_neighbor_penalty_is_symmetric_with_zero_diagonal():
    angles = np.asarray([0, 0.01, 0.05])
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    config = ClusterConfig(mutual_k=1, neighbor_penalty=0.15)
    base = distance_matrix(documents(3), vectors, ClusterConfig())
    matrix = distance_matrix(documents(3), vectors, config)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    assert matrix[0, 1] == pytest.approx(base[0, 1])
    assert matrix[1, 2] == pytest.approx(base[1, 2] + 0.15)


def test_complete_linkage_prevents_a_loose_three_document_merge():
    angles = np.asarray([0, 0.15, 0.30])
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    average = cluster_groups(documents(3), vectors, ClusterConfig(threshold=0.04))
    complete = cluster_groups(documents(3), vectors, ClusterConfig(threshold=0.04, linkage="complete"))
    assert len(set(average.values())) == 1
    assert len(set(complete.values())) == 2


def test_empty_singleton_and_resplit_do_not_force_four_documents_per_group():
    assert cluster_groups([], np.empty((0, 2)), ClusterConfig()) == {}
    assert cluster_groups(documents(1), np.asarray([[1, 0]]), ClusterConfig()) == {"d0": 0}
    groups = cluster_groups(documents(7), np.tile([1.0, 0], (7, 1)), ClusterConfig(resplit=True, max_cluster_size=2))
    assert len(set(groups.values())) == 1


def test_profile_grid_uses_heldout_labels_but_training_only_statistics():
    train = np.asarray([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1]], dtype=np.float32)
    dev = np.asarray([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0], [0.01, 0.99, 0]], dtype=np.float32)
    profile, candidates = select_profile(train, documents(4), dev)
    assert profile["fitted"]["training_vector_sha256"] == vector_hash(train)
    assert profile["fitted"]["fitted_rows"] == len(train)
    assert profile["selection"]["f1"] == pytest.approx(1)
    assert {row["stage"] for row in candidates} == {"geometry", "hybrid", "resplit"}


def test_checkpoint_selection_has_a_search_floor_and_explicit_failure():
    rows = [
        {"checkpoint": "checkpoint-125", "search_recall_at_5": 0.89, "all_event_pair_f1": 0.95, "all_event_pair_false_merges": 1, "search_mrr": 0.8},
        {"checkpoint": "checkpoint-250", "search_recall_at_5": 0.91, "all_event_pair_f1": 0.85, "all_event_pair_false_merges": 2, "search_mrr": 0.8},
    ]
    assert choose_checkpoint(rows)["checkpoint"] == "checkpoint-250"
    assert choose_checkpoint(rows)["eligible_search"] is True
    rows[1]["search_recall_at_5"] = 0.88
    chosen = choose_checkpoint(rows)
    assert chosen["checkpoint"] == "checkpoint-125"
    assert chosen["selection_status"] == "search_target_unmet"


def fake_tokenizer(content):
    return [SimpleNamespace(form=word, tag="NNG") for word in content.split()]


def test_ctfidf_prefers_specific_terms_and_records_support_and_fallback():
    rows = documents(6)
    bodies = ["딸기 딸기 공통", "딸기 공통", "공통 바나나", "공통 바나나", "자두 복숭아", "참외"]
    for row, body in zip(rows, bodies):
        row["text"] = body
    groups = {f"d{i}": i // 2 for i in range(6)}
    labels = extract_labels(rows, groups, fake_tokenizer)
    assert labels[0]["label"] == "딸기"
    assert labels[0]["support_count"] == 2
    assert labels[0]["label_stage"] == "majority"
    assert labels[2]["label_stage"] == "representative_only"
    assert labels[2]["support_count"] == 1
    assert labels[2]["representative_doc_id"] == "d4"
    assert extract_labels([], {}, fake_tokenizer) == []


def test_ctfidf_two_document_fallback_and_person_filter():
    rows = documents(4)
    for row, text in zip(rows, ["김지은 딸기 사과", "딸기 복숭아", "자두 배추", "참외 오이"]):
        row.update(text=text, person="김지은")
    labels = extract_labels(rows, {row["doc_id"]: 0 for row in rows}, fake_tokenizer)
    assert labels[0]["label"] == "딸기"
    assert labels[0]["label_stage"] == "two_documents"
    assert labels[0]["source_doc_ids"] == ["d0", "d1"]


def test_embedding_cache_is_keyed_by_input_model_and_task(tmp_path):
    class Model:
        max_seq_length = 512
        calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return [[1.0, float(len(text))] for text in texts]

        def get_sentence_embedding_dimension(self):
            return 2

    model = Model()
    first = encode_texts(model, ["메모"], 2, tmp_path, "model-a", "clustering")
    np.testing.assert_array_equal(first, encode_texts(model, ["메모"], 2, tmp_path, "model-a", "clustering"))
    assert model.calls == 1
    encode_texts(model, ["메모"], 2, tmp_path, "model-a", "search")
    encode_texts(model, ["새 메모"], 2, tmp_path, "model-a", "clustering")
    encode_texts(model, ["메모"], 2, tmp_path, "model-b", "clustering")
    assert model.calls == 4


def test_error_slices_use_metadata_only_after_prediction():
    rows = documents(2)
    for row in rows:
        row.update(category="소아과", date="2026-03-15", family_id="f" + row["doc_id"], fact_value="병원")
    rows[1]["event_id"] = "other"
    diagnostics = error_diagnostics(rows, {"d0": 0, "d1": 0})
    assert diagnostics["false_merge_slices"]["same_category_and_date"] == 1
    assert "same_family" not in diagnostics["false_merge_slices"]
    assert diagnostics["largest_error_groups"][0]["false_merges"] == 1


def test_geometry_measures_hubs_without_gold_metadata():
    rows = [{"doc_id": f"d{index}"} for index in range(4)]
    values = np.asarray([[1, 0], [0.99, 0.01], [0.98, 0.02], [0, 1]], dtype=np.float32)
    result = geometry_diagnostics(rows, values, k=1)
    assert result["neighbor_k"] == 1
    assert result["indegree_max"] >= 1
    assert result["mean_pair_cosine"] > 0
    assert all(np.isfinite(result["top_pc_variance_ratio"]))
    constant = geometry_diagnostics(rows, np.ones((4, 2)))
    assert constant["top_pc_variance_ratio"] == []
    assert constant["mean_pair_cosine"] == pytest.approx(1)
