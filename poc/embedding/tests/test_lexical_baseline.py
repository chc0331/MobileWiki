import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import issparse

from common import json_sha256, sha256, write_jsonl
from lexical_baseline import (
    feature_texts,
    fit_vectorizer,
    groups_at_threshold,
    lexical_distances,
    run,
    select_development,
)


def rows(split):
    return [
        {"doc_id": f"{split}-{index}", "event_id": f"event-{index // 2}", "split": split,
         "title": "", "text": text}
        for index, text in enumerate(("딸기 수확 밭", "딸기 수확 밭", "바다 물결 배", "바다 물결 배"))
    ]


def test_vocabulary_idf_fit_training_only_and_features_remain_sparse():
    train = rows("train")
    heldout = rows("test")
    heldout[0]["text"] = "오직시험전용표현"
    vectorizer = fit_vectorizer(train)
    before_vocab = dict(vectorizer.vocabulary_)
    before_idf = vectorizer.idf_.copy()
    matrix = vectorizer.transform(feature_texts(heldout))
    assert issparse(matrix)
    assert "시험" not in vectorizer.vocabulary_
    lexical_distances(vectorizer, heldout)
    assert vectorizer.vocabulary_ == before_vocab
    np.testing.assert_array_equal(vectorizer.idf_, before_idf)
    with pytest.raises(ValueError, match="training documents only"):
        fit_vectorizer(heldout)


def test_labels_and_synthetic_metadata_are_not_features():
    original = rows("train")
    changed = copy.deepcopy(original)
    for index, row in enumerate(changed):
        row.update(doc_id=f"other-{index}", event_id="one_event", family_id="different", date="2099-01-01", category="metadata_only", fact_value="hidden_answer")
    assert feature_texts(original) == feature_texts(changed)
    vectorizer = fit_vectorizer(original)
    np.testing.assert_array_equal(lexical_distances(vectorizer, original), lexical_distances(vectorizer, changed))


def test_selection_is_development_only_with_frozen_test_settings():
    vectorizer = fit_vectorizer(rows("train"))
    dev = rows("dev")
    selected, candidates = select_development(dev, lexical_distances(vectorizer, dev))
    assert len(candidates) == 38
    assert selected["f1"] == 1
    assert (selected["linkage"], selected["threshold"], selected["order"]) == ("average", 0.05, 0)
    test = rows("test")
    test[1]["event_id"] = "different_gold"
    groups = groups_at_threshold(test, lexical_distances(vectorizer, test), selected["linkage"], selected["threshold"])
    assert groups["test-0"] == groups["test-1"]
    assert selected["threshold"] == 0.05
    with pytest.raises(ValueError, match="development documents only"):
        select_development(test, lexical_distances(vectorizer, test))


def test_v3_name_masking_is_used_without_changing_original_text():
    train = rows("train")
    for row in train:
        row.update(card_revision="context-hardneg-v3", person="김가은", text="김가은이 딸기를 수확했다.")
    vectorizer = fit_vectorizer(train)
    assert "김가은" not in vectorizer.vocabulary_
    assert "인물" in vectorizer.vocabulary_
    assert train[0]["text"].startswith("김가은")


def test_cli_report_records_hashes_and_rejects_changed_dataset_before_output(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_jsonl(dataset / "documents.jsonl", rows("train") + rows("dev") + rows("test"))
    for name in ("queries.jsonl", "pairs.jsonl", "triplets.jsonl"):
        write_jsonl(dataset / name, [])
    hashes = {name: sha256(dataset / name) for name in ("documents.jsonl", "queries.jsonl", "pairs.jsonl", "triplets.jsonl")}
    (dataset / "manifest.json").write_text(json.dumps({"dataset_revision": "fixture", "files": hashes}))
    output = tmp_path / "baseline.json"
    result = run(SimpleNamespace(dataset=str(dataset), output=str(output)))
    saved = json.loads(output.read_text())
    assert saved["fit_scope"] == "train_only"
    assert saved["fit_rows"] == 4
    assert saved["dataset_files"] == hashes
    assert saved["test_all_event_pairs"]["f1"] == 1
    assert saved["training_input_sha256"] == result["training_input_sha256"]
    (dataset / "documents.jsonl").write_text("changed")
    absent_output = tmp_path / "new" / "rejected.json"
    with pytest.raises(ValueError, match="hash mismatch"):
        run(SimpleNamespace(dataset=str(dataset), output=str(absent_output)))
    assert not absent_output.parent.exists()


def test_explicit_dataset_mask_applies_to_legacy_documents_in_cli(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    documents = rows("train") + rows("dev") + rows("test")
    for document in documents:
        document.update(person="김가은", text="김가은이 " + document["text"])
    write_jsonl(dataset / "documents.jsonl", documents)
    for name in ("queries.jsonl", "pairs.jsonl", "triplets.jsonl"):
        write_jsonl(dataset / name, [])
    hashes = {name: sha256(dataset / name) for name in ("documents.jsonl", "queries.jsonl", "pairs.jsonl", "triplets.jsonl")}
    (dataset / "manifest.json").write_text(json.dumps({"dataset_revision": "legacy", "clustering_deidentify": True, "files": hashes}))
    result = run(SimpleNamespace(dataset=str(dataset), output=str(tmp_path / "baseline.json")))
    train = [document for document in documents if document["split"] == "train"]
    assert result["clustering_deidentify"] is True
    assert result["training_input_sha256"] == json_sha256(feature_texts(train, True))
    assert result["training_input_sha256"] != json_sha256(feature_texts(train, False))
    assert "김가은" not in fit_vectorizer(train, True).vocabulary_
    assert sha256(dataset / "documents.jsonl") == hashes["documents.jsonl"]
