import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common import code_fingerprint, dataset_clustering_deidentify, json_sha256, sha256, write_jsonl
from train import TRAIN_CODE_FILES, training_examples


def dataset(folder, revision="shortcut-hardneg-v2-balanced", policy=None):
    folder.mkdir()
    documents = [
        {"doc_id": f"d{index}", "event_id": "e0" if index < 2 else "e1", "family_id": "f0",
         "category": "병원", "fact_value": "봄빛", "date": "2026-01-01", "split": "train",
         "person": "한지은", "card_revision": "shortcut-hardneg-v2-balanced",
         "title": f"한지은의 예약 {index}", "text": f"한지은의 병원 예약 메모 {index}"}
        for index in range(3)
    ]
    rows = {
        "documents.jsonl": documents,
        "queries.jsonl": [{"query_id": "q", "event_id": "e0", "split": "train", "text": "한지은의 예약한 병원은?", "relevant_doc_ids": ["d0"]}],
        "pairs.jsonl": [],
        "triplets.jsonl": [
            {"task": "search", "anchor_id": "q", "positive_id": "d0", "negative_id": "d1"},
            {"task": "clustering", "anchor_id": "d0", "positive_id": "d1", "negative_id": "d2", "negative_type": "A"},
        ],
    }
    for name, values in rows.items():
        write_jsonl(folder / name, values)
    manifest = {"dataset_revision": revision, "code_files": {}, "files": {name: sha256(folder / name) for name in rows}}
    if policy is not None:
        manifest["clustering_deidentify"] = policy
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(("manifest", "expected"), [
    ({}, False), ({"dataset_revision": "shortcut-hardneg-v2-balanced"}, False),
    ({"dataset_revision": "context-hardneg-v3"}, True),
    ({"dataset_revision": "shortcut-hardneg-v2-cluster-phase2", "clustering_deidentify": True}, True),
    ({"dataset_revision": "context-hardneg-v3", "clustering_deidentify": False}, False),
])
def test_manifest_mask_policy_keeps_legacy_defaults_and_explicit_boolean(manifest, expected):
    assert dataset_clustering_deidentify(manifest) is expected


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_manifest_mask_policy_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        dataset_clustering_deidentify({"clustering_deidentify": value})


@pytest.mark.parametrize(("revision", "policy", "expected"), [
    ("shortcut-hardneg-v2-balanced", None, False), ("context-hardneg-v3", None, True),
    ("shortcut-hardneg-v2-cluster-phase2", True, True), ("context-hardneg-v3", False, False),
])
def test_training_policy_masks_cluster_triplets_and_preserves_search(tmp_path, revision, policy, expected):
    folder = tmp_path / "dataset"
    dataset(folder, revision, policy)
    search, cluster = training_examples(folder)
    assert all("한지은" in text for text in search.values())
    assert all(("[인물]" in text) is expected for text in cluster.values())
    assert all(("한지은" not in text) is expected for text in cluster.values())


def test_fixture_without_manifest_retains_per_document_mask_default(tmp_path):
    folder = tmp_path / "dataset"
    dataset(folder)
    (folder / "manifest.json").unlink()
    assert "한지은" in training_examples(folder)[1]["anchor"]
    rows = [json.loads(line) for line in (folder / "documents.jsonl").read_text().splitlines()]
    for row in rows:
        row["card_revision"] = "context-hardneg-v3"
    write_jsonl(folder / "documents.jsonl", rows)
    assert "한지은" not in training_examples(folder)[1]["anchor"]


@pytest.mark.parametrize("policy", [False, True])
def test_mining_uses_training_mask_policy_and_keeps_search_raw(tmp_path, monkeypatch, policy):
    import mine
    import sentence_transformers

    folder, output = tmp_path / "dataset", tmp_path / "mined"
    manifest = dataset(folder, "shortcut-hardneg-v2-cluster-phase2", policy)
    encoded = []
    def encode(texts, **kwargs):
        encoded.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)
    model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]), encode=encode)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: model)
    args = argparse.Namespace(dataset=str(folder), output=str(output), adapter=None, base_revision="fixed", batch_size=2)
    mine._mine(args, SimpleNamespace(elapsed_seconds=1, path=tmp_path / "budget.json"), manifest, None)
    assert all(("[인물]" in text) is policy for text in encoded[0])
    assert all("한지은" in text for batch in encoded[1:] for text in batch)
    assert json.loads((output / "manifest.json").read_text())["mining"]["clustering_deidentify"] is policy


def selection_module():
    spec = importlib.util.spec_from_file_location("mask_policy_selection", Path(__file__).parents[1] / "select.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("revision", "policy", "option", "expected"), [
    ("shortcut-hardneg-v2-balanced", None, "auto", False), ("context-hardneg-v3", None, "auto", True),
    ("shortcut-hardneg-v2-cluster-phase2", True, "auto", True), ("context-hardneg-v3", False, "auto", False),
    ("shortcut-hardneg-v2-balanced", None, "on", True), ("context-hardneg-v3", True, "off", False),
])
def test_selection_and_evaluation_share_auto_policy_and_explicit_overrides(tmp_path, monkeypatch, revision, policy, option, expected):
    import evaluate
    import sentence_transformers

    folder = tmp_path / "dataset"
    dataset(folder, revision, policy)
    selection = selection_module()
    docs = [{"doc_id": "dev-d0", "split": "dev", "event_id": "dev-e0", "title": "한지은", "text": "메모", "person": "한지은"}]
    queries = [{"split": "dev"}]
    pairs = [{"split": "dev"}]
    for module in (selection, evaluate):
        monkeypatch.setattr(module, "read_jsonl", lambda path: docs if path.name == "documents.jsonl" else queries if path.name == "queries.jsonl" else pairs)
    run = tmp_path / "run"
    (run / "checkpoints/checkpoint-1").mkdir(parents=True)
    (run / "training.json").write_text(json.dumps({"base_revision": "fixed", "settings": {"max_length": 512}}))
    model = SimpleNamespace(load_adapter=lambda path: None)
    monkeypatch.setattr(selection, "SentenceTransformer", lambda *args, **kwargs: model)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: model)
    class EncodedPolicy(Exception):
        pass
    def encoded(*args):
        assert args[-1] is expected
        raise EncodedPolicy
    monkeypatch.setattr(selection, "encode_documents", encoded)
    monkeypatch.setattr(evaluate, "encode_documents", encoded)
    common = dict(dataset=str(folder), deidentify=option, batch_size=2, cache_dir=None)
    with pytest.raises(EncodedPolicy):
        selection.select(argparse.Namespace(**common, run=str(run), hybrid=False))
    with pytest.raises(EncodedPolicy):
        evaluate.evaluate(argparse.Namespace(**common, adapter=None, base_revision="fixed", output=str(tmp_path / "eval.json"), cluster_profile=None, hybrid=False))


def test_mask_change_prevents_reusing_the_previous_training_input_hash(tmp_path, monkeypatch):
    import run_experiment

    folder, output = tmp_path / "dataset", tmp_path / "runs"
    manifest = dataset(folder, "shortcut-hardneg-v2-cluster-phase2", False)
    original_hash = json_sha256(training_examples(folder))
    project = Path(run_experiment.__file__).parent
    previous = output / "qv-margin-01"
    previous.mkdir(parents=True)
    record = {"status": "trained", "base_revision": "fixed", "dataset_manifest": manifest,
              "settings": {"target_modules": ["q_proj", "v_proj"], "triplet_margin": 0.1, "epochs": 2,
                           "max_length": 512, "max_steps": -1, "limit": None},
              "code_sha256": sha256(project / "train.py"),
              "training_code_fingerprint": json_sha256(code_fingerprint(project, TRAIN_CODE_FILES)),
              "training_input_sha256": original_hash}
    (previous / "training.json").write_text(json.dumps(record))
    manifest["clustering_deidentify"] = True
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert json_sha256(training_examples(folder)) != original_hash
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: pytest.fail("Stale weights cannot be reused or retrained implicitly"))
    args = argparse.Namespace(dataset=str(folder), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    with pytest.raises(ValueError, match="different provenance"):
        run_experiment.run(args)


@pytest.mark.parametrize("option", ["auto", "off"])
def test_frozen_profile_mask_policy_remains_authoritative(tmp_path, monkeypatch, option):
    import evaluate
    import sentence_transformers

    folder = tmp_path / "dataset"
    dataset(folder, policy=False)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(evaluate, "load_profile", lambda path: {"provenance": {"clustering_deidentify": True}})
    class EncodedPolicy(Exception):
        pass
    def encoded(*args):
        assert args[-1] is True
        raise EncodedPolicy
    monkeypatch.setattr(evaluate, "encode_documents", encoded)
    args = argparse.Namespace(dataset=str(folder), deidentify=option, adapter=None, base_revision="fixed",
                              output=str(tmp_path / "eval.json"), cluster_profile="frozen.json", batch_size=2)
    if option == "auto":
        with pytest.raises(EncodedPolicy):
            evaluate.evaluate(args)
    else:
        with pytest.raises(ValueError, match="differs from frozen cluster profile"):
            evaluate.evaluate(args)


def test_training_records_the_policy_used_for_actual_cluster_inputs(tmp_path, monkeypatch):
    import datasets
    import peft
    import sentence_transformers
    import sentence_transformers.sentence_transformer.losses as losses
    import torch
    import train

    folder, output = tmp_path / "dataset", tmp_path / "trained"
    dataset(folder, "shortcut-hardneg-v2-cluster-phase2", True)
    observed = []
    monkeypatch.setattr(datasets.Dataset, "from_list", lambda rows: observed.extend(rows) or rows)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=True, numel=lambda: 1)],
                            add_adapter=lambda adapter: None, save_pretrained=lambda path: Path(path).mkdir())
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: model)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformerTrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(sentence_transformers, "SentenceTransformerTrainer", lambda **kwargs: SimpleNamespace(train=lambda: SimpleNamespace(global_step=1, training_loss=0)))
    monkeypatch.setattr(peft, "LoraConfig", lambda **kwargs: None)
    monkeypatch.setattr(losses, "TripletLoss", lambda *args, **kwargs: None)
    args = argparse.Namespace(dataset=str(folder), output=str(output), generation_budget=None, limit=None,
                              base_revision="fixed", max_length=512, lora_targets=["q_proj", "v_proj"],
                              triplet_margin=0.1, epochs=2, max_steps=-1, max_hours=1)
    train.train(args)
    assert "[인물]" in observed[1]["anchor"] and "한지은" in observed[0]["anchor"]
    assert json.loads((output / "training.json").read_text())["settings"]["clustering_deidentify"] is True
