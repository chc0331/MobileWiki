import argparse
import json
from pathlib import Path
import numpy as np
import pytest

from common import code_fingerprint, generation_budget_path, json_sha256, sha256, verified_dataset_manifest, write_jsonl
from mine import candidate_documents, mine, mine_triplets
from run_experiment import choose_candidate
from train import TRAIN_CODE_FILES, lora_targets, train, training_examples
from budget import WallBudget


def document(doc_id, event, family, category="병원", fact="봄빛", day="2026-01-01", split="train"):
    return {"doc_id": doc_id, "event_id": event, "family_id": family, "category": category, "fact_value": fact,
            "date": day, "split": split, "person": "한지은", "title": "예약", "text": "한지은의 병원 예약 메모"}


def write_dataset(folder, revision="shortcut-hardneg-v2-balanced"):
    folder.mkdir(parents=True)
    rows = {
        "documents.jsonl": [document("d0", "e0", "f0"), document("d1", "e0", "f0")],
        "queries.jsonl": [{"query_id": "q", "event_id": "e0", "split": "train", "text": "예약한 병원은?", "relevant_doc_ids": ["d0"]}],
        "pairs.jsonl": [],
        "triplets.jsonl": [{"task": "search", "anchor_id": "q", "positive_id": "d0", "negative_id": "d1"}],
    }
    for name, values in rows.items():
        write_jsonl(folder / name, values)
    manifest = {"dataset_revision": revision, "files": {name: sha256(folder / name) for name in rows}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def training_record(dataset):
    import run_experiment

    project = Path(run_experiment.__file__).parent
    return {
        "status": "trained", "base_revision": "fixed", "dataset_manifest": verified_dataset_manifest(dataset),
        "code_sha256": sha256(project / "train.py"),
        "training_code_fingerprint": json_sha256(code_fingerprint(project, TRAIN_CODE_FILES)),
        "training_input_sha256": json_sha256(training_examples(dataset)), "steps": 300,
        "settings": {"target_modules": ["q_proj", "v_proj"], "triplet_margin": 0.1,
                     "epochs": 2, "max_length": 512, "max_steps": -1, "limit": None},
    }


def test_training_rejects_held_out_references(tmp_path):
    docs = [document("d0", "e0", "f0"), document("d1", "e1", "f1", split="dev")]
    queries = [{"query_id": "q", "split": "train", "text": "한지은의 병원은?"}]
    write_jsonl(tmp_path / "documents.jsonl", docs)
    write_jsonl(tmp_path / "queries.jsonl", queries)
    write_jsonl(tmp_path / "triplets.jsonl", [{"task": "search", "anchor_id": "q", "positive_id": "d0", "negative_id": "d1"}])
    with pytest.raises(ValueError, match="must not include"):
        training_examples(tmp_path)


def test_lora_targets_preserve_exact_attention_modules():
    assert lora_targets("q_proj, k_proj,v_proj,o_proj") == ["q_proj", "k_proj", "v_proj", "o_proj"]
    for value in ("", "q_proj,q_proj", "all-linear"):
        with pytest.raises(argparse.ArgumentTypeError):
            lora_targets(value)


def test_mining_preserves_disjoint_negative_semantics_and_excludes_test():
    rows = [document("a", "e0", "f0"), document("samefamily", "e1", "f0"),
            document("b", "e2", "f1", fact="푸른숲"), document("c", "e3", "f2"),
            document("test", "e4", "f3", fact="푸른숲", split="test")]
    rows[3]["text"] = "봄빛으로 예약했다."
    docs = {row["doc_id"]: row for row in rows}
    base = {"task": "clustering", "anchor_id": "a", "positive_id": "a", "negative_id": "b"}
    for kind, expected in (("A", "samefamily"), ("B", "b"), ("C", "c")):
        pool = candidate_documents({**base, "negative_type": kind}, docs, {})
        assert [row["doc_id"] for row in pool] == [expected]


def test_c_mining_requires_observed_fact_in_negative_and_v3_anchor():
    docs = [document("anchor", "e0", "f0"), document("metadata_only", "e1", "f1"),
            document("bearing", "e1", "f1"), document("wrong_fact", "e2", "f2", fact="푸른숲")]
    docs[0].update(card_revision="context-hardneg-v3", text="봄빛에서 진료를 받는다.")
    docs[2]["title"] = "봄빛 예약 확인"
    docs[3]["text"] = "봄빛 메모와 다른 푸른숲 사건이다."
    vectors = np.array([[1, 0], [1, 0], [0.6, 0.8], [0.9, 0.1]], dtype=np.float32)
    triple = {"task": "clustering", "anchor_id": "anchor", "positive_id": "anchor",
              "negative_id": "metadata_only", "negative_type": "C"}
    selected = mine_triplets(docs, [], [triple], vectors, vectors, np.empty((0, 2)))
    assert selected[0]["negative_id"] == "bearing"
    assert selected[0]["mining_cosine"] == pytest.approx(0.6)
    assert selected[0]["mining_pool_documents"] == 1
    docs[0]["text"] = "진료 전 준비를 확인한다."
    with pytest.raises(ValueError, match="v3 C mining anchor must contain its fact"):
        candidate_documents(triple, {row["doc_id"]: row for row in docs}, {})
    del docs[0]["card_revision"]
    assert [row["doc_id"] for row in candidate_documents(triple, {row["doc_id"]: row for row in docs}, {})] == ["bearing"]


def test_v3_c_anchor_is_checked_before_mining_model_or_budget_output(tmp_path, monkeypatch):
    import mine as mining

    dataset, output = tmp_path / "dataset", tmp_path / "mined"
    manifest = write_dataset(dataset, revision="context-hardneg-v3")
    docs = [document("anchor", "e0", "f0"), document("bearing", "e1", "f1")]
    for row in docs:
        row["card_revision"] = "context-hardneg-v3"
    docs[1]["text"] = "봄빛에서 진료를 받는다."
    triple = {"task": "clustering", "anchor_id": "anchor", "positive_id": "anchor",
              "negative_id": "bearing", "negative_type": "C"}
    write_jsonl(dataset / "documents.jsonl", docs)
    write_jsonl(dataset / "triplets.jsonl", [triple])
    for name in ("documents.jsonl", "triplets.jsonl"):
        manifest["files"][name] = sha256(dataset / name)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (dataset.parent / "generation_budget.json").write_text(json.dumps({"elapsed_seconds": 10}))
    monkeypatch.setattr(mining, "_mine", lambda *args: pytest.fail("An invalid C anchor must fail before model loading"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24)
    with pytest.raises(ValueError, match="v3 C mining anchor must contain its fact"):
        mine(args)
    assert not output.exists()
    assert not (output.parent / "mined.mining_budget.json").exists()


def test_miner_selects_highest_similarity_within_valid_pool():
    docs = [document("a", "e0", "f0"), document("weak", "e1", "f0"), document("hard", "e2", "f0")]
    vectors = np.array([[1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    triples = [{"task": "clustering", "anchor_id": "a", "positive_id": "a", "negative_id": "weak", "negative_type": "A"}]
    selected = mine_triplets(docs, [], triples, vectors, vectors, np.empty((0, 2)))
    assert selected[0]["negative_id"] == "hard"
    assert selected[0]["mining_cosine"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="Only training"):
        mine_triplets([{**docs[0], "split": "test"}], [], [], vectors, vectors, np.empty((0, 2)))


def candidate(name, recall, f1, fp=10, mrr=0.8, order=0):
    return {"name": name, "search_recall_at_5": recall, "all_event_pair_f1": f1,
            "all_event_pair_false_merges": fp, "search_mrr": mrr, "steps": 300, "order": order}


def test_selection_honors_search_constraint_before_cluster_f1():
    chosen = choose_candidate([candidate("bad-search", 0.89, 0.99), candidate("eligible", 0.90, 0.8)])
    assert chosen["name"] == "eligible"
    assert chosen["eligible_search"] is True
    missing = choose_candidate([candidate("low-search", 0.7, 0.9), candidate("closer", 0.88, 0.7)])
    assert missing["name"] == "closer"
    assert missing["selection_status"] == "search_target_missed"


def test_budget_records_failed_attempts_and_rejects_exhausted_resume(tmp_path, monkeypatch):
    import budget

    clock = [100.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: clock[0])
    path = tmp_path / "budget.json"
    with pytest.raises(ValueError, match="failed operation"):
        with WallBudget(path, hours=1):
            clock[0] += 12.5
            raise ValueError("failed operation")
    saved = json.loads(path.read_text())
    assert saved["elapsed_seconds"] == pytest.approx(12.5)
    assert saved["status"] == "interrupted_or_failed"
    with pytest.raises(TimeoutError, match="exhausted"):
        with WallBudget(path, hours=1, prior_seconds=3590):
            pytest.fail("An exhausted budget cannot enter the operation")


def test_experiment_persists_elapsed_when_child_training_fails(tmp_path, monkeypatch):
    import run_experiment
    import subprocess

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset)
    clock = [100.0]
    monkeypatch.setattr(run_experiment.time, "monotonic", lambda: clock[0])
    def failed(command, **kwargs):
        assert kwargs["timeout"] > 0
        clock[0] += 9.5
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(run_experiment.subprocess, "run", failed)
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None,
                              max_hours=24, epochs=2, base_revision="fixed")
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment.run(args)
    saved = json.loads((output / "experiment.json").read_text())
    assert saved["elapsed_seconds"] == pytest.approx(9.5)
    assert saved["status"] == "interrupted_or_failed"
    assert saved["current_stage"] == "training"


def test_experiment_does_not_reuse_a_shorter_training_run(tmp_path, monkeypatch):
    import run_experiment
    from common import sha256
    from pathlib import Path

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset)
    previous = output / "qv-margin-01"
    previous.mkdir(parents=True)
    training = training_record(dataset)
    training["settings"]["epochs"] = 1
    (previous / "training.json").write_text(json.dumps(training))
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *a, **kw: pytest.fail("A mismatched run cannot be reused"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None,
                              max_hours=24, epochs=2, base_revision="fixed")
    with pytest.raises(ValueError, match="different provenance"):
        run_experiment.run(args)


def test_hash_verification_rejects_missing_and_changed_required_files(tmp_path):
    dataset = tmp_path / "dataset"
    manifest = write_dataset(dataset)
    assert verified_dataset_manifest(dataset)["files"] == manifest["files"]
    manifest["files"].pop("pairs.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing required"):
        verified_dataset_manifest(dataset)
    manifest["files"]["pairs.jsonl"] = sha256(dataset / "pairs.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (dataset / "documents.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch: documents"):
        verified_dataset_manifest(dataset)


def test_all_training_entrypoints_reject_corruption_before_loading_or_mutating(tmp_path, monkeypatch):
    import mine as mining
    import run_experiment

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset)
    (dataset / "triplets.jsonl").write_text("changed")
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: pytest.fail("No child may load a model"))
    monkeypatch.setattr(mining, "_mine", lambda *args: pytest.fail("The miner may not load a model"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    for operation in (run_experiment.run, mine, train):
        with pytest.raises(ValueError, match="hash mismatch: triplets"):
            operation(args)
    assert not output.exists()
    assert not (tmp_path / "runs.mining_budget.json").exists()


@pytest.mark.parametrize("corruption", ["changed", "missing_hash"])
def test_evaluation_and_selection_verify_sources_before_reading_or_loading(tmp_path, monkeypatch, corruption):
    import importlib.util
    import evaluate
    import sentence_transformers

    spec = importlib.util.spec_from_file_location("checkpoint_selection", Path(evaluate.__file__).with_name("select.py"))
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)
    dataset = tmp_path / "dataset"
    manifest = write_dataset(dataset)
    if corruption == "changed":
        (dataset / "triplets.jsonl").write_text("changed")
        message = "hash mismatch: triplets"
    else:
        manifest["files"].pop("pairs.jsonl")
        (dataset / "manifest.json").write_text(json.dumps(manifest))
        message = "missing required"
    fail = lambda *args, **kwargs: pytest.fail("Invalid sources must fail before row parsing or model loading")
    monkeypatch.setattr(evaluate, "read_jsonl", fail)
    monkeypatch.setattr(selection, "read_jsonl", fail)
    monkeypatch.setattr(selection, "SentenceTransformer", fail)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)
    args = argparse.Namespace(dataset=str(dataset), run=str(tmp_path / "run"))
    for operation in (evaluate.evaluate, selection.select):
        with pytest.raises(ValueError, match=message):
            operation(args)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("split", ["dev", "test"])
def test_original_search_negative_is_checked_before_mining_or_budget_output(tmp_path, monkeypatch, split):
    import mine as mining

    dataset, output = tmp_path / "dataset", tmp_path / "mined"
    manifest = write_dataset(dataset)
    docs = [document("d0", "e0", "f0"), document("d1", "e0", "f0", split=split)]
    write_jsonl(dataset / "documents.jsonl", docs)
    manifest["files"]["documents.jsonl"] = sha256(dataset / "documents.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(mining, "_mine", lambda *args: pytest.fail("Held-out original negatives must fail before mining"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24)
    with pytest.raises(ValueError, match="search negatives must be training"):
        mine(args)
    assert not output.exists()
    assert not (tmp_path / "mined.mining_budget.json").exists()


def test_mining_registers_and_preserves_exact_original_triplet_backup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evaluate
    import mine as mining
    import sentence_transformers

    dataset, output = tmp_path / "dataset", tmp_path / "mined"
    manifest = write_dataset(dataset)
    triple = {"task": "search", "anchor_id": "q", "positive_id": "d0", "negative_id": "d1"}
    original = (json.dumps(triple, separators=(",", ":")) + "\n").encode()
    (dataset / "triplets.jsonl").write_bytes(original)
    manifest["files"]["triplets.jsonl"] = sha256(dataset / "triplets.jsonl")
    manifest["code_files"] = {}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    model = SimpleNamespace(encode=lambda texts, **kwargs: np.ones((len(texts), 2), dtype=np.float32))
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: model)
    monkeypatch.setattr(evaluate, "encode_documents", lambda model, docs, *args: (np.eye(len(docs), dtype=np.float32), [doc["doc_id"] for doc in docs]))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), adapter=None, base_revision="fixed", batch_size=2)
    budget = SimpleNamespace(elapsed_seconds=1.0, path=tmp_path / "budget.json")
    mining._mine(args, budget, manifest, None)
    mined = verified_dataset_manifest(output)
    assert (output / "triplets.initial.jsonl").read_bytes() == original
    assert mined["files"]["triplets.initial.jsonl"] == mined["mining"]["source_triplets_sha256"]
    (output / "triplets.initial.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch: triplets.initial"):
        verified_dataset_manifest(output)


def test_generation_budget_is_discovered_for_direct_and_copied_mined_data(tmp_path):
    dataset = tmp_path / "source" / "dataset"
    manifest = write_dataset(dataset, revision="context-hardneg-v3")
    budget_path = dataset.parent / "generation_budget.json"
    budget_path.write_text(json.dumps({"elapsed_seconds": 120}))
    assert generation_budget_path(dataset, manifest) == budget_path.resolve()
    mined = tmp_path / "moved" / "mined_dataset"
    mined_manifest = write_dataset(mined, revision="context-hardneg-v3")
    mined_manifest["mining"] = {"generation_budget_path": str(budget_path.resolve()), "source_dataset_path": str(dataset.resolve())}
    assert generation_budget_path(mined, mined_manifest) == budget_path.resolve()
    del mined_manifest["mining"]["generation_budget_path"]
    assert generation_budget_path(mined, mined_manifest) == budget_path.resolve()
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"elapsed_seconds": 200}))
    assert generation_budget_path(mined, mined_manifest, explicit) == explicit.resolve()


def test_v3_missing_generation_budget_fails_before_any_output_or_model(tmp_path, monkeypatch):
    import mine as mining
    import run_experiment

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset, revision="context-hardneg-v3")
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: pytest.fail("No child may start"))
    monkeypatch.setattr(mining, "_mine", lambda *args: pytest.fail("No mining model may load"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    for operation in (run_experiment.run, mine, train):
        with pytest.raises(ValueError, match="requires its generation budget"):
            operation(args)
    assert not output.exists()
    assert not (tmp_path / "runs.mining_budget.json").exists()


def test_runner_automatically_counts_generation_time(tmp_path, monkeypatch):
    import run_experiment
    import subprocess

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset, revision="context-hardneg-v3")
    (dataset.parent / "generation_budget.json").write_text(json.dumps({"elapsed_seconds": 120}))
    def failed(command, **kwargs):
        assert kwargs["timeout"] == pytest.approx(24 * 3600 - 120, abs=1)
        assert command[command.index("--generation-budget") + 1] == str((dataset.parent / "generation_budget.json").resolve())
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(run_experiment.subprocess, "run", failed)
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment.run(args)
    report = json.loads((output / "experiment.json").read_text())
    assert report["prior_generation_and_mining_seconds"] == 120
    assert report["generation_budget_path"] == str((dataset.parent / "generation_budget.json").resolve())


def test_runner_does_not_reuse_training_after_common_only_change(tmp_path, monkeypatch):
    import run_experiment

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset)
    previous = output / "qv-margin-01"
    previous.mkdir(parents=True)
    (previous / "training.json").write_text(json.dumps(training_record(dataset)))
    actual_fingerprint = run_experiment.code_fingerprint
    def changed(project, names):
        values = actual_fingerprint(project, names)
        values["common.py"] = "changed input code"
        return values
    monkeypatch.setattr(run_experiment, "code_fingerprint", changed)
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: pytest.fail("Stale weights cannot be reused"))
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    with pytest.raises(ValueError, match="different provenance"):
        run_experiment.run(args)


def test_runner_reselects_after_clustering_dependency_changes(tmp_path, monkeypatch):
    import run_experiment

    dataset, output = tmp_path / "dataset", tmp_path / "runs"
    write_dataset(dataset)
    previous = output / "qv-margin-01"
    previous.mkdir(parents=True)
    path = previous / "training.json"
    record = training_record(dataset)
    selected = {"checkpoint": "checkpoint-150", "threshold": 0.15, "all_event_pair_f1": 0.85,
                "all_event_pair_false_merges": 10, "search_recall_at_5": 0.92, "search_mrr": 0.8,
                "profile_path": "old-profile.json"}
    record["development_selection"] = {"selected": selected}
    project = Path(run_experiment.__file__).parent
    record["development_selection_code_fingerprint"] = json_sha256(code_fingerprint(project, run_experiment.SELECTION_CODE_FILES))
    path.write_text(json.dumps(record))
    monkeypatch.setattr(run_experiment, "CANDIDATES", run_experiment.CANDIDATES[:1])
    actual_fingerprint = run_experiment.code_fingerprint
    def changed(project, names):
        values = actual_fingerprint(project, names)
        if "clustering.py" in names:
            values["clustering.py"] = "changed cluster code"
        return values
    monkeypatch.setattr(run_experiment, "code_fingerprint", changed)
    calls = []
    def selected_again(command, **kwargs):
        assert Path(command[1]).name == "select.py"
        calls.append(command)
        record["development_selection"] = {"selected": selected}
        path.write_text(json.dumps(record))
    monkeypatch.setattr(run_experiment.subprocess, "run", selected_again)
    args = argparse.Namespace(dataset=str(dataset), output=str(output), generation_budget=None, max_hours=24, epochs=2, base_revision="fixed")
    run_experiment.run(args)
    assert len(calls) == 1
    saved = json.loads(path.read_text())
    assert saved["development_selection_code_files"]["clustering.py"] == "changed cluster code"
