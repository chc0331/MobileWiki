import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter

import pytest

from common import cards, format_event_date, read_jsonl, sha256, write_jsonl
from phase2 import PRESERVED_FILES, REVISION, SOURCE_REVISION, TRAINING_POLICY, build, make_triplets, validate_triplets
from prepare import compile_rows, write_review
from validate import validate_dataset


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("phase2-source")
    raw = []
    for card in cards():
        evidence = f"확인한 내용은 {card.fact_value}로 정했다."
        documents = []
        for slot in range(4):
            digest = hashlib.sha256(f"{card.event_id}/{slot}".encode()).hexdigest()
            distinct = "".join("가나다라마바사아자차카타파하허호"[int(value, 16)] for value in digest)
            person = f"{card.person}은 " if card.identity_style == "named" else ""
            text = f"{format_event_date(card.date, card.document_date_styles[slot])} {person}{card.category} 기록이다. {distinct}."
            if slot == card.answer_doc_index:
                text += " " + evidence
            documents.append({"title": f"생활 메모 {slot}", "text": text})
        query = f"{card.person}은 " if card.identity_style == "named" else ""
        if card.query_date_style != "none":
            query += format_event_date(card.date, card.query_date_style) + " "
        query += f"{card.category}에서 확인한 내용은 무엇이지?"
        raw.append({**card.as_dict(), "documents": documents, "query": query, "evidence": evidence,
                    "teacher_id": "fixture", "teacher_revision": "fixed", "prompt_version": "fixture"})
    rows = compile_rows(raw)
    for name, values in rows.items():
        write_jsonl(folder / f"{name}.jsonl", values)
    write_review(folder / "review.csv", rows)
    manifest = {"dataset_revision": SOURCE_REVISION, "status": "provisional_pending_human_review", "code_files": {},
                "counts": {"documents": {"train": 800, "dev": 300, "test": 1000}, "queries": {"train": 200, "dev": 75, "test": 250}, "pairs": {"dev": 100, "test": 100}},
                "files": {f"{name}.jsonl": sha256(folder / f"{name}.jsonl") for name in rows},
                "review_file_sha256": sha256(folder / "review.csv")}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    validate_dataset(folder)
    return folder


@pytest.fixture(scope="module")
def derived_dataset(source_dataset, tmp_path_factory):
    output = tmp_path_factory.mktemp("phase2-parent") / "derived"
    build(argparse.Namespace(dataset=str(source_dataset), output=str(output)))
    return output


def changed_dataset(derived, output):
    shutil.copytree(derived, output)
    return output


def update_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["files"] = {name: sha256(folder / name) for name in manifest["files"]}
    manifest["review_file_sha256"] = sha256(folder / "review.csv")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_phase2_keeps_original_bytes_and_exact_observed_fact_recipe(source_dataset, derived_dataset):
    documents, queries = read_jsonl(derived_dataset / "documents.jsonl"), read_jsonl(derived_dataset / "queries.jsonl")
    triples = read_jsonl(derived_dataset / "triplets.jsonl")
    assert triples == make_triplets(documents, queries)
    validate_triplets(triples, documents, queries, triples)
    assert Counter(row["negative_type"] for row in triples) == Counter(A=400, B=240, C=160, within_event=200, same_family_other_event=200)
    docs = {row["doc_id"]: row for row in documents}
    c_rows = [row for row in triples if row["negative_type"] == "C"]
    assert Counter(docs[row["anchor_id"]]["category"] for row in c_rows) == Counter({category: 16 for category in {doc["category"] for doc in documents}})
    for row in c_rows:
        for role in ("anchor_id", "negative_id"):
            doc = docs[row[role]]
            assert doc["doc_id"].endswith(f"-d{doc['answer_doc_index']}")
            assert doc["fact_value"] in doc["title"] + doc["text"]
    for name in PRESERVED_FILES:
        assert (derived_dataset / name).read_bytes() == (source_dataset / name).read_bytes()
    assert (derived_dataset / "triplets.source.jsonl").read_bytes() == (source_dataset / "triplets.jsonl").read_bytes()
    assert (derived_dataset / "source.manifest.json").read_bytes() == (source_dataset / "manifest.json").read_bytes()
    manifest = json.loads((derived_dataset / "manifest.json").read_text())
    assert manifest["dataset_revision"] == REVISION and manifest["training_policy"] == TRAINING_POLICY
    assert manifest["clustering_deidentify"] is True
    assert manifest["status"] == manifest["source_status"] == "provisional_pending_human_review"
    assert validate_dataset(derived_dataset)["triplets"] == 1200
    assert validate_dataset(source_dataset)["triplets"] == 1000


def test_phase2_refuses_existing_output_before_modifying_it(source_dataset, derived_dataset):
    before = (derived_dataset / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build(argparse.Namespace(dataset=str(source_dataset), output=str(derived_dataset)))
    assert (derived_dataset / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("name", ["documents.jsonl", "queries.jsonl", "pairs.jsonl", "review.csv", "triplets.source.jsonl", "source.manifest.json"])
def test_phase2_rejects_source_changes_even_if_outer_hashes_are_updated(derived_dataset, tmp_path, name):
    folder = changed_dataset(derived_dataset, tmp_path / "changed")
    with (folder / name).open("a") as stream:
        stream.write("\n")
    update_manifest(folder)
    with pytest.raises(ValueError, match="source bytes changed|backup differs|manifest proof differs"):
        validate_dataset(folder)


@pytest.mark.parametrize("kind", ["A", "B", "C", "within_event", "same_family_other_event"])
def test_phase2_mined_negatives_must_preserve_their_typed_relation(derived_dataset, tmp_path, kind):
    folder = changed_dataset(derived_dataset, tmp_path / "mined")
    original = read_jsonl(folder / "triplets.jsonl")
    write_jsonl(folder / "triplets.initial.jsonl", original)
    triples = copy.deepcopy(original)
    target = next(row for row in triples if row["negative_type"] == kind)
    docs = read_jsonl(folder / "documents.jsonl")
    positive = next(doc for doc in docs if doc["doc_id"] == target["positive_id"])
    def wrong(doc):
        if doc["split"] != "train" or doc["event_id"] == positive["event_id"]:
            return False
        if kind == "A" or kind == "same_family_other_event":
            return doc["family_id"] != positive["family_id"]
        if kind == "within_event":
            return doc["family_id"] == positive["family_id"]
        return doc["family_id"] != positive["family_id"] and doc["category"] == positive["category"] and ((doc["fact_value"] == positive["fact_value"]) == (kind == "B"))
    target["negative_id"] = next(doc["doc_id"] for doc in docs if wrong(doc))
    write_jsonl(folder / "triplets.jsonl", triples)
    manifest = update_manifest(folder)
    manifest["files"]["triplets.initial.jsonl"] = sha256(folder / "triplets.initial.jsonl")
    manifest["mining"] = {"scope": "train_only", "clustering_deidentify": True, "source_triplets_sha256": manifest["source_proof"]["initial_triplets_sha256"]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="search evidence or negative|clustering positive/negative event relation|does not satisfy type"):
        validate_dataset(folder)


def test_phase2_accepts_valid_mining_and_checks_initial_backup(derived_dataset, tmp_path):
    from mine import candidate_documents

    folder = changed_dataset(derived_dataset, tmp_path / "mined")
    triples = read_jsonl(folder / "triplets.jsonl")
    docs = {row["doc_id"]: row for row in read_jsonl(folder / "documents.jsonl")}
    queries = {row["query_id"]: row for row in read_jsonl(folder / "queries.jsonl")}
    write_jsonl(folder / "triplets.initial.jsonl", triples)
    for triple in triples:
        candidates = candidate_documents(triple, docs, queries)
        assert candidates
        triple["negative_id"] = candidates[-1]["doc_id"]
        triple["mining_cosine"] = 0.5
    write_jsonl(folder / "triplets.jsonl", triples)
    manifest = update_manifest(folder)
    manifest["files"]["triplets.initial.jsonl"] = sha256(folder / "triplets.initial.jsonl")
    manifest["mining"] = {"scope": "train_only", "clustering_deidentify": True, "source_triplets_sha256": manifest["source_proof"]["initial_triplets_sha256"]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert validate_dataset(folder)["triplets"] == 1200
    with (folder / "triplets.initial.jsonl").open("a") as stream:
        stream.write("\n")
    update_manifest(folder)
    with pytest.raises(ValueError, match="mining initial triplets differ"):
        validate_dataset(folder)


def test_phase2_detects_anchor_duplicates_and_physical_fact_absence(derived_dataset):
    docs = read_jsonl(derived_dataset / "documents.jsonl")
    queries = read_jsonl(derived_dataset / "queries.jsonl")
    expected = read_jsonl(derived_dataset / "triplets.jsonl")
    triples = copy.deepcopy(expected)
    clustered = [row for row in triples if row["task"] == "clustering"]
    clustered[1]["anchor_id"] = clustered[0]["anchor_id"]
    with pytest.raises(ValueError, match="anchor exactly once"):
        validate_triplets(triples, docs, queries, expected)
    for role in ("anchor_id", "negative_id"):
        changed = copy.deepcopy(docs)
        c_row = next(row for row in expected if row["negative_type"] == "C")
        document = next(doc for doc in changed if doc["doc_id"] == c_row[role])
        document["title"] = document["title"].replace(document["fact_value"], "")
        document["text"] = document["text"].replace(document["fact_value"], "")
        with pytest.raises(ValueError, match="does not satisfy type C"):
            validate_triplets(expected, changed, queries, expected)


@pytest.mark.parametrize("field, message", [("clustering_deidentify", "explicit name masking"), ("training_policy", "training policy"), ("recipe", "recipe file hash"), ("source_status", "status provenance")])
def test_phase2_rejects_changed_policy_and_source_metadata(derived_dataset, tmp_path, field, message):
    folder = changed_dataset(derived_dataset, tmp_path / "changed")
    manifest = json.loads((folder / "manifest.json").read_text())
    if field == "recipe":
        manifest["source_proof"]["recipe_sha256"] = "0" * 64
    else:
        manifest[field] = False if field == "clustering_deidentify" else "changed"
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate_dataset(folder)


def test_phase2_rejects_heldout_references(derived_dataset):
    docs, queries = read_jsonl(derived_dataset / "documents.jsonl"), read_jsonl(derived_dataset / "queries.jsonl")
    expected = read_jsonl(derived_dataset / "triplets.jsonl")
    triples = copy.deepcopy(expected)
    triples[0]["negative_id"] = next(doc["doc_id"] for doc in docs if doc["split"] == "test")
    with pytest.raises(ValueError, match="training rows only"):
        validate_triplets(triples, docs, queries, expected)
