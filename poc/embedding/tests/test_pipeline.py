import json
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pytest

from common import (
    DATE_PATTERN,
    RELATIVE_DATE_PATTERN,
    cards,
    event_card,
    format_event_date,
    normalize_event_dates,
    normalize_relative_dates,
    remove_event_dates,
)
from curate import curate_row
from evaluate import all_event_pair_f1, pair_f1, select_cluster_threshold, shortcut_baselines
from generate import parse_response
from prepare import compile_rows
from repair_duplicates import duplicate_events, normalized_texts


def sample_row(index):
    card = event_card("train", index)
    target = f"{card.fact_label}: {card.fact_value}로 정했다."
    documents = []
    for number in range(4):
        body = f"{format_event_date(card.date, card.document_date_styles[number])} {card.category} 일정을 확인했다."
        if card.identity_style == "named":
            body += f" {card.person}과 내용을 다시 확인했다."
        if number == card.answer_doc_index:
            body += f" {target} 가족에게 준비 내용을 전하고 변경 사항을 기록했다."
        else:
            body += f" {number + 1}번째 메모로 준비 상태를 점검하고 가족에게 일정 변경을 알리기로 했다."
        documents.append({"title": f"{card.category} 메모 {number + 1}", "text": body})

    question = (
        f"{card.person}은 {card.category}에서 정한 {card.fact_label}이 무엇이었지?"
        if card.identity_style == "named"
        else f"{card.category}에서 정한 {card.fact_label}이 무엇이었지?"
    )
    if card.query_date_style != "none":
        question = f"{format_event_date(card.date, card.query_date_style)}에 {question}"
    return {
        **card.as_dict(),
        "documents": documents,
        "query": question,
        "evidence": target,
        "teacher_id": "qwen3:8b",
        "teacher_revision": "a" * 40,
        "prompt_version": "test",
    }


def test_assignments_are_balanced_by_split_and_category():
    for split in ("train", "dev", "test"):
        grouped = defaultdict(list)
        for card in cards(split):
            grouped[card.category].append(card)
        for category, members in grouped.items():
            answer_counts = Counter(card.answer_doc_index for card in members)
            identity_counts = Counter(card.identity_style for card in members)
            assert max(answer_counts.values()) - min(answer_counts.values()) <= 1
            assert max(identity_counts.values()) - min(identity_counts.values()) <= 1


@pytest.mark.parametrize(
    ("text", "style", "expected"),
    [
        ("예약은 2026 년 3 월 13 일이다.", "korean_full", "예약은 2026년 3월 13일이다."),
        ("예약은 3 월 13 일이다.", "korean_short", "예약은 3월 13일이다."),
        ("예약은 2026-3-13이다.", "iso", "예약은 2026-03-13이다."),
    ],
)
def test_date_spacing_and_single_digit_iso_preserve_the_same_date(text, style, expected):
    assert normalize_event_dates(text, "2026-03-13", style) == (expected, 1)
    with pytest.raises(ValueError, match="Unexpected date"):
        normalize_event_dates("2026 년 3 월 14 일", "2026-03-13", style)


def test_date_normalization_preserves_only_the_canonical_event_date():
    card = event_card("train", 0)
    formatted = format_event_date(card.date, "korean_full")
    normalized, count = normalize_event_dates(f"메모 날짜는 {card.date}입니다.", card.date, "korean_full")
    assert normalized == f"메모 날짜는 {formatted}입니다."
    assert count == 1
    with pytest.raises(ValueError, match="Unexpected date"):
        normalize_event_dates("2024년 3월 15일", card.date, "korean_short")
    assert RELATIVE_DATE_PATTERN.search("이번 주 토요일 예약")
    assert RELATIVE_DATE_PATTERN.search("3월 중순에 다시 확인")
    replaced, count = normalize_relative_dates("이번 주 토요일에 가기로 했다. 3월 중순에 다시 확인한다.", card.date, "korean_short")
    assert replaced == f"{format_event_date(card.date, 'korean_short')}에 가기로 했다. {format_event_date(card.date, 'korean_short')}에 다시 확인한다."
    assert count == 2
    removed, count = normalize_relative_dates("내일 병원 일정을 확인한다.", card.date, None)
    assert removed == "병원 일정을 확인한다."
    assert count == 1
    removed, count = remove_event_dates(f"{formatted}에 병원 예약을 확인한다.", card.date)
    assert removed == "병원 예약을 확인한다."
    assert count == 1
    with pytest.raises(ValueError, match="Unexpected date"):
        remove_event_dates("2024년 3월 15일에 병원에 간다.", card.date)


def test_teacher_response_requires_grounded_answer_at_assigned_location():
    row = sample_row(0)
    card = event_card("train", 0)
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    parsed = parse_response(response, card.as_dict())
    assert row["evidence"] in parsed["evidence"]
    assert card.fact_value in parsed["documents"][card.answer_doc_index]["text"]
    assert all(card.fact_value not in parsed["documents"][index]["text"] for index in range(4) if index != card.answer_doc_index)


def test_answer_in_wrong_document_is_moved_to_assigned_document():
    index = next(index for index in range(200) if event_card("train", index).answer_doc_index != 0)
    row = sample_row(index)
    card = event_card("train", index)
    source = card.answer_doc_index
    row["documents"][source], row["documents"][0] = row["documents"][0], row["documents"][source]
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    parsed = parse_response(response, card.as_dict())
    assert card.fact_value in parsed["documents"][card.answer_doc_index]["text"]
    assert parsed["evidence"] in parsed["documents"][card.answer_doc_index]["text"]
    assert all(card.fact_value not in parsed["documents"][slot]["text"] for slot in range(4) if slot != card.answer_doc_index)


def test_teacher_response_rejects_implicit_person_leakage():
    index = next(index for index in range(200) if event_card("train", index).identity_style == "implicit")
    row = sample_row(index)
    card = event_card("train", index)
    row["query"] += f" {card.person} 메모였지?"
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    with pytest.raises(ValueError, match="implicit query"):
        parse_response(response, card.as_dict())


def test_teacher_response_removes_an_unassigned_absolute_query_date():
    index = next(index for index in range(200) if event_card("train", index).query_date_style == "none")
    row = sample_row(index)
    card = event_card("train", index)
    row["query"] = f"{card.date}에 {row['query']}"
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    parsed = parse_response(response, card.as_dict())
    assert not DATE_PATTERN.search(parsed["query"])
    assert "query: removed unassigned absolute date" in parsed["repairs"]


def test_teacher_response_adds_name_when_named_event_text_omits_it():
    index = next(index for index in range(200) if event_card("train", index).identity_style == "named")
    row = sample_row(index)
    card = event_card("train", index)
    row["documents"] = [
        {"title": item["title"], "text": item["text"].replace(card.person, "")}
        for item in row["documents"]
    ]
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    parsed = parse_response(response, card.as_dict())
    assert card.person in " ".join(document["text"] for document in parsed["documents"])
    assert any("added assigned fictional name" in item for item in parsed["repairs"])


def test_teacher_response_rejects_unmodeled_relationship_aliases():
    index = next(index for index in range(200) if event_card("train", index).identity_style == "implicit")
    row = sample_row(index)
    card = event_card("train", index)
    row["query"] += " 남편은 어떻게 했지?"
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    with pytest.raises(ValueError, match="Identity shortcut"):
        parse_response(response, card.as_dict())


def test_teacher_response_rejects_duplicate_document_text():
    row = sample_row(0)
    card = event_card("train", 0)
    distractors = [slot for slot in range(4) if slot != card.answer_doc_index]
    row["documents"][distractors[1]]["text"] = row["documents"][distractors[0]]["text"]
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    with pytest.raises(ValueError, match="Duplicate document text"):
        parse_response(response, card.as_dict())


def test_duplicate_repair_selects_later_event_once():
    first, second = sample_row(0), sample_row(20)
    second["documents"][0]["text"] = first["documents"][0]["text"]
    second["documents"][1]["text"] = first["documents"][1]["text"]
    assert duplicate_events([first, second]) == [second["event_id"]]
    assert len(normalized_texts({"documents": first["documents"]}, event_card("train", 0))) == 4


def test_duplicate_repair_accounts_for_date_inserted_during_curation():
    first, second = sample_row(0), sample_row(20)
    body = "필요한 부품은 이미 준비 완료 상태다. 작업 전에 다시 한 번 점검해야 할 사항이 있다."
    first["documents"][0]["text"] = body
    second_card = event_card("train", 20)
    second["documents"][0]["text"] = (
        f"{format_event_date(second_card.date, second_card.document_date_styles[0])} {body}"
    )
    assert duplicate_events([first, second]) == [second["event_id"]]


def test_curation_removes_answer_cues_without_changing_answer_position():
    index = next(index for index in range(200) if event_card("train", index).answer_doc_index != 1)
    row = sample_row(index)
    card = event_card("train", index)
    distractor = next(slot for slot in range(4) if slot != card.answer_doc_index)
    cue = card.fact_value.split()[0]
    row["documents"][distractor]["text"] += f" {cue}를 다시 확인했다."
    row["documents"][distractor]["title"] += f" {cue}"
    row["query"] += f" {cue}였지?"
    curated = curate_row(row)
    assert card.fact_value in curated["documents"][card.answer_doc_index]["text"]
    assert curated["evidence"] in curated["documents"][card.answer_doc_index]["text"]
    assert all(cue not in doc["title"] + doc["text"] for slot, doc in enumerate(curated["documents"]) if slot != card.answer_doc_index)
    assert cue not in curated["query"]
    assert "query: replaced answer clue with paraphrase" in curated["curation_repairs"]


def test_curation_can_complete_a_short_d0_distractor():
    index = next(index for index in range(200) if event_card("train", index).answer_doc_index != 0)
    row = sample_row(index)
    row["documents"][0]["text"] = "메모"
    curated = curate_row(row)
    assert len(curated["documents"][0]["text"]) >= 40
    assert event_card("train", index).fact_value not in curated["documents"][0]["text"]


def test_curation_restores_date_if_answer_cue_removal_removed_it():
    index = next(index for index in range(200) if event_card("train", index).answer_doc_index != 1)
    row = sample_row(index)
    card = event_card("train", index)
    style = card.document_date_styles[1]
    row["documents"][1]["text"] = f"{format_event_date(card.date, style)} {card.fact_value}를 다시 확인했다."
    curated = curate_row(row)
    assert format_event_date(card.date, style) in curated["documents"][1]["text"]


def test_compile_uses_dynamic_evidence_and_non_d0_pairs():
    rows = compile_rows([sample_row(0), sample_row(20)], allow_partial=True)
    assert len(rows["documents"]) == 8
    for query in rows["queries"]:
        card = event_card("train", int(query["event_id"].split("-")[-1]))
        assert query["evidence_doc_id"] == f"{card.event_id}-d{card.answer_doc_index}"
        assert query["relevant_doc_ids"] == [query["evidence_doc_id"]]
    assert len(rows["triplets"]) == 10
    for triplet in rows["triplets"]:
        same_event = triplet["positive_id"].rsplit("-d", 1)[0] == triplet["negative_id"].rsplit("-d", 1)[0]
        assert same_event == (triplet["task"] == "search")


def test_full_train_triplets_have_exact_hard_negative_mix():
    rows = compile_rows([sample_row(index) for index in range(200)], allow_partial=True)
    cluster = [item for item in rows["triplets"] if item["task"] == "clustering"]
    assert len(cluster) == 800
    assert Counter(item["negative_type"] for item in cluster) == Counter({"A": 400, "B": 240, "C": 160})
    docs = {document["doc_id"]: document for document in rows["documents"]}
    for triplet in cluster:
        anchor = docs[triplet["anchor_id"]]
        negative = docs[triplet["negative_id"]]
        if triplet["negative_type"] == "A":
            assert anchor["family_id"] == negative["family_id"]
        elif triplet["negative_type"] == "B":
            assert anchor["category"] == negative["category"]
            assert anchor["family_id"] != negative["family_id"]
        else:
            close_date = abs((date.fromisoformat(anchor["date"]) - date.fromisoformat(negative["date"])).days) <= 30
            assert anchor["family_id"] != negative["family_id"]
            assert anchor["category"] != negative["category"]
            assert close_date or anchor["fact_value"] == negative["fact_value"]


def test_pair_f1_counts_false_positive_and_false_negative():
    pairs = [
        {"left_id": "a", "right_id": "b", "related": True},
        {"left_id": "a", "right_id": "c", "related": False},
    ]
    result = pair_f1(pairs, {"a": 1, "b": 1, "c": 1})
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_shortcut_baselines_handle_named_and_implicit_queries():
    docs, queries = [], []
    for index in range(4):
        card = event_card("test", index)
        doc_ids = [f"{card.event_id}-d{slot}" for slot in range(4)]
        docs.extend(
            {"doc_id": doc_id, "person": card.person, "event_id": card.event_id}
            for doc_id in doc_ids
        )
        queries.append({
            "query_id": f"{card.event_id}-q0",
            "identity_style": card.identity_style,
            "text": f"{card.person} 질문" if card.identity_style == "named" else "주어 없는 질문",
            "relevant_doc_ids": [f"{card.event_id}-d{card.answer_doc_index}"],
        })
    baselines = shortcut_baselines(docs, queries)
    assert baselines["name_only_named_queries"]["total"] == sum(query["identity_style"] == "named" for query in queries)
    assert baselines["d0_all_queries"]["total"] == len(queries)


def test_all_event_pairs_exposes_overmerged_groups():
    docs = [
        {"doc_id": "a0", "event_id": "a"}, {"doc_id": "a1", "event_id": "a"},
        {"doc_id": "b0", "event_id": "b"}, {"doc_id": "b1", "event_id": "b"},
    ]
    result = all_event_pair_f1(docs, {"a0": 0, "a1": 0, "b0": 0, "b1": 0})
    assert result["tp"] == 2
    assert result["fp"] == 4
    assert result["fn"] == 0
    assert result["f1"] == pytest.approx(0.5)


def test_cluster_threshold_uses_all_event_pairs_and_tiebreaks_toward_fewer_merges():
    docs = [
        {"doc_id": "a0", "event_id": "a"}, {"doc_id": "a1", "event_id": "a"},
        {"doc_id": "b0", "event_id": "b"}, {"doc_id": "b1", "event_id": "b"},
    ]
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    selected, candidates = select_cluster_threshold(docs, vectors, (0.05, 0.99))
    assert [candidate["f1"] for candidate in candidates] == pytest.approx([1.0, 0.5])
    assert selected["threshold"] == 0.05
    assert selected["fp"] == 0
