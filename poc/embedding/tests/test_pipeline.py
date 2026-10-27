import json

import pytest

from common import event_card
from curate import curate_row
from evaluate import pair_f1
from generate import parse_response
from prepare import compile_rows


def sample_row(index):
    card = event_card("train", index)
    target = f"{card.fact_label}: {card.fact_value}"
    documents = [
        {
            "title": f"{card.category} {number}",
            "text": (
                f"{card.date} {card.category}을 확인했다. {target}로 정했고 가족에게 다시 전달했다. 첫 기록에 상세 내용을 남긴다."
                if number == 0 else
                f"{card.date} {card.category} 일정의 {number}번째 메모다. 가족에게 필요한 내용을 다시 알려 주고 준비 상태를 확인했다."
            ),
        }
        for number in range(4)
    ]
    return {
        **card.as_dict(),
        "documents": documents,
        "query": f"{card.date}에 정한 {card.fact_label}이 뭐였지?",
        "evidence": target,
        "teacher_id": "qwen3:8b",
        "teacher_revision": "a" * 40,
        "prompt_version": "test",
    }


def test_teacher_response_requires_grounded_answer():
    row = sample_row(0)
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    assert parse_response(response, event_card("train", 0).as_dict())["evidence"] == row["evidence"]
    broken = json.dumps({**json.loads(response), "evidence": "확인했다"}, ensure_ascii=False)
    repaired = parse_response(broken, event_card("train", 0).as_dict())
    assert row["evidence"] in repaired["evidence"]
    assert repaired["repairs"] == ["evidence: copied source sentence"]


def test_answer_sentence_is_removed_from_same_event_distractor():
    row = sample_row(0)
    row["documents"][1]["text"] += f" {event_card('train', 0).fact_value}를 다시 확인했다."
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    parsed = parse_response(response, event_card("train", 0).as_dict())
    assert event_card("train", 0).fact_value not in parsed["documents"][1]["text"]
    assert "d1: removed answer-bearing sentence" in parsed["repairs"]


def test_teacher_response_rejects_duplicate_document_text():
    row = sample_row(0)
    row["documents"][2]["text"] = row["documents"][1]["text"]
    response = json.dumps({key: row[key] for key in ("documents", "query", "evidence")}, ensure_ascii=False)
    with pytest.raises(ValueError, match="Duplicate document text"):
        parse_response(response, event_card("train", 0).as_dict())


def test_curation_removes_partial_answer_clues_from_distractors_and_query():
    row = sample_row(5)
    row["documents"][1]["text"] += " 행사 장소는 강당이야."
    row["documents"][2]["title"] = "강당 준비"
    row["query"] = "강당에서 열리는 학교 행사가 언제지?"
    curated = curate_row(row)
    assert all("강당" not in doc["title"] + doc["text"] for doc in curated["documents"][1:])
    assert "강당" not in curated["query"]
    assert row["documents"][1]["text"].endswith("강당이야.")
    assert "d1: removed answer clue from text" in curated["curation_repairs"]


def test_search_uses_same_event_distractor_and_clustering_uses_other_event():
    rows = compile_rows([sample_row(0), sample_row(10)], allow_partial=True)
    assert len(rows["documents"]) == 8
    assert rows["documents"][0]["title"].startswith(event_card("train", 0).person)
    assert event_card("train", 0).date not in rows["queries"][0]["text"]
    assert [query["query_source"] for query in rows["queries"]] == ["paraphrase_template", "teacher"]
    assert len(rows["triplets"]) == 10
    for triplet in rows["triplets"]:
        same_event = triplet["positive_id"].split("-d")[0] == triplet["negative_id"].split("-d")[0]
        assert same_event == (triplet["task"] == "search")


def test_teacher_prompt_versions_cannot_be_mixed():
    first, second = sample_row(0), sample_row(10)
    second["prompt_version"] = "other"
    with pytest.raises(ValueError, match="prompt version"):
        compile_rows([first, second], allow_partial=True)


def test_search_negative_avoids_partial_answer_words():
    first, second = sample_row(0), sample_row(10)
    first["documents"][1]["text"] += " 푸른숲 일정을 다시 확인했다."
    rows = compile_rows([first, second], allow_partial=True)
    selected = next(item for item in rows["triplets"] if item["task"] == "search" and item["anchor_id"] == "train-0000-q0")
    assert selected["negative_id"] == "train-0000-d2"


def test_pair_f1_counts_false_positive_and_false_negative():
    pairs = [
        {"left_id": "a", "right_id": "b", "related": True},
        {"left_id": "a", "right_id": "c", "related": False},
    ]
    result = pair_f1(pairs, {"a": 1, "b": 1, "c": 1})
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["f1"] == pytest.approx(2 / 3)
