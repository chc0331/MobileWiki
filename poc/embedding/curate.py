"""Remove answer-bearing clues from synthetic search distractors."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from common import (
    DATE_PATTERN,
    PARAPHRASE_QUERIES,
    RELATIVE_DATE_PATTERN,
    RELATIONSHIP_PATTERN,
    event_card,
    format_event_date,
    normalize_event_dates,
    normalize_relative_dates,
    remove_event_dates,
    read_jsonl,
    sha256,
    write_jsonl,
)

CURATION_VERSION = "answer-cues-v2-balanced-shortcut"

# A clue is sufficient to identify the requested fact without the evidence note.
# Generic event words such as "소아과" and "회의" deliberately stay in distractors.
ANSWER_CUES = {
    "푸른숲 소아과": ("푸른숲",),
    "햇살 소아과": ("햇살",),
    "봄빛 소아과": ("봄빛",),
    "실내화": ("실내화",),
    "여벌 양말": ("양말",),
    "그림책": ("그림책",),
    "바람길 펜션": ("바람길",),
    "하늘마루 호텔": ("하늘마루",),
    "파도빛 게스트하우스": ("파도빛",),
    "주간 보고서": ("주간 보고서",),
    "고객 문의 목록": ("고객 문의 목록", "문의 목록"),
    "회의 자료": ("회의 자료",),
    "서쪽 보건소": ("서쪽",),
    "중앙 의원": ("중앙",),
    "동네 보건지소": ("동네 보건지소",),
    "학교 강당": ("강당",),
    "운동장": ("운동장",),
    "도서관": ("도서관",),
    "와이퍼 교체": ("와이퍼",),
    "타이어 점검": ("타이어",),
    "엔진오일 교환": ("엔진오일", "오일 교환"),
    "동네 도서관": ("도서관",),
    "하천 산책로": ("산책로", "하천"),
    "시민 공원": ("공원",),
    "주방 수도꼭지": ("수도꼭지",),
    "현관 조명": ("조명",),
    "욕실 환풍기": ("환풍기",),
    "예산 초안": ("예산 초안",),
    "발표 슬라이드": ("슬라이드",),
    "회의록": ("회의록",),
}

FALLBACKS = (
    "",
    "필요한 준비를 정리하고, 일정에 맞춰 다시 확인하기로 했다.",
    "가족에게 일정을 알려 주고, 바뀌는 내용이 있으면 다시 전하기로 했다.",
    "진행 상황을 나중에 확인하고, 변경 사항이 있으면 메모를 고치기로 했다.",
)


def answer_cues(fact_value: str) -> tuple[str, ...]:
    return ANSWER_CUES[fact_value]


def curate_row(source: dict) -> dict:
    if source.get("card_revision") == "context-hardneg-v3":
        from data_v3 import curate_row as curate_v3_row
        return curate_v3_row(source)
    row = copy.deepcopy(source)
    split, index = row["event_id"].split("-")
    card = event_card(split, int(index))
    if any(row[key] != value for key, value in card.as_dict().items()):
        raise ValueError(f"Card mismatch: {row['event_id']}")
    cues = answer_cues(card.fact_value)
    repairs = []
    answer_index = card.answer_doc_index
    for number, document in enumerate(row["documents"]):
        style = card.document_date_styles[number]
        document["title"], title_relative_dates = normalize_relative_dates(
            document["title"], card.date, style
        )
        document["text"], text_relative_dates = normalize_relative_dates(
            document["text"], card.date, style
        )
        if title_relative_dates or text_relative_dates:
            repairs.append(f"d{number}: replaced relative date with exact event date")
        document["title"], _ = normalize_event_dates(document["title"], card.date, style)
        document["text"], date_count = normalize_event_dates(document["text"], card.date, style)
        if not date_count:
            document["text"] = f"{format_event_date(card.date, style)} {document['text']}"
            repairs.append(f"d{number}: restored event date")
        if number == answer_index:
            continue
        if any(cue in document["title"] for cue in cues):
            document["title"] = f"{card.category} 기록 {number}"
            repairs.append(f"d{number}: removed answer clue from title")
        sentences = re.split(r"(?<=[.!?])\s+", document["text"])
        kept = [sentence for sentence in sentences if not any(cue in sentence for cue in cues)]
        if len(kept) != len(sentences):
            document["text"] = " ".join(kept).strip()
            repairs.append(f"d{number}: removed answer clue from text")
        if len(document["text"]) < 40:
            fallback = FALLBACKS[number] or FALLBACKS[1]
            document["text"] = f"{document['text']} {fallback}".strip()
            repairs.append(f"d{number}: completed short distractor")
        document["text"], date_count = normalize_event_dates(document["text"], card.date, style)
        if not date_count:
            document["text"] = f"{format_event_date(card.date, style)} {document['text']}"
            repairs.append(f"d{number}: restored event date after clue removal")
        if RELATIVE_DATE_PATTERN.search(document["title"] + document["text"]):
            raise ValueError(f"Relative or approximate date found: {row['event_id']}-d{number}")
        if not 40 <= len(document["text"]) <= 500:
            raise ValueError(f"Invalid curated document length: {row['event_id']}-d{number}")
        if any(cue in document["title"] + document["text"] for cue in cues):
            raise ValueError(f"Answer clue remains: {row['event_id']}-d{number}")
    row["evidence"] = next(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", row["documents"][answer_index]["text"])
        if card.fact_value in sentence
    )
    if any(cue in row["query"] for cue in cues):
        row["query"] = PARAPHRASE_QUERIES[card.category]
        if card.identity_style == "named":
            row["query"] = f"{card.person}에 관해 {row['query']}"
        repairs.append("query: replaced answer clue with paraphrase")
    row["query"], relative_dates = normalize_relative_dates(
        row["query"], card.date, card.query_date_style if card.query_date_style != "none" else None
    )
    if relative_dates:
        repairs.append(
            "query: removed relative date"
            if card.query_date_style == "none"
            else "query: replaced relative date with exact event date"
        )
    if RELATIVE_DATE_PATTERN.search(row["query"]):
        raise ValueError(f"Relative or approximate date found in query: {row['event_id']}")
    if card.query_date_style == "none":
        row["query"], removed_dates = remove_event_dates(row["query"], card.date)
        if removed_dates:
            repairs.append("query: removed unassigned absolute date")
        if DATE_PATTERN.search(row["query"]):
            raise ValueError(f"Unexpected query date: {row['event_id']}")
    else:
        row["query"], date_count = normalize_event_dates(row["query"], card.date, card.query_date_style)
        if not date_count:
            row["query"] = f"{format_event_date(card.date, card.query_date_style)}에 {row['query']}"
    if card.identity_style == "implicit" and (
        card.person in row["query"] or RELATIONSHIP_PATTERN.search(row["query"])
    ):
        raise ValueError(f"Identity shortcut in implicit query: {row['event_id']}")
    if card.identity_style == "named" and card.person not in row["query"]:
        raise ValueError(f"Missing person name in named query: {row['event_id']}")
    if any(card.person in doc["title"] for doc in row["documents"]):
        raise ValueError(f"Person name in title: {row['event_id']}")
    if card.identity_style == "implicit" and (
        card.person in " ".join(doc["text"] for doc in row["documents"])
        or RELATIONSHIP_PATTERN.search(" ".join(doc["title"] + " " + doc["text"] for doc in row["documents"]))
    ):
        raise ValueError(f"Identity shortcut in implicit event: {row['event_id']}")
    texts = [" ".join(item["text"].split()) for item in row["documents"]]
    if len(set(texts)) != 4:
        raise ValueError(f"Duplicate curated document text: {row['event_id']}")
    row["curation_version"] = CURATION_VERSION
    row["curation_repairs"] = repairs
    return row


def curate(input_path: Path, output_path: Path, manifest_path: Path) -> None:
    if input_path == output_path:
        raise ValueError("Curated output must differ from the first-pass generated file")
    source = read_jsonl(input_path)
    rows = [curate_row(row) for row in source]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    summary = {
        "version": rows[0].get("curation_version", CURATION_VERSION) if rows else CURATION_VERSION,
        "events": len(rows),
        "teacher_raw_sha256": sha256(input_path),
        "curated_sha256": sha256(output_path),
        "repair_counts": dict(Counter(repair for row in rows for repair in row["curation_repairs"])),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/raw_events.jsonl")
    parser.add_argument("--output", default="artifacts/curated_events.jsonl")
    parser.add_argument("--manifest", default="artifacts/curation.json")
    args = parser.parse_args()
    curate(Path(args.input), Path(args.output), Path(args.manifest))


if __name__ == "__main__":
    main()
