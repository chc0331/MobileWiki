"""Shared, deterministic data definitions for the text embedding POC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

SEED = 42
SPLIT_EVENTS = {"train": 200, "dev": 75, "test": 250}
DOCUMENTS_PER_EVENT = 4
TEACHER_MODELS = (
    "qwen3:8b",
    "qwen3:4b",
)
STUDENT_MODEL = "google/embeddinggemma-2"

TOPICS = (
    ("소아과 예약", "진료 장소", ("푸른숲 소아과", "햇살 소아과", "봄빛 소아과")),
    ("어린이집 준비", "준비물", ("실내화", "여벌 양말", "그림책")),
    ("가족 여행", "숙소", ("바람길 펜션", "하늘마루 호텔", "파도빛 게스트하우스")),
    ("연차 계획", "업무 인계", ("주간 보고서", "고객 문의 목록", "회의 자료")),
    ("예방접종 예약", "방문 장소", ("서쪽 보건소", "중앙 의원", "동네 보건지소")),
    ("학교 행사", "행사 장소", ("학교 강당", "운동장", "도서관")),
    ("차량 정비", "정비 항목", ("와이퍼 교체", "타이어 점검", "엔진오일 교환")),
    ("주말 약속", "만남 장소", ("동네 도서관", "하천 산책로", "시민 공원")),
    ("집 수리", "수리 대상", ("주방 수도꼭지", "현관 조명", "욕실 환풍기")),
    ("회사 일정", "준비 자료", ("예산 초안", "발표 슬라이드", "회의록")),
)
PARAPHRASE_QUERIES = {
    "소아과 예약": "아이 병원 어디로 가기로 했었지?",
    "어린이집 준비": "어린이집에 챙겨 갈 물건이 뭐였지?",
    "가족 여행": "여행 때 어디에서 묵기로 했더라?",
    "연차 계획": "쉬기 전에 동료에게 넘길 자료가 뭐였지?",
    "예방접종 예약": "예방접종을 어디에서 받기로 했지?",
    "학교 행사": "학교 행사는 어디에서 열리지?",
    "차량 정비": "차에서 무엇을 손보기로 했지?",
    "주말 약속": "주말에 어디에서 만나기로 했지?",
    "집 수리": "집에서 무엇을 고치기로 했지?",
    "회사 일정": "회사 회의에 가져갈 문서가 뭐였지?",
}
SURNAMES = ("김", "이", "박", "최", "정", "한", "윤", "강", "조", "신")
FIRST_SYLLABLES = ("가", "나", "다", "라", "서", "지", "하", "도", "유", "민")
LAST_SYLLABLES = ("은", "윤", "현", "린", "준", "서", "원", "우", "빈", "희")
FAMILY_OFFSETS = {"train": 0, "dev": 40, "test": 55}


@dataclass(frozen=True)
class EventCard:
    event_id: str
    split: str
    family_id: str
    person: str
    category: str
    date: str
    fact_label: str
    fact_value: str

    def as_dict(self) -> dict:
        return asdict(self)


def event_card(split: str, index: int) -> EventCard:
    if split not in SPLIT_EVENTS or not 0 <= index < SPLIT_EVENTS[split]:
        raise ValueError(f"Invalid event: {split}/{index}")
    category, label, values = TOPICS[index % len(TOPICS)]
    day = date(2025, 1, 1) + timedelta(days=(index * 13 + len(split) * 37) % 650)
    family_number = FAMILY_OFFSETS[split] + index // 5
    person = (
        SURNAMES[family_number % 10]
        + FIRST_SYLLABLES[(family_number // 10) % 10]
        + LAST_SYLLABLES[(family_number // 100) % 10]
    )
    return EventCard(
        event_id=f"{split}-{index:04d}",
        split=split,
        family_id=f"{split}-family-{index // 5:03d}",
        person=person,
        category=category,
        date=day.isoformat(),
        fact_label=label,
        fact_value=values[(index // len(TOPICS)) % len(values)],
    )


def cards(split: str | None = None):
    splits = (split,) if split else SPLIT_EVENTS
    for current_split in splits:
        for index in range(SPLIT_EVENTS[current_split]):
            yield event_card(current_split, index)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matching_doc_text(document: dict) -> str:
    return f"title: {document['title']} | text: {document['text']}"


def search_query_text(query: str) -> str:
    return f"task: search result | query: {query}"


def clustering_text(document: dict) -> str:
    return f"task: clustering | query: {document['title']}\n{document['text']}"
