"""Shared, deterministic data definitions for the text embedding POC."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

SEED = 42
SPLIT_EVENTS = {"train": 200, "dev": 75, "test": 250}
DOCUMENTS_PER_EVENT = 4
TEACHER_MODELS = (
    "qwen3:8b",
    "qwen3:4b",
    "qwen3:4b-instruct-2507-q8_0",
    "qwen3.5:9b",
    "qwen3.5:9b-q4_K_M",
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
DATE_STYLES = ("iso", "korean_full", "korean_short")
DATE_PATTERN = re.compile(
    r"(?P<iso>\d{4}-\d{1,2}-\d{1,2})"
    r"|(?P<full>\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)"
    r"|(?P<short>\d{1,2}\s*월\s*\d{1,2}\s*일)"
)
RELATIVE_DATE_PATTERN = re.compile(
    r"이번\s*주말|다음\s*주말|지난\s*주말|"
    r"(?:이번|다음|지난)\s*주\s*(?:[월화수목금토일]요일|초반?|중반|후반|말)?|"
    r"(?:이번|다음|지난)\s*달\s*(?:초순|중순|하순|초|말)?|"
    r"오늘|어제|그제|내일|모레|글피|올해|내년|작년|"
    r"이번\s*달|다음\s*달|지난\s*달|"
    r"\d{1,2}월\s*(?:초순|중순|하순)|월\s*(?:초순|중순|하순)|"
    r"(?<!\d)[월화수목금토일]요일"
)
RELATIONSHIP_PATTERN = re.compile(
    r"(?:첫째\s*아이|둘째\s*아이|남편|아내|부모님|어머님|아버님|어머니|아버지|엄마|아빠|"
    r"아들|딸|형제|자매|언니|오빠|누나|형|동생)"
    r"(?=(?:은|는|이|가|을|를|의|에게|께서|도|와|과|랑|으로|로)?(?:\s|$|[.,!?]))"
)


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
    answer_doc_index: int
    identity_style: str
    document_date_styles: tuple[str, ...]
    query_date_style: str

    def as_dict(self) -> dict:
        value = asdict(self)
        value["document_date_styles"] = list(self.document_date_styles)
        return value


def _seed_for(*parts: object) -> int:
    value = "|".join([str(SEED), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


@lru_cache(maxsize=None)
def _balanced_assignment(split: str, category: str, domain: str, values: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    members = [
        index for index in range(SPLIT_EVENTS[split])
        if TOPICS[index % len(TOPICS)][0] == category
    ]
    ordered = sorted(members, key=lambda index: (_seed_for(domain, split, category, index), index))
    assignments = list(values) * (len(ordered) // len(values)) + list(values[: len(ordered) % len(values)])
    random.Random(_seed_for("balanced", domain, split, category)).shuffle(assignments)
    return tuple(zip(ordered, assignments))


def _assigned_value(split: str, index: int, category: str, domain: str, values: tuple[str, ...]) -> str:
    return dict(_balanced_assignment(split, category, domain, values))[index]


def format_event_date(value: str, style: str) -> str:
    event_date = date.fromisoformat(value)
    if style == "iso":
        return event_date.isoformat()
    if style == "korean_full":
        return f"{event_date.year}년 {event_date.month}월 {event_date.day}일"
    if style == "korean_short":
        return f"{event_date.month}월 {event_date.day}일"
    raise ValueError(f"Unknown date style: {style}")


def _parse_date_match(match: re.Match, event_year: int) -> date:
    if match.group("iso"):
        return date(*(int(value) for value in match.group("iso").split("-")))
    if match.group("full"):
        values = re.findall(r"\d+", match.group("full"))
        return date(*(int(value) for value in values))
    month, day = map(int, re.findall(r"\d+", match.group("short")))
    return date(event_year, month, day)


def normalize_event_dates(text: str, event_date: str, style: str) -> tuple[str, int]:
    canonical = date.fromisoformat(event_date)
    rendered = format_event_date(event_date, style)
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        if _parse_date_match(match, canonical.year) != canonical:
            raise ValueError(f"Unexpected date for event {event_date}: {match.group(0)}")
        count += 1
        return rendered

    return DATE_PATTERN.sub(replace, text), count


def remove_event_dates(text: str, event_date: str) -> tuple[str, int]:
    canonical = date.fromisoformat(event_date)
    output = []
    cursor = 0
    count = 0
    for match in DATE_PATTERN.finditer(text):
        if _parse_date_match(match, canonical.year) != canonical:
            raise ValueError(f"Unexpected date for event {event_date}: {match.group(0)}")
        output.append(text[cursor : match.start()])
        cursor = match.end()
        suffix = re.match(r"\s*(?:에|은|는|부터|까지)(?=\s|$|[,.!?])", text[cursor:])
        if suffix:
            cursor += suffix.end()
        count += 1
    output.append(text[cursor:])
    normalized = "".join(output)
    normalized = re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s{2,}", " ", normalized)).strip()
    return normalized, count


def normalize_relative_dates(text: str, event_date: str, style: str | None) -> tuple[str, int]:
    replacement = format_event_date(event_date, style) if style is not None else ""
    normalized, count = RELATIVE_DATE_PATTERN.subn(replacement, text)
    return re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s{2,}", " ", normalized)).strip(), count


def event_card(split: str, index: int, revision: str = "shortcut-hardneg-v2-balanced") -> EventCard:
    if revision == "context-hardneg-v3":
        from data_v3 import event_card_v3
        return event_card_v3(split, index)
    if revision != "shortcut-hardneg-v2-balanced":
        raise ValueError(f"Unknown card revision: {revision}")
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
        answer_doc_index=int(_assigned_value(split, index, category, "answer-position", ("0", "1", "2", "3"))),
        identity_style=_assigned_value(split, index, category, "identity-style", ("named", "implicit")),
        document_date_styles=tuple(
            _assigned_value(split, index, category, f"document-date-{doc_index}", DATE_STYLES)
            for doc_index in range(DOCUMENTS_PER_EVENT)
        ),
        query_date_style=_assigned_value(
            split,
            index,
            category,
            "query-date-style",
            ("none", "none", "iso", "korean_full"),
        ),
    )


def cards(split: str | None = None, revision: str = "shortcut-hardneg-v2-balanced"):
    splits = (split,) if split else SPLIT_EVENTS
    for current_split in splits:
        for index in range(SPLIT_EVENTS[current_split]):
            yield event_card(current_split, index, revision)


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


DATASET_REQUIRED_FILES = ("documents.jsonl", "queries.jsonl", "pairs.jsonl", "triplets.jsonl")


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_fingerprint(project: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256(project / name) for name in names}


def dataset_clustering_deidentify(manifest: dict) -> bool:
    if "clustering_deidentify" in manifest:
        value = manifest["clustering_deidentify"]
        if type(value) is not bool:
            raise ValueError("Dataset clustering_deidentify must be a boolean")
        return value
    return str(manifest.get("dataset_revision", "")).startswith("context-")


def verified_dataset_manifest(folder: Path) -> dict:
    """Verify the declared source files before model loading or output mutation."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("files", {})
    missing = set(DATASET_REQUIRED_FILES) - hashes.keys()
    if missing:
        raise ValueError(f"Dataset manifest is missing required file hashes: {sorted(missing)}")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError(f"Dataset manifest contains a non-local filename: {name}")
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    if manifest.get("review_file_sha256") and (
        not (folder / "review.csv").is_file()
        or sha256(folder / "review.csv") != manifest["review_file_sha256"]
    ):
        raise ValueError("Dataset hash mismatch: review.csv")
    return manifest


def elapsed_budget_seconds(path: Path) -> float:
    value = json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid elapsed compute budget: {path}")
    return float(value)


def generation_budget_path(folder: Path, manifest: dict, explicit: str | Path | None = None) -> Path | None:
    """Locate the same generation ledger for direct and copied/mined datasets."""
    if explicit is not None:
        selected = Path(explicit).resolve()
        if not selected.is_file():
            raise ValueError(f"Generation budget file is missing: {selected}")
        elapsed_budget_seconds(selected)
        return selected
    mining = manifest.get("mining", {})
    candidates = []
    if mining.get("generation_budget_path"):
        declared = Path(mining["generation_budget_path"])
        candidates.append(declared if declared.is_absolute() else folder.parent / declared)
    candidates.append(folder.parent / "generation_budget.json")
    if mining.get("source_dataset_path"):
        source = Path(mining["source_dataset_path"])
        candidates.append((source if source.is_absolute() else folder.parent / source).parent / "generation_budget.json")
    for candidate in candidates:
        if candidate.is_file():
            elapsed_budget_seconds(candidate)
            return candidate.resolve()
    if str(manifest.get("dataset_revision", "")).startswith("context-"):
        raise ValueError("A v3 dataset requires its generation budget before mining or training")
    return None


def matching_doc_text(document: dict) -> str:
    return f"title: {document['title']} | text: {document['text']}"


def search_query_text(query: str) -> str:
    return f"task: search result | query: {query}"


def deidentify_person(text: str, person: str) -> str:
    if not person:
        return text
    boundary = r"(?<![가-힣A-Za-z])"
    particles = r"(?:이에게|에게서|에게|께서|으로|이가|이는|이를|이의|이와|은|는|이|가|을|를|의|도|와|과|랑|아|야)"
    text = re.sub(boundary + re.escape(person), "[인물]", text)
    # A bare given name can also be an ordinary Korean word (e.g. '집을 지은 뒤').
    # Only explicit name-like particle/address forms are masked without a surname.
    if len(person) >= 3:
        text = re.sub(boundary + re.escape(person[1:]) + rf"{particles}(?=\s|$|[.,!?])", "[인물]", text)
    return text


def clustering_text(document: dict, deidentify: bool | None = None) -> str:
    if deidentify is None:
        deidentify = document.get("card_revision") == "context-hardneg-v3"
    title, text = document["title"], document["text"]
    if deidentify:
        title = deidentify_person(title, document.get("person", ""))
        text = deidentify_person(text, document.get("person", ""))
    return f"task: clustering | query: {title}\n{text}"
