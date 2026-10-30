"""Generate fictional Korean note events with a local Ollama Qwen teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import (
    DATE_PATTERN,
    DOCUMENTS_PER_EVENT,
    RELATIVE_DATE_PATTERN,
    RELATIONSHIP_PATTERN,
    TEACHER_MODELS,
    cards,
    format_event_date,
    normalize_event_dates,
    normalize_relative_dates,
    remove_event_dates,
)

PROMPT_VERSION = "qwen-event-v6-balanced-shortcut"
OLLAMA_URL = "http://127.0.0.1:11434"
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
                "required": ["title", "text"],
            },
            "minItems": DOCUMENTS_PER_EVENT,
            "maxItems": DOCUMENTS_PER_EVENT,
        },
        "query": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["documents", "query", "evidence"],
}


def prompt(card: dict) -> str:
    date_instructions = "\n".join(
        f"d{index}: {format_event_date(card['date'], style)}"
        for index, style in enumerate(card["document_date_styles"])
    )
    query_date = (
        "질문에는 날짜를 쓰지 않는다."
        if card["query_date_style"] == "none"
        else f"질문에는 '{format_event_date(card['date'], card['query_date_style'])}'를 자연스럽게 포함한다."
    )
    identity_instruction = (
        f"가상 인물 이름 '{card['person']}'을 질문과 메모 본문에 자연스럽게 한 번 이상 쓴다. 제목에는 이름을 넣지 않는다."
        if card["identity_style"] == "named"
        else f"주어를 생략하고 쓴다. 가상 인물 이름 '{card['person']}'이나 관계 호칭을 질문·제목·본문 어디에도 쓰지 않는다."
    )
    answer_index = card["answer_doc_index"]
    return f"""다음은 완전히 가상의 한국어 생활 메모 데이터다. JSON 객체 하나만 출력하라.
사건 카드: {json.dumps(card, ensure_ascii=False)}

documents는 같은 구체적 사건에 관한 서로 다른 메모 4개다. 각 메모는 title과 text가 있어야 한다.
예약 확인, 준비, 가족에게 전달, 나중에 다시 확인 등 서로 다른 관점으로 쓴다. 같은 문장을 반복하지 말라.
각 text는 50~350자이고 자연스러운 한국어로 쓴다. 네 메모 모두 사건의 날짜와 주제를 알 수 있어야 한다.
문서별 날짜 표기:
{date_instructions}
질문 날짜 규칙: {query_date}
인물 표현 규칙: {identity_instruction}
정확한 절대 날짜만 쓴다. '이번 주 토요일', '다음 주', '3월 중순' 같은 상대·대략 날짜는 쓰지 않는다.
제목에는 인물 이름을 넣지 않는다. 이름을 기계적으로 제목 앞에 붙이지 않는다.
event_id와 family_id는 내부 식별자이므로 메모에 쓰지 말라.
중국어·한자는 쓰지 말고, 고유명사 외에는 영어도 쓰지 말라.
정답은 d{answer_index}에만 둔다. 나머지 세 메모에는 '{card['fact_value']}'를 절대 쓰지 말라. 네 메모는 같은 사건의 일정·준비·후속 기록이다.
query는 정답이 든 메모를 찾는 짧고 자연스러운 질문이다. 질문에 정답 표현을 쓰지 말라.
evidence는 d{answer_index}에서 답이 들어 있는 문장을 글자 그대로 복사한다.
실제 인물의 연락처, 주민번호, 실제 진료 조언, 약 복용량은 만들지 말라.

출력 형식: {{"documents":[{{"title":"...","text":"..."}} 4개],"query":"...","evidence":"..."}}
JSON 밖의 설명이나 코드 블록은 쓰지 말라."""


def parse_response(response: str, card: dict) -> dict:
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Teacher response has no JSON object")
    result = json.loads(response[start : end + 1])
    documents = result.get("documents")
    if not isinstance(documents, list) or len(documents) != DOCUMENTS_PER_EVENT:
        raise ValueError("Expected exactly four documents")
    repairs = []
    answer_index = card["answer_doc_index"]
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError("Invalid document object")
        title, body = document.get("title"), document.get("text")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Missing title")
        if not isinstance(body, str):
            raise ValueError("Missing document text")
        if "小朋友" in title + body:
            title, body = title.replace("小朋友", "아이"), body.replace("小朋友", "아이")
            repairs.append(f"d{index}: replaced Chinese word for child")
        style = card["document_date_styles"][index]
        title, title_relative_dates = normalize_relative_dates(title, card["date"], style)
        body, body_relative_dates = normalize_relative_dates(body, card["date"], style)
        if title_relative_dates or body_relative_dates:
            repairs.append(f"d{index}: replaced relative date with exact event date")
        title, title_dates = normalize_event_dates(title, card["date"], style)
        body, body_dates = normalize_event_dates(body, card["date"], style)
        if title_dates or body_dates:
            repairs.append(f"d{index}: normalized event date to {style}")
        if not body_dates:
            body = f"{format_event_date(card['date'], style)} {body}"
            repairs.append(f"d{index}: inserted event date in {style} format")
        if RELATIVE_DATE_PATTERN.search(title + body):
            raise ValueError(f"Relative or approximate date found in d{index}")
        if card["person"] in title:
            raise ValueError(f"Person name found in d{index} title")
        if card["identity_style"] == "implicit" and (
            card["person"] in title + body or RELATIONSHIP_PATTERN.search(title + body)
        ):
            raise ValueError(f"Identity shortcut found in implicit d{index}")
        document["title"], document["text"] = title.strip(), body.strip()
        if re.search(r"[\u4e00-\u9fff]", title + body):
            raise ValueError("Chinese characters found in document")
    target = f"{card['fact_label']}: {card['fact_value']}"
    answer_sentence = None
    provided_evidence = result.get("evidence")
    sentence_lists = [re.split(r"(?<=[.!?])\s+", document["text"]) for document in documents]
    if isinstance(provided_evidence, str) and card["fact_value"] in provided_evidence:
        answer_sentence = next(
            (sentence.strip() for sentences in sentence_lists for sentence in sentences if provided_evidence in sentence and card["fact_value"] in sentence),
            None,
        )
    if answer_sentence is None:
        answer_sentence = next(
            (sentence.strip() for sentences in sentence_lists for sentence in sentences if card["fact_value"] in sentence),
            None,
        )
    if answer_sentence is None:
        answer_sentence = target + "."
        repairs.append(f"d{answer_index}: inserted card fact")
    answer_was_in_target = any(card["fact_value"] in sentence for sentence in sentence_lists[answer_index])
    for index, document in enumerate(documents):
        sentences = sentence_lists[index]
        if index == answer_index:
            kept = [sentence.strip() for sentence in sentences if card["fact_value"] not in sentence]
            document["text"] = " ".join(kept + [answer_sentence]).strip()
            if not answer_was_in_target:
                repairs.append(f"d{answer_index}: moved answer to configured document")
            if card["fact_value"] in document["title"]:
                document["title"] = f"{card['category']} 기록 {index}"
                repairs.append(f"d{index}: replaced answer-bearing title")
        else:
            kept = [sentence.strip() for sentence in sentences if card["fact_value"] not in sentence]
            if len(kept) != len(sentences):
                document["text"] = " ".join(kept).strip()
                repairs.append(f"d{index}: removed answer-bearing sentence")
            if card["fact_value"] in document["title"]:
                document["title"] = f"{card['category']} 기록 {index}"
                repairs.append(f"d{index}: replaced answer-bearing title")
    answer_style = card["document_date_styles"][answer_index]
    documents[answer_index]["text"], _ = normalize_event_dates(
        documents[answer_index]["text"], card["date"], answer_style
    )
    answer_sentence = next(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", documents[answer_index]["text"])
        if card["fact_value"] in sentence
    )
    if card["identity_style"] == "named" and card["person"] not in " ".join(
        document["text"] for document in documents
    ):
        name_index = int.from_bytes(
            hashlib.sha256(card["event_id"].encode("utf-8")).digest()[:4], "big"
        ) % DOCUMENTS_PER_EVENT
        documents[name_index]["text"] += f" {card['person']}이 다시 확인했다."
        repairs.append(f"d{name_index}: added assigned fictional name to document text")
    work_category = card["category"] in {"연차 계획", "회사 일정"}
    fallback = (
        "일정과 필요한 자료를 다시 확인하고, 바뀌는 점이 있으면 동료에게 알려 두기로 했다."
        if work_category else
        "일정과 필요한 준비를 다시 확인하고, 바뀌는 점이 있으면 가족에게 알려 두기로 했다."
    )
    for index, document in enumerate(documents):
        if index == answer_index:
            continue
        if len(document["text"]) < 40:
            document["text"] = (document["text"] + " " + fallback).strip()
            repairs.append(f"d{index}: completed short distractor")
        if card["fact_value"] in document["text"]:
            raise ValueError(f"Answer remains in distractor d{index}")
        if card["person"] in document["title"]:
            raise ValueError(f"Person name found in d{index} title")
        if card["identity_style"] == "implicit" and card["person"] in document["text"]:
            raise ValueError(f"Person name found in implicit d{index}")
    for document in documents:
        if not 40 <= len(document["text"]) <= 500:
            raise ValueError("Document length outside 40–500 characters")
    normalized_texts = [
        " ".join(normalize_event_dates(document["text"], card["date"], "iso")[0].split())
        for document in documents
    ]
    if len(set(normalized_texts)) != DOCUMENTS_PER_EVENT:
        raise ValueError("Duplicate document text within one event")
    question, evidence = result.get("query"), answer_sentence
    if not isinstance(question, str):
        raise ValueError("Invalid query")
    if "小朋友" in question:
        question = question.replace("小朋友", "아이")
        repairs.append("query: replaced Chinese word for child")
    if card["fact_value"] in question:
        last = card["fact_label"][-1]
        particle = "은" if (ord(last) - 0xAC00) % 28 else "는"
        question = f"{card['category']}의 {card['fact_label']}{particle} 무엇이었지?"
        repairs.append("query: removed answer leakage")
    date_style = card["query_date_style"]
    question, relative_dates = normalize_relative_dates(
        question,
        card["date"],
        date_style if date_style != "none" else None,
    )
    if relative_dates:
        repairs.append("query: removed relative date" if date_style == "none" else "query: replaced relative date with exact event date")
    if RELATIVE_DATE_PATTERN.search(question):
        raise ValueError("Relative or approximate date found in query")
    if date_style == "none":
        question, removed_dates = remove_event_dates(question, card["date"])
        if removed_dates:
            repairs.append("query: removed unassigned absolute date")
        if DATE_PATTERN.search(question):
            raise ValueError("Query has a date despite the no-date assignment")
    else:
        question, date_count = normalize_event_dates(question, card["date"], date_style)
        if not date_count:
            question = f"{format_event_date(card['date'], date_style)}에 {question}"
            repairs.append(f"query: inserted event date in {date_style} format")
    if card["identity_style"] == "implicit" and (
        card["person"] in question or RELATIONSHIP_PATTERN.search(question)
    ):
        raise ValueError("Identity shortcut found in implicit query")
    if card["identity_style"] == "named" and card["person"] not in question:
        last_syllable = ord(card["person"][-1]) - 0xAC00
        particle = "이" if 0 <= last_syllable <= 11171 and last_syllable % 28 else "가"
        question = f"{card['person']}{particle} 남긴 기록에서 {question}"
        repairs.append("query: added assigned fictional person naturally")
    if card["person"] in " ".join(document["title"] for document in documents):
        raise ValueError("Person name found in a title")
    all_document_text = " ".join(document["text"] for document in documents)
    if card["identity_style"] == "implicit" and (
        card["person"] in all_document_text or RELATIONSHIP_PATTERN.search(all_document_text)
    ):
        raise ValueError("Identity shortcut found in implicit event")
    if card["identity_style"] == "named" and card["person"] not in all_document_text:
        raise ValueError("Expected the fictional person name in named event text")
    if not 8 <= len(question.strip()) <= 150 or re.search(r"[\u4e00-\u9fff]", question):
        raise ValueError("Invalid or non-Korean query")
    if evidence not in documents[answer_index]["text"] or card["fact_value"] not in evidence:
        raise ValueError(f"Unable to ground answer in configured document d{answer_index}")
    return {"documents": documents, "query": question.strip(), "evidence": evidence.strip(), "repairs": repairs}


def ollama_request(endpoint: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_URL + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama {endpoint} HTTP {error.code}: {detail}") from error


def teacher_digest(model: str) -> str:
    for item in ollama_request("/api/tags")["models"]:
        if item["name"] == model:
            return item["digest"]
    raise RuntimeError(f"Local teacher {model} is missing. Run: ollama pull {model}")


def generate(args: argparse.Namespace) -> None:
    if getattr(args, "revision", "shortcut-hardneg-v2-balanced") == "context-hardneg-v3":
        from data_v3 import generate as generate_v3
        generate_v3(args)
        return
    if args.teacher not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {args.teacher}")
    teacher_revision = teacher_digest(args.teacher)
    print(f"Teacher: {args.teacher}@{teacher_revision}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            existing = [json.loads(line) for line in stream if line.strip()]
        if existing and any(
            row["teacher_id"] != args.teacher
            or row["teacher_revision"] != teacher_revision
            or row["prompt_version"] != PROMPT_VERSION
            for row in existing
        ):
            raise ValueError("Existing rows use a different teacher snapshot or prompt version")
    completed = {row["event_id"] for row in existing}
    selected = list(cards(args.split))
    if args.max_events:
        selected = selected[: args.max_events]
    budget_path = output.parent / "generation_budget.json"
    previous_seconds = 0.0
    if budget_path.exists():
        previous_seconds = json.loads(budget_path.read_text(encoding="utf-8"))["elapsed_seconds"]
    started = time.monotonic()
    for number, card in enumerate(selected, start=1):
        if card.event_id in completed:
            continue
        if previous_seconds + time.monotonic() - started >= args.max_hours * 3600:
            raise TimeoutError("Generation time budget reached")
        user_text = prompt(card.as_dict())
        for attempt in range(1, 4):
            result = ollama_request(
                "/api/chat",
                {
                    "model": args.teacher,
                    "messages": [
                        {"role": "system", "content": "한국어 생활 메모 작성자다. 중국어와 한자를 절대 쓰지 않는다. JSON만 출력한다."},
                        {"role": "user", "content": user_text},
                    ],
                    "format": RESPONSE_SCHEMA,
                    "think": False,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0.2, "num_predict": args.max_tokens, "seed": 42 + number + attempt},
                },
            )
            try:
                parsed = parse_response(result["message"]["content"], card.as_dict())
                break
            except (ValueError, json.JSONDecodeError) as error:
                print(f"{card.event_id} attempt {attempt}: {error}", flush=True)
                if attempt == 3:
                    raise RuntimeError(f"Unable to generate {card.event_id}") from error
        row = {
            **card.as_dict(),
            **parsed,
            "teacher_id": args.teacher,
            "teacher_revision": teacher_revision,
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        budget_path.write_text(
            json.dumps({"elapsed_seconds": previous_seconds + time.monotonic() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{number}/{len(selected)} {card.event_id}", flush=True)
    spent = time.monotonic() - started
    budget_path.write_text(
        json.dumps({"elapsed_seconds": previous_seconds + spent}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(selected)} events in {spent:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/raw_events.jsonl")
    parser.add_argument("--teacher", default=TEACHER_MODELS[0], choices=TEACHER_MODELS)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--revision", choices=("shortcut-hardneg-v2-balanced", "context-hardneg-v3"), default="shortcut-hardneg-v2-balanced")
    parser.add_argument("--seed-offset", type=int, default=0, help="v3 only: use this fresh cache/response namespace for unfinished and subsequent new events; completed rows are validated and skipped")
    args = parser.parse_args()
    if args.seed_offset < 0:
        parser.error("--seed-offset must be >= 0")
    if args.seed_offset and args.revision != "context-hardneg-v3":
        parser.error("--seed-offset is supported only for context-hardneg-v3")
    if args.max_hours is None:
        args.max_hours = 24 if args.revision == "context-hardneg-v3" else 12
    generate(args)


if __name__ == "__main__":
    main()
