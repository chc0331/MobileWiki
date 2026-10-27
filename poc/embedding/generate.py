"""Generate fictional Korean note events with a local Ollama Qwen teacher."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import DOCUMENTS_PER_EVENT, TEACHER_MODELS, cards

PROMPT_VERSION = "qwen-event-v5"
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
    return f"""다음은 완전히 가상의 한국어 생활 메모 데이터다. JSON 객체 하나만 출력하라.
사건 카드: {json.dumps(card, ensure_ascii=False)}

documents는 같은 구체적 사건에 관한 서로 다른 메모 4개다. 각 메모는 title과 text가 있어야 한다.
예약 확인, 준비, 가족에게 전달, 나중에 다시 확인 등 서로 다른 관점으로 쓴다. 같은 문장을 반복하지 말라.
각 text는 50~350자이고 자연스러운 한국어로 쓴다. 네 메모 모두 '{card['date']}'를 그대로 넣고 사건의 주제를 알 수 있어야 한다.
event_id와 family_id는 내부 식별자이므로 메모에 쓰지 말라.
중국어·한자는 쓰지 말고, 고유명사 외에는 영어도 쓰지 말라.
첫 메모에는 '{card['fact_value']}'를 답으로 분명하게 남긴다.
두 번째부터 네 번째 메모에는 '{card['fact_value']}'를 절대 쓰지 말라. 그 메모들은 같은 사건의 일정·준비·후속 기록이지만 질문의 답은 담지 않는다.
query는 첫 메모에 있는 그 사실을 찾는 짧고 자연스러운 질문이다. 질문에 정답 표현을 쓰지 말라.
evidence는 첫 메모에서 답이 들어 있는 문장을 글자 그대로 복사한다.
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
        def normalize_date(match: re.Match) -> str:
            year, month, day = map(int, match.groups())
            normalized = f"{year:04d}-{month:02d}-{day:02d}"
            return normalized if normalized == card["date"] else match.group()

        normalized_body = re.sub(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", normalize_date, body)
        if normalized_body != body:
            body = normalized_body
            repairs.append(f"d{index}: normalized event date")
        if card["date"] not in body:
            body = f"{card['date']} {body}"
            repairs.append(f"d{index}: inserted event date")
        document["title"], document["text"] = title.strip(), body.strip()
        if re.search(r"[\u4e00-\u9fff]", title + body):
            raise ValueError("Chinese characters found in document")
    target = f"{card['fact_label']}: {card['fact_value']}"
    if card["fact_value"] not in documents[0]["text"]:
        documents[0]["text"] += f" {target}."
        repairs.append("d0: inserted card fact")
    work_category = card["category"] in {"연차 계획", "회사 일정"}
    fallback = (
        "일정과 필요한 자료를 다시 확인하고, 바뀌는 점이 있으면 동료에게 알려 두기로 했다."
        if work_category else
        "일정과 필요한 준비를 다시 확인하고, 바뀌는 점이 있으면 가족에게 알려 두기로 했다."
    )
    for index, document in enumerate(documents[1:], start=1):
        if card["fact_value"] in document["title"]:
            document["title"] = f"{card['category']} 기록 {index}"
            repairs.append(f"d{index}: replaced answer-bearing title")
        if card["fact_value"] in document["text"]:
            sentences = re.split(r"(?<=[.!?])\s+", document["text"])
            document["text"] = " ".join(sentence for sentence in sentences if card["fact_value"] not in sentence)
            repairs.append(f"d{index}: removed answer-bearing sentence")
        if len(document["text"]) < 40:
            document["text"] = (document["text"] + " " + fallback).strip()
            repairs.append(f"d{index}: completed short distractor")
        if card["date"] not in document["text"]:
            document["text"] = f"{card['date']} {document['text']}"
            repairs.append(f"d{index}: restored event date")
    for document in documents:
        if not 40 <= len(document["text"]) <= 500:
            raise ValueError("Document length outside 40–500 characters")
    question, evidence = result.get("query"), result.get("evidence")
    if not isinstance(question, str):
        raise ValueError("Invalid query")
    if "小朋友" in question:
        question = question.replace("小朋友", "아이")
        repairs.append("query: replaced Chinese word for child")
    if card["fact_value"] in question:
        last = card["fact_label"][-1]
        particle = "은" if (ord(last) - 0xAC00) % 28 else "는"
        question = f"{card['date']} {card['category']}의 {card['fact_label']}{particle} 무엇이었지?"
        repairs.append("query: removed answer leakage")
    if not 8 <= len(question.strip()) <= 150 or re.search(r"[\u4e00-\u9fff]", question):
        raise ValueError("Invalid or non-Korean query")
    if not isinstance(evidence, str) or card["fact_value"] not in evidence or evidence not in documents[0]["text"]:
        candidates = re.split(r"(?<=[.!?])\s+", documents[0]["text"])
        evidence = next((item for item in candidates if card["fact_value"] in item), target)
        repairs.append("evidence: copied source sentence")
    if evidence not in documents[0]["text"] or card["fact_value"] not in evidence:
        raise ValueError("Unable to ground answer in first document")
    return {"documents": documents, "query": question.strip(), "evidence": evidence.strip(), "repairs": repairs}


def ollama_request(endpoint: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_URL + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def teacher_digest(model: str) -> str:
    for item in ollama_request("/api/tags")["models"]:
        if item["name"] == model:
            return item["digest"]
    raise RuntimeError(f"Local teacher {model} is missing. Run: ollama pull {model}")


def generate(args: argparse.Namespace) -> None:
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
    parser.add_argument("--max-hours", type=float, default=12)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
