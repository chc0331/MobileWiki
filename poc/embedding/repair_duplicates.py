"""Regenerate later events whose note text duplicates another event."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from common import DATE_PATTERN, event_card, format_event_date, read_jsonl, write_jsonl
from generate import PROMPT_VERSION, RESPONSE_SCHEMA, ollama_request, parse_response, prompt, teacher_digest

REPAIR_PROMPT_VERSION = "duplicate-rewrite-v5-balanced-shortcut"


def normalized_text(text: str, card, slot: int) -> str:
    style = card.document_date_styles[slot]
    if not DATE_PATTERN.search(text):
        text = f"{format_event_date(card.date, style)} {text}"
    text = DATE_PATTERN.sub("<DATE>", text)
    text = text.replace(card.person, "")
    return " ".join(text.split())


def normalized_texts(row: dict, card=None) -> set[str]:
    if card is None:
        split, index = row["event_id"].split("-")
        card = event_card(split, int(index), row.get("card_revision", "shortcut-hardneg-v2-balanced"))
    return {
        normalized_text(document["text"], card, slot)
        for slot, document in enumerate(row["documents"])
    }


def duplicate_events(rows: list[dict]) -> list[str]:
    first_seen: dict[str, str] = {}
    later_events = []
    for row in rows:
        for text in normalized_texts(row):
            if text in first_seen and first_seen[text] != row["event_id"]:
                if row["event_id"] not in later_events:
                    later_events.append(row["event_id"])
            else:
                first_seen[text] = row["event_id"]
    return later_events


def repair(args: argparse.Namespace) -> None:
    path = Path(args.input)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No generated events to repair")
    if args.max_hours is None:
        args.max_hours = 24 if rows[0].get("card_revision") == "context-hardneg-v3" else 12
    if rows[0].get("card_revision") == "context-hardneg-v3":
        from data_v3 import repair_duplicates as repair_v3_duplicates
        repair_v3_duplicates(args)
        return
    teacher = rows[0]["teacher_id"]
    revision = teacher_digest(teacher)
    if any(
        row["teacher_id"] != teacher or row["teacher_revision"] != revision or row["prompt_version"] != PROMPT_VERSION
        for row in rows
    ):
        raise ValueError("Generated events use another teacher snapshot or prompt version")
    targets = duplicate_events(rows)
    if not targets:
        print("No cross-event duplicate note text")
        return
    protected = {f"test-{index:04d}" for index in range(140)}
    if any(event_id in protected for event_id in targets) and not args.allow_test_repair:
        raise ValueError("A reviewed test event would change; renew the review sheet first")
    budget_path = path.parent / "generation_budget.json"
    previous = json.loads(budget_path.read_text(encoding="utf-8"))["elapsed_seconds"]
    started = time.monotonic()
    by_id = {row["event_id"]: index for index, row in enumerate(rows)}
    for event_id in targets:
        index = by_id[event_id]
        original = rows[index]
        split, number = event_id.split("-")
        card = event_card(split, int(number))
        other_rows = [row for i, row in enumerate(rows) if i != index]
        other_texts = set().union(*(normalized_texts(row) for row in other_rows))
        duplicate_signatures = normalized_texts(original, card) & other_texts
        colliding_slots = [
            {"slot": slot, "signature": normalized_text(document["text"], card, slot).replace("<DATE>", "<날짜>")}
            for slot, document in enumerate(original["documents"])
            if normalized_text(document["text"], card, slot) in duplicate_signatures
        ]
        conflicting_notes = []
        for row in other_rows:
            other_split, other_number = row["event_id"].split("-")
            other_card = event_card(other_split, int(other_number))
            conflicting_notes.extend(
                {"event_id": row["event_id"], "text": document["text"]}
                for slot, document in enumerate(row["documents"])
                if normalized_text(document["text"], other_card, slot) in duplicate_signatures
            )
        user_text = prompt(card.as_dict()) + (
            "\n다른 사건의 메모와 같은 문장이 발견되었다. 사건 카드의 주제와 날짜·인물 표현 규칙을 유지하면서 "
            "네 메모 네 개를 모두 새 문장으로 다시 써라. 아래 다른 사건 메모와 표현·문장 구조가 겹치지 않게 하라. "
            "특히 충돌 위치 문장과 금지 표현은 그대로 쓰지 말고, 단어만 바꾼 문장이나 뜻이 같은 상투 문장도 쓰지 말라. "
            "해당 메모는 다른 기록 관점과 사건의 구체적 준비·후속 행동을 담아 문장 구조를 새로 만들어라. "
            "충돌 표식과 다른 사건 메모의 이름·날짜는 복사하지 말라.\n"
            "충돌 위치와 절대 사용하지 말아야 할 정규화 문장:\n"
            + json.dumps(colliding_slots, ensure_ascii=False)
            + "\n"
            "겹친 다른 사건 메모:\n"
            + json.dumps(conflicting_notes, ensure_ascii=False)
            + "\n기존 대상 사건 메모:\n"
            + json.dumps([document["text"] for document in original["documents"]], ensure_ascii=False)
        )
        for attempt in range(1, 21):
            if previous + time.monotonic() - started >= args.max_hours * 3600:
                raise TimeoutError("Teacher generation and repair time budget reached")
            seed = 90000 + int(number) * 10 + attempt
            response = ollama_request(
                "/api/chat",
                {
                    "model": teacher,
                    "messages": [
                        {"role": "system", "content": "한국어 생활 메모 작성자다. 중국어와 한자를 절대 쓰지 않는다. JSON만 출력한다."},
                        {"role": "user", "content": user_text},
                    ],
                    "format": RESPONSE_SCHEMA,
                    "think": False,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0.9, "num_predict": 1200, "seed": seed},
                },
            )
            budget_path.write_text(
                json.dumps({"elapsed_seconds": previous + time.monotonic() - started}, indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                parsed = parse_response(response["message"]["content"], card.as_dict())
            except (ValueError, json.JSONDecodeError) as error:
                print(f"{event_id} attempt {attempt}: {error}", flush=True)
                continue
            if len(normalized_texts(parsed, card)) != 4:
                print(f"{event_id} attempt {attempt}: duplicate within rewritten event", flush=True)
                continue
            if normalized_texts(parsed, card) & other_texts:
                print(f"{event_id} attempt {attempt}: cross-event duplicate remains", flush=True)
                continue
            history = original.get("regeneration_history", [])
            history.append({
                "generated_at": original["generated_at"],
                "documents": original["documents"],
                "query": original["query"],
                "evidence": original["evidence"],
                "repairs": original.get("repairs", []),
            })
            rows[index] = {
                **original,
                **parsed,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "regeneration_seed": seed,
                "repair_prompt_version": REPAIR_PROMPT_VERSION,
                "regeneration_history": history,
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            write_jsonl(temporary, rows)
            os.replace(temporary, path)
            budget_path.write_text(
                json.dumps({"elapsed_seconds": previous + time.monotonic() - started}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Repaired {event_id} with seed {seed}", flush=True)
            break
        else:
            raise RuntimeError(f"Unable to regenerate nonduplicate notes for {event_id}")
    if duplicate_events(rows):
        raise ValueError("Cross-event duplicate note text remains")
    print(f"Repaired {len(targets)} events")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/raw_events.jsonl")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--allow-test-repair", action="store_true", help="Allow regenerating test events in a new, unreviewed dataset revision")
    repair(parser.parse_args())


if __name__ == "__main__":
    main()
