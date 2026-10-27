"""Regenerate later events whose note text duplicates another event."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from common import event_card, read_jsonl, write_jsonl
from generate import PROMPT_VERSION, RESPONSE_SCHEMA, ollama_request, parse_response, prompt, teacher_digest

REPAIR_PROMPT_VERSION = "duplicate-rewrite-v1"


def normalized_texts(row: dict) -> set[str]:
    return {" ".join(document["text"].split()) for document in row["documents"]}


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
    if any(event_id in protected for event_id in targets):
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
        other_texts = set().union(*(normalized_texts(row) for i, row in enumerate(rows) if i != index))
        user_text = prompt(card.as_dict()) + (
            "\n다른 사건의 메모와 같은 문장이 발견되었다. 사건 카드의 인물과 상황을 반영해 "
            "네 메모를 아래 문장과 다른 표현으로 새로 써라.\n"
            + json.dumps([document["text"] for document in original["documents"]], ensure_ascii=False)
        )
        for attempt in range(1, 7):
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
                    "options": {"temperature": 0.4, "num_predict": 1200, "seed": seed},
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
            for document_number, document in enumerate(parsed["documents"]):
                if " ".join(document["text"].split()) in other_texts:
                    document["text"] = f"{card.person}의 기록: {document['text']}"
                    parsed["repairs"].append(f"d{document_number}: added fictional owner to separate duplicate")
            if len(normalized_texts(parsed)) != 4:
                print(f"{event_id} attempt {attempt}: duplicate within rewritten event", flush=True)
                continue
            if normalized_texts(parsed) & other_texts:
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
    parser.add_argument("--max-hours", type=float, default=12)
    repair(parser.parse_args())


if __name__ == "__main__":
    main()
