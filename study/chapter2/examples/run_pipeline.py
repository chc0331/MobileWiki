#!/usr/bin/env python3
"""Read the Chapter 2 samples, check source groups, and write inspectable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[2] / "_workspace" / "chapter2"
HEADER = "IT-02 | LEARNING SAMPLE"
SPLIT_BY_SUFFIX = {"01": "train", "02": "validation", "03": "final_test"}
LABELS = {"IT", "경제", "스포츠"}
LABEL_BY_PREFIX = {"IT": "IT", "ECO": "경제", "SPT": "스포츠"}
DERIVED_FROM = {"IT-01-EN": "IT-01", "ECO-02-COPY": "ECO-02", "SPT-03-FRAG": "SPT-03"}


def normalize_lines(lines: list[str]) -> str:
    return " ".join(re.sub(r"[ \t]+", " ", line.strip()) for line in lines if line.strip()).strip()


def read_source(path: Path) -> tuple[str, list[str], list[str], int, str]:
    source_bytes = path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if path.suffix.lower() == ".txt":
        raw = source_bytes.decode("utf-8", errors="strict")
        clean_input = raw.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [normalize_lines(block.splitlines()) for block in re.split(r"\n\s*\n", clean_input)]
        return raw, [paragraph for paragraph in paragraphs if paragraph], [], 0, source_hash
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        paragraphs = []
        removed_headers = 0
        for page in page_texts:
            lines = page.splitlines()
            removed_headers += sum(line.strip() == HEADER for line in lines)
            # The controlled sample has one paragraph per page; real PDFs need layout review.
            paragraph = normalize_lines([line for line in lines if line.strip() != HEADER])
            if paragraph:
                paragraphs.append(paragraph)
        return "\f".join(page_texts), paragraphs, page_texts, removed_headers, source_hash
    raise ValueError("입력은 TXT 또는 PDF여야 합니다")


def read_record(item: dict[str, object], input_dir: Path) -> dict[str, object]:
    record = dict(item)
    path = input_dir / str(record["source_path"])
    record.update(raw_text=None, clean_text="", paragraphs=[], sentences=[], page_texts=[],
                  removed_headers=0, sha256=None, duplicate_of=None, error=None)
    try:
        raw, paragraphs, page_texts, removed_headers, source_hash = read_source(path)
        sentences = [sentence for paragraph in paragraphs
                     for sentence in re.split(r"(?<=[.!?。！？])\s+", paragraph) if sentence]
        observed = "empty" if not paragraphs else "hold" if record["status"] == "hold" else "ready"
        record.update(raw_text=raw, clean_text="\n\n".join(paragraphs), paragraphs=paragraphs,
                      sentences=sentences, page_texts=page_texts, removed_headers=removed_headers,
                      sha256=source_hash)
    except (OSError, UnicodeError, PdfReadError, ValueError) as error:
        observed = "read_error"
        record["error"] = f"{type(error).__name__}: {error}"
    if observed != record["status"]:
        raise ValueError(f"{record['doc_id']}: manifest 상태 {record['status']}와 실제 상태 {observed}가 다릅니다")
    return record


def split_records(records: list[dict[str, object]]) -> dict[str, dict[str, list[str]]]:
    splits = {name: {"group_ids": [], "doc_ids": []} for name in SPLIT_BY_SUFFIX.values()}
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["status"] == "ready":
            groups[str(record["group_id"])].append(record)
    if len(groups) != 9:
        raise ValueError("라벨이 붙은 원문 묶음은 9개여야 합니다")
    for group_id, members in groups.items():
        if not re.fullmatch(r"G-(IT|ECO|SPT)-0[1-3]", group_id):
            raise ValueError(f"알 수 없는 원문 묶음: {group_id}")
        labels = {member["label"] for member in members}
        if labels != {LABEL_BY_PREFIX[group_id.split("-")[1]]}:
            raise ValueError(f"원문 묶음 안의 라벨이 일치하지 않습니다: {group_id}")
        name = SPLIT_BY_SUFFIX[group_id[-2:]]
        splits[name]["group_ids"].append(group_id)
        splits[name]["doc_ids"].extend(str(member["doc_id"]) for member in members)
    for name, part in splits.items():
        part["group_ids"].sort()
        if len(part["group_ids"]) != 3 or {groups[group][0]["label"] for group in part["group_ids"]} != LABELS:
            raise ValueError(f"{name}에 세 라벨의 원문 묶음이 하나씩 있어야 합니다")
    assigned = [group for part in splits.values() for group in part["group_ids"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("서로 다른 분할에 같은 원문 묶음이 있습니다")
    splits["excluded"] = {"group_ids": [], "doc_ids": [str(r["doc_id"]) for r in records if r["status"] != "ready"]}
    return splits


def array_report(records: list[dict[str, object]]) -> dict[str, object]:
    original = [record for record in records if record["status"] == "ready"
                and record["doc_id"] == str(record["group_id"])[2:]]
    if len(original) != 9:
        raise ValueError("원문 문서가 9개여야 합니다")
    lengths = [[len(paragraph) for paragraph in record["paragraphs"]] for record in original]
    small = np.asarray([[12, 8], [5, 7], [9, 4]], dtype=np.int64)
    source = {"doc_ids": [record["doc_id"] for record in original],
              "paragraph_lengths": lengths, "paragraph_counts": [len(row) for row in lengths],
              "matrix": None, "shape": None, "axis0_sum": None, "axis1_sum": None,
              "note": None}
    if all(len(row) == 2 for row in lengths):
        matrix = np.asarray(lengths, dtype=np.int64)
        source.update(matrix=matrix.tolist(), shape=list(matrix.shape),
                      axis0_sum=matrix.sum(axis=0).tolist(),
                      axis1_sum=matrix.sum(axis=1).tolist())
    else:
        source["note"] = "실습 원문 중 두 문단이 아닌 문서가 있어 9×2 배열을 만들지 않았습니다."
    return {
        "unit": "문자 수",
        "small_example": {"matrix": small.tolist(), "shape": list(small.shape),
                          "axis0_sum": small.sum(axis=0).tolist(), "axis1_sum": small.sum(axis=1).tolist()},
        "source_documents": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE,
                        help="manifest.json과 입력 TXT·PDF가 있는 디렉터리")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="결과 JSON을 쓸 디렉터리(기본값: _workspace/chapter2)")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [read_record(item, input_dir) for item in manifest]
    if (len(records) != 15 or len({r["doc_id"] for r in records}) != len(records)
            or len({r["source_path"] for r in records}) != len(records)):
        raise ValueError("manifest에는 서로 다른 문서 ID와 입력 경로 15개가 있어야 합니다")
    by_id = {str(record["doc_id"]): record for record in records}
    for derived_id, original_id in DERIVED_FROM.items():
        if derived_id not in by_id or original_id not in by_id:
            raise ValueError(f"파생 문서 또는 원문 ID가 없습니다: {derived_id} / {original_id}")
        if by_id[derived_id]["group_id"] != by_id[original_id]["group_id"]:
            raise ValueError(f"파생 문서와 원문의 묶음이 다릅니다: {derived_id} / {original_id}")
    first_hash: dict[str, dict[str, object]] = {}
    for record in records:
        source_hash = record["sha256"]
        if source_hash is None:
            continue
        if source_hash in first_hash:
            first = first_hash[source_hash]
            record["duplicate_of"] = first["doc_id"]
            if record["group_id"] != first["group_id"]:
                raise ValueError(f"완전 중복 문서의 원문 묶음이 다릅니다: {record['doc_id']}")
        else:
            first_hash[source_hash] = record
    splits = split_records(records)
    arrays = array_report(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("records.json", records), ("splits.json", splits), ("paragraph_lengths.json", arrays)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {status: sum(record["status"] == status for record in records)
              for status in ("ready", "hold", "empty", "read_error")}
    print(f"입력 {len(records)}건: 처리 {counts['ready']}, 보류 {counts['hold']}, 빈 입력 {counts['empty']}, 읽기 오류 {counts['read_error']}")
    for name in SPLIT_BY_SUFFIX.values():
        part = splits[name]
        print(f"{name}: 원문 묶음 {len(part['group_ids'])}개, 문서 {len(part['doc_ids'])}개")
    print("분할 간 원문 묶음 겹침: 0개")
    pdf = next(record for record in records if record["doc_id"] == "IT-02")
    print(f"IT-02 PDF: {len(pdf['page_texts'])}쪽, 반복 머리말 {pdf['removed_headers']}개 제거")
    print(f"완전 중복: " + ", ".join(f"{r['doc_id']}→{r['duplicate_of']}" for r in records if r["duplicate_of"]))
    print(f"NumPy 3×2 예제: axis=0 {arrays['small_example']['axis0_sum']}, axis=1 {arrays['small_example']['axis1_sum']}")
    if arrays["source_documents"]["matrix"] is None:
        print(f"실제 원문 배열: {arrays['source_documents']['note']}")
    else:
        print(f"실제 원문 배열: shape={arrays['source_documents']['shape']}")
    print(f"출력: {output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"자료 검사 오류: {error}") from error
