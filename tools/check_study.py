#!/usr/bin/env python3
"""Check study Markdown links, PDF pairs, and roadmap registration without writing files."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CHAPTER = re.compile(r"^## Chapter ([1-9]\d*)\. (.+)$")
LESSON_TITLE = re.compile(r"^# 과목 ([1-9]\d*) (.+)$")
EXTERNAL = re.compile(r"^[A-Za-z][A-Za-z\d+.-]*:")
RAW_EXTERNAL = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>]+")


def location(root: Path, path: Path, line: int) -> str:
    try:
        name = path.relative_to(root)
    except ValueError:
        name = path
    return f"{name}:{line}"


def visible_lines(source: Path) -> Iterator[tuple[int, str]]:
    fenced = False
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        yield number, re.sub(r"`[^`]*`", "", line)


def has_link_to(source: Path, target: Path) -> bool:
    for _, line in visible_lines(source):
        for match in LINK.finditer(line):
            if match.group().startswith("!"):
                continue
            file_part = unquote(match.group(1).split("#", 1)[0].split("?", 1)[0])
            if file_part and (source.parent / file_part).resolve() == target.resolve():
                return True
    return False


def check_links(root: Path, source: Path, problems: list[str]) -> None:
    for number, line in visible_lines(source):
        for match in RAW_EXTERNAL.finditer(LINK.sub("", line)):
            problems.append(f"{location(root, source, number)}: 외부 주소를 학습 문서에 넣을 수 없습니다: {match.group()}")
        for match in LINK.finditer(line):
            target = match.group(1)
            where = location(root, source, number)
            if EXTERNAL.match(target) or target.startswith("//"):
                problems.append(f"{where}: 외부 링크를 학습 문서에 넣을 수 없습니다: {target}")
                continue
            if target.startswith("/"):
                problems.append(f"{where}: 절대 경로 대신 상대 링크를 사용하세요: {target}")
                continue
            file_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if file_part and not (source.parent / file_part).is_file():
                problems.append(f"{where}: 링크 대상 파일이 없습니다: {target}")


def check_pairs(root: Path, study: Path, problems: list[str]) -> None:
    for source in sorted(study.rglob("*.md")):
        pdf = (study / "학습 계획.pdf") if source == study / "README.md" else source.with_suffix(".pdf")
        if not pdf.is_file():
            problems.append(f"{location(root, source, 1)}: 짝이 되는 PDF가 없습니다: {pdf.name}")
    for pdf in sorted(study.rglob("*.pdf")):
        if pdf.parent.name == "examples" and re.fullmatch(r"chapter[1-9]\d*", pdf.parent.parent.name):
            continue
        source = (study / "README.md") if pdf == study / "학습 계획.pdf" else pdf.with_suffix(".md")
        if not source.is_file():
            problems.append(f"{location(root, pdf, 1)}: 짝이 되는 Markdown이 없습니다: {source.name}")


def check_roadmap(root: Path, study: Path, problems: list[str]) -> None:
    roadmap = study / "README.md"
    if not roadmap.is_file():
        problems.append(f"{location(root, roadmap, 1)}: 전체 학습 로드맵이 없습니다")
        return

    lines = roadmap.read_text(encoding="utf-8").splitlines()
    chapters: dict[int, tuple[str, int, list[tuple[int, str]]]] = {}
    current: int | None = None
    for number, line in enumerate(lines, 1):
        if line.startswith("## "):
            match = CHAPTER.fullmatch(line)
            current = int(match.group(1)) if match else None
            if line.startswith("## Chapter ") and not match:
                problems.append(f"{location(root, roadmap, number)}: '## Chapter 1. 과목명' 형식으로 쓰세요")
            if match:
                if current in chapters:
                    problems.append(f"{location(root, roadmap, number)}: Chapter {current}가 중복됐습니다")
                else:
                    chapters[current] = (match.group(2).strip(), number, [])
        elif current is not None and line.startswith(("**교재 PDF:**", "**예정 PDF:**")):
            chapters[current][2].append((number, line))

    if not chapters:
        problems.append(f"{location(root, roadmap, 1)}: Chapter 목차가 없습니다")
    elif sorted(chapters) != list(range(1, max(chapters) + 1)):
        problems.append(f"{location(root, roadmap, 1)}: Chapter 번호는 1부터 빈 번호 없이 이어져야 합니다")

    for number, (title, heading_line, materials) in sorted(chapters.items()):
        stem = title.replace(" ", "_")
        folder = study / f"chapter{number}"
        source = folder / f"{stem}.md"
        pdf = folder / f"{stem}.pdf"
        if folder.is_dir():
            if not source.is_file():
                problems.append(f"{location(root, roadmap, heading_line)}: 과목 파일명이 Chapter 제목과 다릅니다: {source.name}")
            else:
                if not has_link_to(roadmap, source):
                    problems.append(f"{location(root, roadmap, heading_line)}: 과목 Markdown 링크가 없습니다: chapter{number}/{stem}.md")
                first_line, first = next(
                    ((i, line) for i, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1) if line.strip()),
                    (1, ""),
                )
                match = LESSON_TITLE.fullmatch(first)
                if not match or int(match.group(1)) != number or match.group(2).strip() != title:
                    problems.append(f"{location(root, source, first_line)}: 첫 제목은 '# 과목 {number} {title}'이어야 합니다")
                if not has_link_to(source, roadmap):
                    problems.append(f"{location(root, source, first_line)}: 전체 로드맵 링크가 없습니다: ../README.md")
        if len(materials) != 1:
            problems.append(f"{location(root, roadmap, heading_line)}: 교재 PDF 등록 줄이 하나여야 합니다")
            continue

        material_line, material = materials[0]
        where = location(root, roadmap, material_line)
        if pdf.is_file():
            link = re.fullmatch(r"\*\*교재 PDF:\*\* \[[^\]]+\]\(([^)]+)\)", material)
            expected = f"chapter{number}/{stem}.pdf"
            if not link or unquote(link.group(1)) != expected or "학습 전" in material:
                problems.append(f"{where}: 등록된 PDF는 교재 PDF 링크로 표시하세요: {expected}")
        elif (not material.startswith("**예정 PDF:**")
              or f"`{stem}.pdf`" not in material
              or "**학습 전**" not in material
              or LINK.search(material)):
            problems.append(f"{where}: 없는 PDF는 예정 파일명과 학습 전으로 표시하세요: {stem}.pdf")

    for folder in sorted(study.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("chapter"):
            continue
        match = re.fullmatch(r"chapter([1-9]\d*)", folder.name)
        if not match:
            problems.append(f"{location(root, folder, 1)}: 디렉터리는 chapter1처럼 0 없이 이름 붙이세요")
        elif int(match.group(1)) not in chapters:
            problems.append(f"{location(root, folder, 1)}: 로드맵에 없는 Chapter 디렉터리입니다")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="검사할 저장소 루트(기본값: 이 스크립트의 상위 디렉터리)")
    root = parser.parse_args().root.resolve()
    study = root / "study"
    if not study.is_dir():
        print(f"{location(root, study, 1)}: study 디렉터리가 없습니다")
        return 1

    problems: list[str] = []
    for source in sorted(study.rglob("*.md")):
        check_links(root, source, problems)
    check_pairs(root, study, problems)
    check_roadmap(root, study, problems)
    if problems:
        for problem in problems:
            print(problem)
        print(f"학습 자료 검사 실패: {len(problems)}건")
        return 1
    print("학습 자료 검사 통과: 링크·PDF 짝·Chapter 등록 상태 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
