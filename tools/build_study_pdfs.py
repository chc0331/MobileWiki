#!/usr/bin/env python3
"""Build book-style PDF mirrors of all Markdown documents under study/."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re
import textwrap
from urllib.parse import quote

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study"
PLAN_SOURCE = STUDY / "README.md"
PLAN_PDF = STUDY / "학습 계획.pdf"
PAGE_W, PAGE_H = 182 * mm, 257 * mm
MARGIN = 52
CONTENT_W = PAGE_W - MARGIN * 2
INK = colors.HexColor("#243442")
NAVY = colors.HexColor("#183247")
TEAL = colors.HexColor("#167D8D")
CREAM = colors.HexColor("#F7F3EB")
PALE = colors.HexColor("#EEF5F5")
MUTED = colors.HexColor("#667783")
LINE = colors.HexColor("#D9E1E5")


def find_font(env_name: str, candidates: list[str]) -> str:
    if os.environ.get(env_name):
        candidates.insert(0, os.environ[env_name])
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(
        f"한글 TrueType 폰트를 찾을 수 없습니다. {env_name}에 .ttf 경로를 지정하세요."
    )


def register_fonts() -> None:
    serif = find_font("PDF_BODY_FONT", [
        "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
        "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ])
    sans = find_font("PDF_SANS_FONT", [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ])
    mono = find_font("PDF_CODE_FONT", [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ])
    pdfmetrics.registerFont(TTFont("KBody", serif))
    pdfmetrics.registerFont(TTFont("KSans", sans))
    pdfmetrics.registerFont(TTFont("Code", mono))
    pdfmetrics.registerFontFamily("KBody", normal="KBody", bold="KSans", italic="KBody", boldItalic="KSans")
    pdfmetrics.registerFontFamily("KSans", normal="KSans", bold="KSans", italic="KSans", boldItalic="KSans")
    rl_config.warnOnMissingFontGlyphs = 1


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "Body", fontName="KBody", fontSize=11, leading=18.5,
        textColor=INK, spaceAfter=9, wordWrap="CJK", splitLongWords=True,
        allowWidows=0, allowOrphans=0,
    )
    result = {"body": body}
    result["h2"] = ParagraphStyle(
        "H2", parent=body, fontName="KSans", fontSize=16.5, leading=23,
        textColor=NAVY, spaceBefore=22, spaceAfter=12, keepWithNext=True,
    )
    result["h3"] = ParagraphStyle(
        "H3", parent=body, fontName="KSans", fontSize=12.4, leading=19,
        textColor=TEAL, spaceBefore=17, spaceAfter=8, keepWithNext=True,
    )
    result["h4"] = ParagraphStyle(
        "H4", parent=body, fontName="KSans", fontSize=11.3, leading=18,
        textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=True,
    )
    result["cue"] = ParagraphStyle(
        "LearningCue", parent=result["h4"], backColor=PALE,
        borderPadding=6, spaceAfter=12,
    )
    result["list"] = ParagraphStyle(
        "List", parent=body, leftIndent=16, bulletIndent=1,
        bulletFontName="KSans", bulletFontSize=9, spaceAfter=5.5,
    )
    result["quote"] = ParagraphStyle(
        "Quote", parent=body, fontName="KSans", fontSize=10, leading=17,
        borderColor=TEAL, borderWidth=0.6, borderPadding=11,
        backColor=PALE, spaceBefore=8, spaceAfter=14,
    )
    result["caption"] = ParagraphStyle(
        "Caption", parent=body, fontName="KSans", fontSize=9,
        leading=14, textColor=MUTED, spaceBefore=6, spaceAfter=15,
    )
    result["table"] = ParagraphStyle(
        "Cell", parent=body, fontName="KSans", fontSize=9.5,
        leading=14.5, spaceBefore=0, spaceAfter=0, allowWidows=1, allowOrphans=1,
    )
    result["thead"] = ParagraphStyle(
        "TableHead", parent=result["table"], fontSize=9.5, textColor=colors.white,
    )
    result["toc0"] = ParagraphStyle(
        "TOC0", parent=body, fontName="KSans", fontSize=10.3, leading=17,
        spaceBefore=6, spaceAfter=5, leftIndent=0, firstLineIndent=0,
    )
    result["toc1"] = ParagraphStyle(
        "TOC1", parent=body, fontName="KSans", fontSize=9, leading=14,
        spaceBefore=2, spaceAfter=3, leftIndent=12, firstLineIndent=0,
        textColor=MUTED,
    )
    return result


def pdf_link(target: str) -> str:
    if "://" not in target:
        stem, *anchor = target.split("#", 1)
        if stem.endswith(".md"):
            stem = str(Path(stem).with_name("학습 계획.pdf")) if Path(stem).name == "README.md" else stem[:-3] + ".pdf"
        target = stem + (("#" + anchor[0]) if anchor else "")
    return quote(target, safe="/:#?=&%+@")


INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))")


def inline(text: str) -> str:
    def escaped(fragment):
        pieces, start = [], 0
        for tail in re.finditer(r"[^\s<>]{1,5}[’”]", fragment):
            pieces.append(html.escape(fragment[start:tail.start()]))
            pieces.append(f"<nobr>{html.escape(tail.group())}</nobr>")
            start = tail.end()
        pieces.append(html.escape(fragment[start:]))
        return "".join(pieces)

    parts, last = [], 0
    for match in INLINE.finditer(text):
        parts.append(escaped(text[last:match.start()]))
        token = match.group()
        if token.startswith("`"):
            # The SentencePiece whitespace marker is absent from the Korean font.
            code = html.escape(token[1:-1]).replace("▁", '<font name="Code">▁</font>')
            parts.append(f'<font name="KSans" color="#167D8D">{code}</font>')
        elif token.startswith("**"):
            parts.append(f'<b>{inline(token[2:-2])}</b>')
        else:
            label, target = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
            parts.append(f'<link href="{html.escape(pdf_link(target), quote=True)}" color="#167D8D">{html.escape(label)}</link>')
        last = match.end()
    parts.append(escaped(text[last:]))
    return "".join(parts)


def plain(text: str) -> str:
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip("*")


class Cover(Flowable):
    def __init__(self, source: Path, title: str):
        super().__init__()
        self.source, self.title = source, title
        self.width, self.height = CONTENT_W, PAGE_H - 112

    def draw(self):
        c = self.canv
        h, w = self.height, self.width
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(TEAL)
        c.drawString(0, h - 25, "MOBILEWIKI")
        c.setFont("KSans", 9)
        c.setFillColor(MUTED)
        c.drawString(0, h - 47, "온디바이스 문서 AI 학습 시리즈")
        is_lesson = self.source.parent.name.startswith("chapter")
        chapter = re.search(r"chapter(\d+)", self.source.parent.name)
        label = f"과목 {chapter[1]} / 개념과 실습" if chapter else "학습 로드맵 / 입문부터 중급까지"
        c.setFillColor(TEAL)
        c.roundRect(0, h - 103, w, 27, 4, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("KSans", 10)
        c.drawString(12, h - 94, label)
        subject = re.sub(r"^과목\s+\d+\s+", "", self.title)
        words = subject.split()
        if len(words) > 1:
            middle = min(range(1, len(words)), key=lambda n: max(
                pdfmetrics.stringWidth(" ".join(words[:n]), "KSans", 30),
                pdfmetrics.stringWidth(" ".join(words[n:]), "KSans", 30),
            ))
            title = html.escape(" ".join(words[:middle])) + "<br/>" + html.escape(" ".join(words[middle:]))
        else:
            title = html.escape(subject)
        p = Paragraph(title, ParagraphStyle(
            "CoverTitle", fontName="KSans", fontSize=30, leading=43,
            textColor=NAVY, wordWrap="CJK",
        ))
        _, ph = p.wrap(w, 200)
        p.drawOn(c, 0, h - 141 - ph)
        c.setFont("KBody", 12)
        c.setFillColor(INK)
        c.drawString(0, h - 279, "개념을 읽고, 예제로 이해하고, 직접 실습합니다.")
        labels = ["개념", "예제", "실습"] if is_lesson else ["입문", "초급", "중급"]
        step_w = (w - 32) / 3
        y = h - 376
        for i, item in enumerate(labels):
            x = i * (step_w + 16)
            c.setFillColor(colors.white)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, step_w, 63, 8, fill=1, stroke=1)
            c.setFont("Helvetica", 9)
            c.setFillColor(TEAL)
            c.drawString(x + 12, y + 43, f"0{i + 1}")
            c.setFillColor(NAVY)
            c.setFont("KSans", 16)
            c.drawString(x + 12, y + 17, item)
        c.setFillColor(NAVY)
        c.setFont("KSans", 10)
        c.drawString(0, h - 431, "이 책에서 다루는 내용")
        items = re.findall(r"^## (.+)$", self.source.read_text(encoding="utf-8"), re.M)[:3]
        next_y = h - 445
        for i, item in enumerate(items):
            item = re.sub(r"^\d+\.\s*", "", item)
            p = Paragraph(f"{i + 1}. {html.escape(item)}", ParagraphStyle(
                "CoverTopic", fontName="KBody", fontSize=10.5, leading=16,
                textColor=INK, wordWrap="CJK",
            ))
            _, ph = p.wrap(w, 100)
            next_y -= ph
            p.drawOn(c, 0, next_y)
            next_y -= 7
        c.setStrokeColor(TEAL)
        c.line(0, 32, w, 32)
        c.setFillColor(MUTED)
        c.setFont("KSans", 8)
        c.drawString(0, 13, "한국어 개념 설명 · 교육 도식 · 예제 · 실습")


class CodeBlock(Flowable):
    """Draw actual code with monospace Latin and a Korean glyph fallback."""

    def __init__(self, lines: list[str], language: str, continuation=False):
        super().__init__()
        self.lines, self.language, self.continuation = lines, language, continuation
        self.font_size, self.leading = 8.6, 13.2
        self.width = CONTENT_W
        self.prepared = self.wrap_lines(lines)
        self.height = 35 + len(self.prepared) * self.leading

    def char_font(self, char: str) -> str:
        return "Code" if ord(char) < 128 else "KSans"

    def text_width(self, text: str) -> float:
        return sum(pdfmetrics.stringWidth(ch, self.char_font(ch), self.font_size) for ch in text)

    def wrap_lines(self, lines):
        result = []
        for line in lines:
            current, width = "", 0
            for char in line.expandtabs(4):
                cw = pdfmetrics.stringWidth(char, self.char_font(char), self.font_size)
                if width + cw > self.width - 28 and current:
                    result.append(current)
                    current, width = "  ", self.text_width("  ")
                current += char
                width += cw
            result.append(current)
        return result

    def split(self, availWidth, availHeight):
        count = int((availHeight - 35) / self.leading)
        if count < 5:
            return []
        if count >= len(self.prepared):
            return [self]
        first = CodeBlock(self.prepared[:count], self.language, self.continuation)
        second = CodeBlock(self.prepared[count:], self.language, True)
        return [first, second]

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#F3F5F7"))
        c.roundRect(0, 0, self.width, self.height, 5, stroke=0, fill=1)
        c.setFillColor(TEAL)
        c.setFont("KSans", 8)
        if self.language == "python":
            label = "Python 실습"
        elif self.language in {"sh", "bash", "shell"}:
            label = "실행 명령"
        else:
            label = "예상 출력"
        if self.continuation:
            label += " / 계속"
        c.drawString(13, self.height - 17, label)
        for i, line in enumerate(self.prepared):
            x, y = 13, self.height - 34 - i * self.leading
            c.setFillColor(MUTED if line.lstrip().startswith("#") else INK)
            for font, chars in self.runs(line):
                c.setFont(font, self.font_size)
                c.drawString(x, y, chars)
                x += pdfmetrics.stringWidth(chars, font, self.font_size)

    def runs(self, text):
        result = []
        for ch in text:
            font = self.char_font(ch)
            if result and result[-1][0] == font:
                result[-1] = (font, result[-1][1] + ch)
            else:
                result.append((font, ch))
        return result


class StudyBook(BaseDocTemplate):
    def __init__(self, output: Path, source: Path, title: str):
        super().__init__(
            str(output), pagesize=(PAGE_W, PAGE_H),
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=56, bottomMargin=52,
            title=title, author="MobileWiki", pageCompression=1,
            allowSplitting=1,
        )
        self.source, self.book_title = source, title
        frame = Frame(MARGIN, 52, CONTENT_W, PAGE_H - 108,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="Book", frames=[frame], onPage=self.page_art))
        self.heading_counter = 0

    def beforeDocument(self):
        self.heading_counter = 0

    def page_art(self, c, doc):
        c.saveState()
        if doc.page == 1:
            c.setFillColor(CREAM)
            c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
            c.setFillColor(TEAL)
            c.rect(0, 0, 9, PAGE_H, stroke=0, fill=1)
        else:
            c.setFont("Helvetica-Bold", 7.8)
            c.setFillColor(TEAL)
            c.drawString(MARGIN, PAGE_H - 30, "MOBILEWIKI")
            c.setFont("KSans", 7.5)
            c.setFillColor(MUTED)
            c.drawRightString(PAGE_W - MARGIN, PAGE_H - 30, self.book_title)
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(MARGIN, PAGE_H - 39, PAGE_W - MARGIN, PAGE_H - 39)
            c.line(MARGIN, 40, PAGE_W - MARGIN, 40)
            c.setFont("KSans", 7.5)
            c.drawString(MARGIN, 25, "개념 · 예제 · 실습")
            c.setFont("Helvetica", 9)
            c.drawRightString(PAGE_W - MARGIN, 25, str(doc.page))
        c.restoreState()

    def afterFlowable(self, flowable):
        if not hasattr(flowable, "heading_level"):
            return
        self.heading_counter += 1
        key = f"section-{self.heading_counter}"
        text = flowable.heading_text
        self.canv.bookmarkPage(key)
        level = flowable.heading_level - 2
        if level <= 1:
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
        if flowable.toc_include:
            self.notify("TOCEntry", (0 if level == 0 else 1, text, self.page, key))


class StudyPlan(StudyBook):
    """An A4 learning plan; chapter lessons retain the book layout."""

    def __init__(self, output: Path, source: Path, title: str):
        BaseDocTemplate.__init__(
            self, str(output), pagesize=A4,
            title=title, author="MobileWiki", pageCompression=1,
        )
        self.source, self.book_title = source, title
        self.heading_counter = 0
        width, height = A4
        frame = Frame(48, 48, width - 96, height - 104,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="Plan", frames=[frame], onPage=self.page_art))

    def page_art(self, c, doc):
        width, height = A4
        c.saveState()
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(48, height - 30, "MOBILEWIKI")
        c.setFillColor(MUTED)
        c.setFont("KSans", 8)
        c.drawRightString(width - 48, height - 30, "학습 계획 · 기본 과정과 고급 예정 과정")
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(48, height - 39, width - 48, height - 39)
        c.line(48, 36, width - 48, 36)
        c.drawString(48, 22, "Chapter 1~14 · 개념 → 예제 → 실습 → 평가 → POC")
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 48, 22, str(doc.page))
        c.restoreState()


def plan_styles(st):
    result = {key: value.clone("Plan" + key) for key, value in st.items()}
    for key in ("body", "list"):
        result[key].fontSize = 11
        result[key].leading = 18
        result[key].spaceAfter = 8
    result["material"] = ParagraphStyle(
        "PlanMaterial", parent=result["body"], fontName="KSans", fontSize=9,
        leading=13, spaceAfter=6, textColor=MUTED, keepWithNext=True,
    )
    result["h1"] = ParagraphStyle(
        "PlanTitle", parent=result["body"], fontName="KSans", fontSize=22,
        leading=30, textColor=NAVY, spaceAfter=9, keepWithNext=True,
    )
    result["h2"].fontSize = 14.5
    result["h2"].leading = 21
    result["h2"].spaceBefore = 17
    result["h2"].spaceAfter = 8
    result["h3"].fontSize = 11.7
    result["h3"].leading = 17
    result["h3"].spaceBefore = 13
    result["h3"].spaceAfter = 7
    result["caption"].fontSize = 9
    result["caption"].leading = 14
    result["caption"].spaceBefore = 3
    result["caption"].spaceAfter = 6
    result["table"].fontSize = 9.5
    result["table"].leading = 14.5
    return result


def table_flow(rows: list[list[str]], st, width=CONTENT_W):
    headers, n = rows[0], len(rows[0])
    worksheet = "내 판단" in headers
    chapter2_worksheet = headers == ["확인할 것", "실행 전 예상", "실행 결과와 근거"]
    cluster_worksheet = headers == ["설정", "문서들을 묶은 결과", "대표 문서·핵심어·그룹 이름"]
    if cluster_worksheet:
        ratios = [0.12, 0.40, 0.48]
    elif worksheet:
        ratios = [0.07, 0.47, 0.15, 0.31]
    elif chapter2_worksheet:
        ratios = [0.36, 0.21, 0.43]
    elif headers == ["구분", "배터리 수명 안내", "경기 결과 안내", "왜 다른가요?"]:
        ratios = [0.20, 0.21, 0.21, 0.38]
    elif headers == ["중간 칸", "입력에 곱할 가중치", "편향", "계산 결과"]:
        ratios = [0.14, 0.28, 0.12, 0.46]
    elif headers == ["입력 `[x1, x2]`", "`[h1, h2]`", "합", "결과"]:
        ratios = [0.30, 0.30, 0.15, 0.25]
    elif headers == ["부분", "가중치 수", "편향 수", "합계"]:
        ratios = [0.36, 0.30, 0.17, 0.17]
    elif headers == ["문장", "글자 수", "UTF-8 바이트 수", "토큰 수"]:
        ratios = [0.58, 0.12, 0.18, 0.12]
    elif headers == ["참조할 토큰", "가져올 정보 벡터", "혼합 비중", "곱한 값"]:
        ratios = [0.21, 0.30, 0.19, 0.30]
    elif headers == ["단계", "이미 만든 출력", "후보 값의 예", "선택"]:
        ratios = [0.10, 0.25, 0.50, 0.15]
    elif headers == ["조각", "IT", "경제", "스포츠"]:
        ratios = [0.25, 0.25, 0.25, 0.25]
    elif n == 4:
        ratios = [0.09, 0.34, 0.15, 0.42]
    elif n == 6:
        ratios = [0.20, 0.16, 0.16, 0.16, 0.16, 0.16]
    elif n == 3 and headers[1] == "예시 판단":
        ratios = [0.22, 0.18, 0.60]
    elif n == 3 and headers[0] == "새 입력":
        ratios = [0.52, 0.23, 0.25] if headers[1] == "점수 합" else [0.30, 0.22, 0.48]
    elif n == 3 and headers[0] == "번호":
        ratios = [0.11, 0.72, 0.17]
    elif n == 3 and headers[-1] in ("라벨", "정답"):
        ratios = [0.66, 0.17, 0.17]
    elif n == 3:
        ratios = [0.25, 0.38, 0.37]
    elif n == 2 and headers[0] == "상황":
        ratios = [0.58, 0.42]
    elif n == 2 and headers == ["필드", "뜻"]:
        ratios = [0.43, 0.57]
    elif n == 2 and headers[-1] == "라벨":
        ratios = [0.80, 0.20]
    elif n == 2:
        ratios = [0.32, 0.68]
    else:
        ratios = [1 / n] * n
    data = [[Paragraph(inline(cell).replace('color="#167D8D"', 'color="#FFFFFF"')
                       if r == 0 else inline(cell), st["thead"] if r == 0 else st["table"])
             for cell in row] for r, row in enumerate(rows)]
    empty_writing = n == 2 and headers[-1] == "내가 정한 내용"
    row_height = 86 if cluster_worksheet else (65 if chapter2_worksheet else (43 if worksheet or empty_writing else 0))
    minimum = [0] + [row_height] * (len(rows) - 1)
    table = LongTable(
        data, colWidths=[width * ratio for ratio in ratios],
        repeatRows=1, hAlign="LEFT", splitByRow=1, splitInRow=0,
        minRowHeights=minimum,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
    ]))
    if worksheet:
        table.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (0, -1), 4),
            ("RIGHTPADDING", (0, 0), (0, -1), 4),
        ]))
    return table


def is_special(line: str) -> bool:
    return bool(re.match(r"^(#{1,4} |```|\| |>|!\[|- |\d+\. )", line))


def parse_markdown(source: Path, st, *, plan=False, width=CONTENT_W):
    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    units = []
    is_lesson = source.parent.name.startswith("chapter")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        heading = re.match(r"^(#{1,4}) (.+)$", line)
        if heading:
            level, title = len(heading[1]), heading[2]
            if level == 1 and not plan:
                i += 1
                continue
            previous_heading = bool(story and getattr(story[-1], "is_section_heading", False))
            new_part = any(part in title for part in (
                "실습 1 ", "확인 문제", "선택 실습", "완료 확인",
            ))
            if is_lesson and level == 2 and new_part:
                # Avoid leaving only a short closing paragraph before the questions.
                story.append(CondPageBreak(340) if "확인 문제" in title else PageBreak())
            elif not previous_heading and level > 1:
                story.append(CondPageBreak((150 if level == 2 else 90) if plan else (100 if level == 2 else 70)))
            learning_cue = level == 4 and title.startswith((
                "핵심 질문", "함께 풀기", "직접 해보기", "직접 해보고", "결과와 해석", "핵심 정리",
            ))
            p = Paragraph(inline(title), st["cue"] if learning_cue else st[f"h{level}"])
            p.is_section_heading = True
            if level in (2, 3):
                p.heading_level = level
                p.heading_text = plain(title)
                p.toc_include = level == 2
            story.append(p)
            units.append(plain(title))
            i += 1
            continue
        if line.startswith("```"):
            language = line[3:].strip()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            block = CodeBlock(code, language)
            before = Spacer(1, 6)
            if block.height <= 340:
                story.append(CondPageBreak(block.height + 18))
            story.extend([before, block, Spacer(1, 12)])
            units.extend(item for item in code if item.strip())
            i += 1
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)$", line)
        if image:
            image_path = source.parent / image[2]
            with PILImage.open(image_path) as im:
                image_w, image_h = im.size
            ih = width * image_h / image_w
            block = [Image(str(image_path), width=width, height=ih)]
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r"^\*그림 .+\*$", lines[j]):
                caption = lines[j][1:-1]
                block.append(Paragraph(inline(caption), st["caption"]))
                units.append(plain(caption))
                i = j + 1
            else:
                i += 1
            # Keep the short calculation immediately before a figure on its page.
            if story and getattr(story[-1], "short_list", False):
                block = list(story.pop()._content) + [Spacer(1, 6)] + block
            before = Spacer(1, 8)
            before.keepWithNext = True
            story.extend([before, KeepTogether(block)])
            continue
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                    units.extend(plain(cell) for cell in cells if cell)
                i += 1
            table = table_flow(rows, st, width)
            keep_table = len(rows) <= 7 and table.wrap(width, 10000)[1] <= 220
            before = Spacer(1, 4)
            before.keepWithNext = keep_table
            story.extend([before, KeepTogether([table]) if keep_table else table, Spacer(1, 11)])
            continue
        if line.startswith(">"):
            quotes = []
            while i < len(lines) and lines[i].startswith(">"):
                quotes.append(lines[i].lstrip("> "))
                i += 1
            text = " ".join(quotes)
            story.append(Paragraph(inline(text), st["quote"]))
            units.append(plain(text))
            continue
        item = re.match(r"^(- |\d+\. )(.+)$", line)
        if item:
            items = []
            numbered = item[1] != "- "
            while i < len(lines):
                item = re.match(r"^(- |\d+\. )(.+)$", lines[i])
                if not item or (item[1] != "- ") != numbered:
                    break
                marker, text = item.groups()
                if text.startswith("[ ] "):
                    text, marker = text[4:], "☐"
                else:
                    marker = "•" if marker == "- " else marker.strip()
                items.append(Paragraph(inline(text), st["list"], bulletText=marker))
                units.append(plain(text))
                i += 1
            height = sum(p.wrap(width, 10000)[1] + p.getSpaceBefore() + p.getSpaceAfter()
                         for p in items)
            if len(items) <= 4 and height <= 220:
                block = KeepTogether(items)
                block.short_list = True
                story.append(block)
            else:
                story.extend(KeepTogether([p]) for p in items)
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_special(lines[i]):
            paragraph.append(lines[i])
            i += 1
        text = " ".join(paragraph)
        if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
            text = text[1:-1]
        style = st["material"] if plan and text.startswith(("**교재 PDF:**", "**예정 PDF:**")) else st["body"]
        p = Paragraph(inline(text), style)
        standalone_label = bool(re.fullmatch(r"\*\*[^*]+\*\*", text))
        if text.endswith(":") or re.match(r"^\*\*\d+\. 질문:\*\*", text) or standalone_label:
            p.keepWithNext = True
        if standalone_label:
            p.is_section_heading = True
        height = p.wrap(width, 10000)[1]
        if not getattr(p, "keepWithNext", False) and height <= style.leading * 5:
            story.append(KeepTogether([p]))
        else:
            story.append(p)
        units.append(plain(text))
    return story, units


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("\u200b", "")


def bind_heading_blocks(story):
    """Keep a heading and an already grouped table/checklist as one flat group."""
    result = []
    for flow in story:
        if isinstance(flow, KeepTogether):
            start = len(result)
            while start and (isinstance(result[start - 1], Spacer) or
                             getattr(result[start - 1], "is_section_heading", False)):
                start -= 1
            tail = result[start:]
            if any(getattr(item, "is_section_heading", False) for item in tail):
                result[start:] = [KeepTogether(tail + list(flow._content))]
                continue
        result.append(flow)
    return result


def verify_pdf(output: Path, units: list[str]) -> tuple[int, int]:
    reader = PdfReader(output)
    frontmatter_pages = 0 if output == PLAN_PDF else 2
    # Cover and contents are extra navigation. Remove running furniture before
    # checking source elements in order, including repeated section labels.
    extracted = "\n".join("\n".join((page.extract_text() or "").splitlines()[4:]) for page in reader.pages[frontmatter_pages:])
    normalized = normalize(extracted)
    missing, cursor = [], 0
    for unit in units:
        needle = normalize(unit)
        position = normalized.find(needle, cursor)
        if position < 0:
            missing.append(unit)
        else:
            cursor = position + len(needle)
    if missing:
        samples = "\n".join(textwrap.shorten(x, width=150) for x in missing[:8])
        raise RuntimeError(f"PDF 본문 대응 확인 실패: {output}\n{samples}")
    for page in reader.pages:
        for ref in page.get("/Annots", []):
            action = ref.get_object().get("/A")
            if not action or "/URI" not in action:
                continue
            target = str(action["/URI"])
            if "://" not in target and not target.startswith("#"):
                from urllib.parse import unquote
                local = output.parent / unquote(target.split("#", 1)[0])
                if not local.exists():
                    raise RuntimeError(f"PDF 링크 대상 없음: {target}")
    return len(reader.pages), len(units)


def build(source: Path, st):
    text = source.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    is_plan = source == PLAN_SOURCE
    output = PLAN_PDF if is_plan else source.with_suffix(".pdf")
    if is_plan:
        st = plan_styles(st)
    content, units = parse_markdown(source, st, plan=is_plan, width=A4[0] - 96 if is_plan else CONTENT_W)
    content = bind_heading_blocks(content)
    if is_plan:
        StudyPlan(output, source, title).multiBuild(content)
        return output, units
    toc = TableOfContents()
    toc.levelStyles = [st["toc0"], st["toc1"]]
    toc.dotsMinLevel = 0
    story = [
        Cover(source, title), PageBreak(),
        Paragraph("차례", st["h2"]),
        Paragraph("학습 순서에 따라 읽거나, 항목을 눌러 해당 페이지로 이동하세요.", st["caption"]),
        toc, PageBreak(),
    ] + content
    doc = StudyBook(output, source, title)
    doc.multiBuild(story)
    return output, units


def main():
    register_fonts()
    st = styles()
    sources = sorted(STUDY.rglob("*.md"))
    builds = [build(source, st) for source in sources]
    # All mirrors exist before checking relative links between the PDFs.
    for output, units in builds:
        pages, count = verify_pdf(output, units)
        print(f"{output.relative_to(ROOT)}: {pages}쪽, 원문 요소 {count}개 대응 확인")


if __name__ == "__main__":
    main()
