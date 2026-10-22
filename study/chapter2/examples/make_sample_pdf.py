#!/usr/bin/env python3
"""Regenerate the input PDF with reportlab from tools/requirements.txt."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "it02_ko.pdf"
HEADER = "IT-02 | LEARNING SAMPLE"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
)
PAGES = (
    (
        "개발팀은 오래된 휴대전화에서도 메모 앱이 빨리 열리도록",
        "초기 로딩을 줄였다. 실험에는 같은 기기와 메모 100개를 사용했다.",
    ),
    (
        "앱이 첫 화면을 여는 평균 시간은 1.8초에서 1.2초로 줄었다.",
        "저장 속도는 이번 실험에서 측정하지 않았다.",
    ),
)


def main() -> None:
    font = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if font is None:
        raise SystemExit("한글 TrueType 글꼴을 찾을 수 없습니다.")
    pdfmetrics.registerFont(TTFont("SampleKorean", str(font)))
    _, page_height = A4
    document = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=0)
    document.setTitle("IT-02 학습용 입력 문서")
    for lines in PAGES:
        document.setFont("Helvetica", 9)
        document.drawString(55, page_height - 55, HEADER)
        document.setFont("SampleKorean", 12)
        for index, line in enumerate(lines):
            document.drawString(55, page_height - 105 - 25 * index, line)
        document.showPage()
    document.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
