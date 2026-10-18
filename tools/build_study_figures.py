#!/usr/bin/env python3
"""Build the original instructional figures shared by the study MD and PDF."""

from pathlib import Path
import os

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = os.environ.get("CJK_FONT", "/System/Library/Fonts/AppleSDGothicNeo.ttc")
FONT_BOLD_INDEX = 6 if "CJK_FONT" not in os.environ else 0
W = 1600
NAVY = "#183247"
TEAL = "#167D8D"
CREAM = "#F7F3EB"
INK = "#243442"
MUTED = "#516674"
LIGHT = "#E8F3F3"
LINE = "#C9D6DA"
WHITE = "#FFFFFF"


def font(size=40, bold=False):
    return ImageFont.truetype(FONT_PATH, size=size, index=FONT_BOLD_INDEX if bold else 0)


def canvas(height=1000):
    image = Image.new("RGB", (W, height), WHITE)
    return image, ImageDraw.Draw(image)


def txt(draw, xy, text, size=40, bold=False, fill=INK, **kwargs):
    draw.text(xy, text, font=font(size, bold), fill=fill, **kwargs)


def wrapped_lines(draw, text, width, size=40, bold=False):
    result = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font(size, bold)) <= width:
                current = candidate
                continue
            if current:
                result.append(current)
                current = ""
            for char in word:
                if draw.textlength(current + char, font=font(size, bold)) > width:
                    result.append(current)
                    current = ""
                current += char
        result.append(current)
    return result


def paragraph(draw, box, text, size=40, bold=False, fill=INK, align="left", line_height=None):
    x, y, width = box
    step = line_height or round(size * 1.35)
    lines = wrapped_lines(draw, text, width, size, bold)
    for line in lines:
        line_x = x if align == "left" else x + width / 2
        txt(draw, (line_x, y), line, size, bold, fill, anchor="mt" if align == "center" else "lt")
        y += step
    return y


def card(draw, box, fill=CREAM, outline=None, radius=20, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=TEAL, width=7):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2 - 16, y2), fill=color, width=width)
    draw.polygon(((x2, y2), (x2 - 23, y2 - 15), (x2 - 23, y2 + 15)), fill=color)


def heading(draw, text):
    txt(draw, (64, 52), text, 52, True, NAVY, anchor="lt")
    draw.line((64, 120, 1536, 120), fill=LINE, width=3)


def document_icon(draw, x, y, width=72, height=86, color=TEAL):
    draw.rounded_rectangle((x, y, x + width, y + height), radius=7, outline=color, width=5)
    for dy in (24, 42, 60):
        draw.line((x + 15, y + dy, x + width - 15, y + dy), fill=color, width=4)


def phone_icon(draw, x, y, width=82, height=120, color=TEAL):
    draw.rounded_rectangle((x, y, x + width, y + height), radius=13, outline=color, width=5)
    draw.line((x + 23, y + 13, x + width - 23, y + 13), fill=color, width=4)
    draw.line((x + 28, y + height - 13, x + width - 28, y + height - 13), fill=color, width=4)


def computer_icon(draw, x, y, color=TEAL):
    draw.rounded_rectangle((x, y, x + 128, y + 83), radius=7, outline=color, width=5)
    draw.line((x + 64, y + 83, x + 64, y + 108), fill=color, width=5)
    draw.line((x + 32, y + 108, x + 96, y + 108), fill=color, width=5)


def save(image, relative_path):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    print(path.relative_to(ROOT))


def learning_path():
    image, draw = canvas(240)
    stages = [
        ("Chapter 1~3", "첫 오프라인 앱", "입문 → 초급"),
        ("Chapter 4~7", "추가학습·경량화", "초급 → 중급"),
        ("Chapter 8", "생성형 요약 확장", "중급 · 선택 심화"),
    ]
    width, gap = 496, 56
    for index, (chapter, title, level) in enumerate(stages):
        x = index * (width + gap)
        card(draw, (x, 8, x + width, 232), fill=LIGHT if index == 0 else CREAM)
        txt(draw, (x + 28, 32), chapter, 38, True, TEAL, anchor="lt")
        txt(draw, (x + 28, 92), title, 44, True, NAVY, anchor="lt")
        txt(draw, (x + 28, 164), level, 36, False, MUTED, anchor="lt")
        if index < 2:
            arrow(draw, (x + width + 8, 120), (x + width + gap - 8, 120), width=5)
    save(image, "study/assets/learning_path.png")


def app_flow():
    image, draw = canvas(900)
    heading(draw, "문서가 앱의 결과로 바뀌는 과정")
    labels = ["파일 선택", "텍스트 추출", "계산 가능한\n입력으로 변환", "분류·요약", "결과 표시"]
    gap = 36
    width = (1472 - gap * 4) / 5
    for i, label in enumerate(labels):
        x = 64 + i * (width + gap)
        card(draw, (x, 244, x + width, 622), fill=LIGHT if i == 3 else CREAM)
        txt(draw, (x + 24, 266), str(i + 1), 38, True, TEAL, anchor="lt")
        if i in (0, 1):
            document_icon(draw, x + width / 2 - 36, 334)
        elif i == 4:
            phone_icon(draw, x + width / 2 - 36, 316, width=72, height=110)
        elif i == 2:
            for j, length in enumerate((110, 78, 134)):
                draw.line((x + 64, 353 + j * 22, x + 64 + length, 353 + j * 22), fill=TEAL, width=9)
        else:
            draw.rounded_rectangle((x + 64, 332, x + width - 64, 422), radius=12, fill=TEAL)
            txt(draw, (x + width / 2, 377), "AI", 46, True, WHITE, anchor="mm")
        paragraph(draw, (x + 15, 476, width - 30), label, 40, True, NAVY, align="center", line_height=54)
        if i < 4:
            arrow(draw, (x + width + 5, 426), (x + width + gap - 4, 426), width=5)
    txt(draw, (64 + (width + gap) + width / 2, 698), "글자를 꺼내기", 38, True, TEAL, anchor="mt")
    txt(draw, (64 + 3 * (width + gap) + width / 2, 698), "내용을 분석하기", 38, True, TEAL, anchor="mt")
    txt(draw, (800, 812), "텍스트 추출과 내용 분석은 서로 다른 작업입니다.", 40, False, NAVY, anchor="mm")
    save(image, "study/chapter1/assets/app_flow.png")


def ai_ml_dl():
    image, draw = canvas()
    heading(draw, "AI · 머신러닝 · 딥러닝의 포함 관계")
    card(draw, (64, 160, 1536, 940), fill=WHITE, outline=NAVY, radius=26, width=5)
    txt(draw, (108, 198), "AI 인공지능", 48, True, NAVY, anchor="lt")
    txt(draw, (108, 270), "판단·인식·언어 처리 등을 다루는 넓은 분야", 42, False, INK, anchor="lt")
    txt(draw, (108, 342), "모든 AI가 딥러닝인 것은 아닙니다.", 38, False, MUTED, anchor="lt")
    card(draw, (376, 424, 1496, 904), fill=CREAM, outline=TEAL, radius=22, width=4)
    txt(draw, (418, 461), "ML 머신러닝", 46, True, NAVY, anchor="lt")
    txt(draw, (418, 528), "데이터에서 예측 기준을 얻는 방법", 40, False, INK, anchor="lt")
    card(draw, (686, 638, 1458, 866), fill=LIGHT, outline=TEAL, radius=18, width=3)
    txt(draw, (726, 673), "DL 딥러닝", 44, True, NAVY, anchor="lt")
    paragraph(draw, (726, 738, 690), "여러 층의 신경망을 학습하는 방식", 40, False, INK, line_height=52)
    save(image, "study/chapter1/assets/ai_ml_dl.png")


def training_inference():
    image, draw = canvas()
    heading(draw, "학습 · 추론 · 평가의 차이")
    rows = [
        ("학습", "정답이 붙은 문서\n+ 학습 절차", "학습된 모델"),
        ("추론", "새 문서\n+ 학습된 모델", "예측"),
        ("평가", "예측\n+ 따로 보관한 정답", "결과 비교"),
    ]
    for row, (title, source, result) in enumerate(rows):
        y = 168 + row * 238
        card(draw, (64, y, 244, y + 194), fill=NAVY)
        txt(draw, (154, y + 97), title, 46, True, WHITE, anchor="mm")
        card(draw, (286, y, 900, y + 194), fill=CREAM)
        paragraph(draw, (318, y + 43, 550), source, 42, False, INK, align="center", line_height=59)
        arrow(draw, (926, y + 97), (1006, y + 97))
        card(draw, (1032, y, 1536, y + 194), fill=LIGHT)
        txt(draw, (1284, y + 97), result, 44, True, NAVY, anchor="mm")
    txt(draw, (800, 939), "일반적인 추론에서는 모델의 숫자를 바꾸지 않습니다.", 40, True, NAVY, anchor="mm")
    save(image, "study/chapter1/assets/training_inference.png")


def document_tasks():
    image, draw = canvas()
    heading(draw, "같은 원문에서 세 가지 다른 결과")
    card(draw, (64, 160, 1536, 362), fill=CREAM)
    txt(draw, (94, 181), "원문", 38, True, TEAL, anchor="lt")
    paragraph(draw, (94, 239, 1412), "앱 업데이트는 금요일에 배포된다.\nPDF 요약 기능이 추가된다. 설정 아이콘도 바뀐다.", 42, False, INK, line_height=57)
    columns = [
        ("분류", "정해진 종류 선택", "IT"),
        ("추출 요약", "원문 문장 선택", "앱 업데이트는 금요일에 배포된다. PDF 요약 기능이 추가된다."),
        ("생성형 요약", "새 문장 작성", "금요일 앱 업데이트에 PDF 요약 기능이 추가된다."),
    ]
    width = 464
    for i, (title, action, output) in enumerate(columns):
        x = 64 + i * 504
        card(draw, (x, 420, x + width, 874), fill=LIGHT if i == 2 else WHITE, outline=LINE)
        txt(draw, (x + 26, 449), title, 44, True, NAVY, anchor="lt")
        txt(draw, (x + 26, 517), action, 38, False, TEAL, anchor="lt")
        draw.line((x + 26, 571, x + width - 26, 571), fill=LINE, width=2)
        paragraph(draw, (x + 26, 607, width - 52), output, 40, i == 0, INK, line_height=53)
    txt(draw, (800, 940), "표현이 자연스러워도 원문의 사실과 맞는지 확인합니다.", 40, True, NAVY, anchor="mm")
    save(image, "study/chapter1/assets/document_tasks.png")


def on_device_flow():
    image, draw = canvas(900)
    heading(draw, "PC에서 학습하고 휴대폰에서 추론하기")
    positions = [(64, 494), (584, 1014), (1104, 1536)]
    for i, (x1, x2) in enumerate(positions):
        card(draw, (x1, 212, x2, 668), fill=LIGHT if i == 2 else CREAM)
        center = (x1 + x2) / 2
        if i == 0:
            computer_icon(draw, center - 64, 268)
        elif i == 1:
            document_icon(draw, center - 40, 264, width=80, height=110)
        else:
            phone_icon(draw, center - 40, 252, width=80, height=124)
        title = ["PC·Colab", "모델 파일", "Android 앱"][i]
        detail = ["문서와 정답으로\n학습", "학습된 숫자와\n실행에 필요한 정보", "새 문서를\n기기에서 분석"][i]
        txt(draw, (center, 428), title, 46, True, NAVY, anchor="mt")
        paragraph(draw, (x1 + 26, 510, x2 - x1 - 52), detail, 40, False, INK, align="center", line_height=56)
        if i < 2:
            arrow(draw, (x2 + 13, 432), (positions[i + 1][0] - 13, 432))
    txt(draw, (800, 741), "학습 결과를 앱에 배포해 기기에서 실행합니다.", 40, True, NAVY, anchor="mm")
    txt(draw, (800, 825), "모델 다운로드와 사용자 문서 분석은 별개의 과정입니다.", 38, False, MUTED, anchor="mm")
    save(image, "study/chapter1/assets/on_device_flow.png")


if __name__ == "__main__":
    learning_path()
    app_flow()
    ai_ml_dl()
    training_inference()
    document_tasks()
    on_device_flow()
