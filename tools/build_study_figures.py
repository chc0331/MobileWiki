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
    image, draw = canvas(310)
    stages = [
        ("Chapter 1~6", "기초·분류\n설계 학습", "이론·실습·평가"),
        ("POC A", "분류 모델\n설계 검증", "PC 시제품"),
        ("Chapter 7", "요약 모델\n설계 학습", "이론·실습·평가"),
        ("POC B", "요약 모델\n설계 검증", "PC 시제품"),
        ("Chapter 8", "Android\n배포 학습", "이론·실습·평가"),
        ("POC C", "Android\n통합 검증", "오프라인 앱"),
    ]
    width, gap = 240, 32
    for index, (chapter, title, level) in enumerate(stages):
        x = index * (width + gap)
        card(draw, (x, 8, x + width, 302), fill=LIGHT if index % 2 == 0 else CREAM)
        txt(draw, (x + 18, 34), chapter, 32, True, TEAL, anchor="lt")
        paragraph(draw, (x + 18, 105, width - 36), title, 36, True, NAVY, line_height=49)
        txt(draw, (x + 18, 247), level, 32, False, MUTED, anchor="lt")
        if index < len(stages) - 1:
            arrow(draw, (x + width + 4, 155), (x + width + gap - 4, 155), width=5)
    save(image, "study/assets/learning_path.png")


def model_flow():
    image, draw = canvas(620)
    heading(draw, "‘배터리 수명 안내’가 IT로 분류되는 과정")
    labels = ["문서 입력", "숫자로 바꾸기", "점수 계산", "결과"]
    gap = 36
    width = (1472 - gap * 3) / 4
    for i, label in enumerate(labels):
        x = 64 + i * (width + gap)
        card(draw, (x, 172, x + width, 468), fill=LIGHT if i == 2 else CREAM)
        txt(draw, (x + width / 2, 199), label, 38, True, NAVY, anchor="mt")
        if i == 0:
            paragraph(draw, (x + 18, 302, width - 36), "배터리\n수명 안내", 42, False, INK, align="center", line_height=58)
        elif i == 1:
            txt(draw, (x + width / 2, 315), "[1, 0]", 50, True, TEAL, anchor="mm")
            paragraph(draw, (x + 12, 387, width - 24), "배터리 1 · 경기 0", 36, False, INK, align="center")
        elif i == 2:
            paragraph(draw, (x + 18, 288, width - 36), "1 × (+1)\n+ 0 × (-2)\n= +1", 40, True, TEAL, align="center", line_height=53)
        else:
            txt(draw, (x + width / 2, 315), "IT", 76, True, TEAL, anchor="mm")
            paragraph(draw, (x + 18, 390, width - 36), "합계가 양수", 36, False, INK, align="center")
        if i < 3:
            arrow(draw, (x + width + 5, 320), (x + width + gap - 4, 320), width=5)
    txt(draw, (800, 532), "추론에서는 점수표를 바꾸지 않고 문서의 합계만 계산합니다.", 38, True, TEAL, anchor="mm")
    txt(draw, (800, 588), "설명용 예제: 배터리 +1, 경기 -2는 이미 학습한 점수라고 가정합니다.", 36, False, NAVY, anchor="mm")
    save(image, "study/chapter1/assets/model_flow.png")


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


def dataset_split():
    image, draw = canvas(1110)
    heading(draw, "서로 다른 원문 100개를 역할별로 나누기")
    card(draw, (64, 154, 1536, 284), fill=CREAM)
    txt(draw, (800, 186), "원문 단위로 나눈 뒤 조각을 만듭니다.", 43, True, NAVY, anchor="mt")
    txt(draw, (800, 248), "같은 원문의 복사본·조각·번역본은 같은 쪽에 둡니다.", 36, False, INK, anchor="mm")
    columns = [
        ("학습", "60개", "모델의 숫자를 구함", LIGHT),
        ("검증", "20개", "개선·선택에 사용", CREAM),
        ("최종 평가", "20개", "선택을 마친 뒤 확인", "#EEF0F7"),
    ]
    width = 464
    for i, (title, count, purpose, fill) in enumerate(columns):
        x = 64 + i * 504
        draw.line((x + width / 2, 284, x + width / 2, 338), fill=LINE, width=4)
        card(draw, (x, 338, x + width, 626), fill=fill, outline=LINE)
        txt(draw, (x + width / 2, 373), title, 43, True, NAVY, anchor="mt")
        txt(draw, (x + width / 2, 447), count, 54, True, TEAL, anchor="mt")
        paragraph(draw, (x + 24, 548, width - 48), purpose, 36, False, INK, align="center")
    card(draw, (64, 676, 1536, 918), fill=WHITE, outline=LINE)
    txt(draw, (99, 705), "학습 → 검증 → 개선·재학습을 반복해 모델을 선택", 39, True, NAVY, anchor="lt")
    arrow(draw, (100, 815), (202, 815), width=6)
    txt(draw, (225, 815), "선택한 모델을 최종 평가 자료로 마지막에 확인", 39, True, TEAL, anchor="lm")
    txt(draw, (800, 973), "60 : 20 : 20은 설명용 예시이며 정해진 비율은 아닙니다.", 36, False, MUTED, anchor="mm")
    txt(draw, (800, 1045), "각 자료가 실제 사용할 한국어·영어 문서와 주제를 담는지도 확인합니다.", 34, False, INK, anchor="mm")
    save(image, "study/chapter1/assets/dataset_split.png")


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
    heading(draw, "모델을 준비한 뒤 Android에서 분석하기")
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
        title = ["PC·Colab", "배포 묶음", "Android 앱"][i]
        detail = ["분류·요약 모델 준비\n라벨 기준 확정", "모델·tokenizer\n전처리·라벨 정보", "같은 전처리로\n분류·요약 실행"][i]
        txt(draw, (center, 428), title, 46, True, NAVY, anchor="mt")
        paragraph(draw, (x1 + 26, 510, x2 - x1 - 52), detail, 40, False, INK, align="center", line_height=56)
        if i < 2:
            arrow(draw, (x2 + 13, 432), (positions[i + 1][0] - 13, 432))
    txt(draw, (800, 741), "모델 준비와 문서 분석의 실행 위치를 구분합니다.", 40, True, NAVY, anchor="mm")
    txt(draw, (800, 825), "모델 다운로드와 사용자 문서 분석은 별개의 과정입니다.", 38, False, MUTED, anchor="mm")
    save(image, "study/assets/on_device_flow.png")


def chapter2_data_pipeline():
    image, draw = canvas(800)
    heading(draw, "문서 파일에서 실습용 수치 배열까지")
    stages = [
        ("1  글자 추출", "TXT 또는 글자를\n선택할 수 있는 PDF", "파일에서 글자를 읽는다"),
        ("2  원문 보존", "추출한 글자를\n그대로 따로 둔다", "원본과 대조할 기준"),
        ("3  정리·기록", "정리본과 문서 ID·\n출처·언어·라벨 기록", "바뀐 곳을 확인한다"),
        ("4  묶음 분리", "원문 묶음을\n통째로 나눈다", "자료 누출 방지"),
        ("5  배열 만들기", "같은 단위의 수치를\n행과 열로 놓는다", "예: 문단별 글자 수"),
    ]
    width, gap = 270, 29
    for index, (title, detail, purpose) in enumerate(stages):
        x = 64 + index * (width + gap)
        card(draw, (x, 185, x + width, 592), fill=LIGHT if index in (1, 3) else CREAM)
        txt(draw, (x + 18, 221), title, 32, True, NAVY, anchor="lt")
        draw.line((x + 18, 277, x + width - 18, 277), fill=LINE, width=2)
        paragraph(draw, (x + 18, 315, width - 36), detail, 33, False, INK, line_height=50)
        paragraph(draw, (x + 18, 503, width - 36), purpose, 29, True, TEAL, line_height=42)
        if index < len(stages) - 1:
            arrow(draw, (x + width + 4, 387), (x + width + gap - 5, 387), width=5)
    card(draw, (64, 646, 1536, 746), fill=WHITE, outline=LINE)
    txt(draw, (800, 693), "이미지로만 된 PDF는 이 장의 글자 추출 방법으로 읽을 수 없습니다.", 36, True, NAVY, anchor="mm")
    save(image, "study/chapter2/assets/data_pipeline.png")


def chapter2_numpy_axes():
    image, draw = canvas(1130)
    heading(draw, "NumPy 배열: 문서 3개 × 문단 2개의 글자 수")
    card(draw, (64, 153, 1536, 254), fill=CREAM)
    txt(draw, (800, 203), "모든 칸의 단위는 글자 수 · 손으로 만든 2문단 예제", 38, True, NAVY, anchor="mm")

    x_edges = [105, 353, 638, 923]
    y_edges = [332, 430, 548, 666, 784]
    headers = ["문서", "문단 1", "문단 2"]
    for column, label in enumerate(headers):
        x1, x2 = x_edges[column], x_edges[column + 1]
        card(draw, (x1, y_edges[0], x2, y_edges[1]), fill=NAVY, radius=9)
        txt(draw, ((x1 + x2) / 2, (y_edges[0] + y_edges[1]) / 2), label, 37, True, WHITE, anchor="mm")
    rows = [("A", 12, 8), ("B", 5, 7), ("C", 9, 4)]
    for row_index, row in enumerate(rows, start=1):
        y1, y2 = y_edges[row_index], y_edges[row_index + 1]
        for column, value in enumerate(row):
            x1, x2 = x_edges[column], x_edges[column + 1]
            card(draw, (x1, y1, x2, y2), fill=LIGHT if column == 0 else WHITE, outline=LINE, radius=9)
            txt(draw, ((x1 + x2) / 2, (y1 + y2) / 2), str(value), 47, column > 0, NAVY, anchor="mm")

    card(draw, (1010, 332, 1536, 784), fill=CREAM, outline=LINE)
    txt(draw, (1273, 368), "axis=1 → 행 안에서 합", 36, True, TEAL, anchor="mt")
    txt(draw, (1273, 424), "문서마다 한 값", 32, False, MUTED, anchor="mt")
    for row_index, (label, left, right) in enumerate(rows):
        y = (y_edges[row_index + 1] + y_edges[row_index + 2]) / 2
        txt(draw, (1273, y), f"{label}: {left} + {right} = {left + right}", 38, True, NAVY, anchor="mm")

    card(draw, (353, 836, 923, 1006), fill=LIGHT, outline=LINE)
    txt(draw, (638, 868), "axis=0 ↓ 문서 방향으로 합", 37, True, TEAL, anchor="mt")
    txt(draw, (493, 959), "12 + 5 + 9 = 26", 34, True, NAVY, anchor="mm")
    txt(draw, (788, 959), "8 + 7 + 4 = 19", 34, True, NAVY, anchor="mm")
    txt(draw, (800, 1070), "shape=(3, 2): 행은 문서 3개, 열은 문단 2개입니다.", 37, True, NAVY, anchor="mm")
    save(image, "study/chapter2/assets/numpy_axes.png")


def chapter2_group_split():
    image, draw = canvas(1090)
    heading(draw, "같은 원문에서 나온 자료는 함께 나누기")
    card(draw, (64, 150, 780, 374), fill=LIGHT, outline=LINE)
    txt(draw, (99, 182), "원문 묶음 G-IT-01", 41, True, NAVY, anchor="lt")
    txt(draw, (99, 251), "IT-01 원문", 37, False, INK, anchor="lt")
    txt(draw, (99, 310), "IT-01-EN 영어 번역본", 37, False, INK, anchor="lt")
    card(draw, (820, 150, 1536, 374), fill=CREAM, outline=LINE)
    txt(draw, (855, 182), "원문 묶음 G-ECO-02", 41, True, NAVY, anchor="lt")
    txt(draw, (855, 251), "ECO-02 원문", 37, False, INK, anchor="lt")
    txt(draw, (855, 310), "ECO-02-COPY 복사본", 37, False, INK, anchor="lt")
    txt(draw, (800, 428), "원문·번역본·조각·복사본에 같은 묶음 ID를 붙입니다.", 37, True, NAVY, anchor="mm")

    columns = [
        ("학습", "01 묶음", "G-IT-01\nG-ECO-01\nG-SPT-01", LIGHT),
        ("검증", "02 묶음", "G-IT-02\nG-ECO-02\nG-SPT-02", CREAM),
        ("최종 평가", "03 묶음", "G-IT-03\nG-ECO-03\nG-SPT-03", "#EEF0F7"),
    ]
    for i, (title, family, groups, fill) in enumerate(columns):
        x = 64 + i * 504
        card(draw, (x, 498, x + 464, 897), fill=fill, outline=LINE)
        txt(draw, (x + 232, 531), title, 43, True, NAVY, anchor="mt")
        txt(draw, (x + 232, 604), family, 36, True, TEAL, anchor="mt")
        draw.line((x + 28, 673, x + 436, 673), fill=LINE, width=2)
        paragraph(draw, (x + 28, 701, 408), groups, 36, False, INK, align="center", line_height=56)
    txt(draw, (800, 966), "같은 묶음 ID가 둘 이상의 칸에 나타나면 자료 누출입니다.", 38, True, TEAL, anchor="mm")
    txt(draw, (800, 1031), "이 9개 묶음은 분리 원리 연습용이며 성능 평가용 표본은 아닙니다.", 34, False, MUTED, anchor="mm")
    save(image, "study/chapter2/assets/group_split.png")


if __name__ == "__main__":
    learning_path()
    model_flow()
    ai_ml_dl()
    training_inference()
    dataset_split()
    document_tasks()
    on_device_flow()
    chapter2_data_pipeline()
    chapter2_numpy_axes()
    chapter2_group_split()
