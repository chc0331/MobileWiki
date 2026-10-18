# 학습 자료 PDF 생성

`study/`의 Markdown을 원본으로 사용해 같은 디렉터리에 같은 이름의 PDF를 만든다. 본문, 표, 실습, 해설과 코드를 유지하고 표지, 목차, 북마크, 페이지 번호와 학습서 편집을 적용한다.

## 준비

Python 3 환경에서 필요한 패키지를 설치한다.

```sh
python3 -m pip install -r tools/requirements.txt
```

macOS에서는 시스템의 한글 글꼴과 Menlo를 기본으로 사용한다. Linux에서는 NanumMyeongjo, NanumGothic, DejaVuSansMono의 TrueType 글꼴을 사용할 수 있다. 다른 환경에서는 다음 환경 변수에 사용 가능한 글꼴 파일 경로를 지정한다. 폰트 파일은 저장소에 복사하지 않는다.

| 환경 변수 | 용도 |
| --- | --- |
| `PDF_BODY_FONT` | PDF 한글 본문용 TrueType 글꼴 |
| `PDF_SANS_FONT` | PDF 제목·표·한글 코드용 TrueType 글꼴 |
| `PDF_CODE_FONT` | 코드의 영문·숫자용 고정폭 TrueType 글꼴 |
| `CJK_FONT` | 그림의 한글 글꼴. macOS 이외에서는 명시적으로 지정 |

## 생성

그림을 추가·수정한 경우 그림을 먼저 생성하고 PDF를 갱신한다.

```sh
python3 tools/build_study_figures.py
python3 tools/build_study_pdfs.py
```

PDF 생성 도구는 모든 `study/**/*.md`를 처리한다. 원문의 텍스트 요소가 PDF에 같은 순서로 포함됐는지 확인하며, PDF의 상대 링크는 다른 PDF를 향하도록 변환한다. 새 그림을 추가할 때는 그림 생성 코드와 Markdown의 이미지 링크도 함께 작성한다.

## 시각 검토

생성 후 Poppler의 `pdftoppm`으로 모든 페이지를 이미지로 렌더링한다. 임시 파일은 저장소 밖에 둔다.

```sh
mkdir -p /tmp/mobilewiki-pdf-review
pdftoppm -r 120 -png study/README.pdf /tmp/mobilewiki-pdf-review/roadmap
pdftoppm -r 120 -png study/chapter1/AI와_머신러닝_기본_개념.pdf /tmp/mobilewiki-pdf-review/chapter1
```

Fontconfig가 별도로 설정된 환경에서는 해당 환경의 `FONTCONFIG_FILE`을 지정한다. 검토 이미지는 Git에 추가하지 않는다.

다음 항목을 모든 페이지에서 확인하고 수정한 PDF를 다시 렌더링한다.

- 한글, 문장부호, 체크박스와 코드의 글자가 깨지지 않는지
- 제목이 본문과 함께 있고, 표·그림·코드가 잘리지 않는지
- 반복 표 머리행과 실습 빈칸이 읽고 쓰기 편한지
- 실습과 정답 해설이 구분되고 목차와 북마크가 실제 페이지로 이동하는지
- 상대 링크, 그림과 PDF가 원본 Markdown의 최신 내용과 일치하는지
