# 한국어 문서 검색·묶음 임베딩 실험

이 실험은 메모의 표현이 달라도 필요한 원문을 찾고 같은 사건의 메모를 모을 수 있는지 확인한다. 예를 들어 “연차 허가 받은 기록”으로 “휴가 신청 승인” 메모를 찾는다. 임베딩(embedding)은 문서의 뜻을 숫자 배열로 표현한 것이다. 이 숫자가 비슷한 문서를 찾는 데 쓰인다. 묶음의 이름은 메모에 실제로 나온 낱말에서 고른다. 이 단계는 M2 Mac의 텍스트 문서만 다루며 Android 앱의 성능을 판정하지 않는다.

## 자료가 만들어지는 순서

1. 코드가 가상의 가족·날짜·사건·확인할 사실을 정한다. 같은 사건에는 메모 4개를 배정하고 사건 카드의 가상 인물 이름을 메모 제목에 붙인다.
2. 로컬 Qwen3 8B가 사건 카드로 한국어 메모와 검색 질문을 쓴다. 답은 첫 메모에만 두고 나머지 메모는 같은 사건의 준비·후속 기록으로 만든다. 근거 구절이 원문과 일치하는지 코드가 검사한다. 이어서 답을 짐작하게 하는 단서도 다른 메모에서 제거한다. 예를 들어 정답이 ‘학교 강당’이면 다른 메모의 ‘강당’을 지운다. 생성 단계의 1차 보정본과 단서 제거본을 따로 보관하고 두 단계의 수정 사항을 각각 `repairs`와 `curation_repairs`에 남긴다. 교사 모델의 보정 전 응답 전문은 따로 저장하지 않는다.
3. 가족·사건 단위로 학습 800건, 개발 300건, 고정 평가 1,000건을 나눈다. 고정 평가 질문 140개 중 70개는 원문과 다른 표현으로 바꾼다. 질문에는 가상 인물 이름을 넣되 날짜를 일률적으로 덧붙이지 않는다. 정답을 가까이, 비슷한 오답을 멀리 놓는 학습 세 쌍을 검색용 200개와 묶음용 800개 만든다.
4. 원본 EmbeddingGemma 2와 미세 조정한 텍스트 전용 모델을 같은 자료로 비교한다. 개발 자료로 군집 기준값과 모델 체크포인트를 고른다. 고정 평가 자료는 모델 선택에 사용하지 않는다.

질문 상위 5개 결과에 근거 문서가 있으면 검색 성공으로 센다. 목표는 고정 평가 질문 140개의 성공률 90% 이상이다. 같은 사건의 다른 메모와 근거 메모를 구별하는지 보려고 첫 결과의 정답률도 따로 기록한다. 문서 묶기는 관련·무관 문서 쌍 100개에서 F1 0.80 이상을 목표로 한다. F1은 잘못 묶은 경우와 놓친 경우를 함께 반영하는 점수다. 사람이 평가 정답을 원문과 대조하기 전의 결과는 잠정치다. PDF 추출, 생성형 답변, 스마트폰 저장 공간과 속도는 이 결과에 포함되지 않는다.

가족과 사건은 분할 사이에 겹치지 않지만 10가지 주제와 일부 정답 표현은 반복된다. 따라서 이 평가는 처음 보는 사건을 찾는 능력을 확인한다. 새로운 주제나 실제 사람이 쓰는 메모에서도 같은 점수가 나오는지는 확인하지 못한다.

이 합성 자료에는 가상 인물 이름과 첫 메모 위치를 이용하는 지름길이 있다. 앞의 평가 질문 140개에서는 이 규칙만으로도 정답 메모가 상위 5개 안에 모두 들어갔다. 첫 결과 정답률은 20%였다. 따라서 상위 5개 성공률만으로 의미 검색 능력을 판정하지 않는다. 모델의 첫 결과 정답률과 평균 역순위(MRR, 정답을 앞쪽에 둘수록 커지는 값)를 이 위치 규칙 및 낱말 검색과 함께 기록한다.

## 모델이 배우는 관계

검색 학습에는 질문, 답이 든 첫 메모, 같은 사건이지만 답이 없는 다른 메모를 한 묶음으로 넣는다. 예를 들어 “소아과 어디로 가지?”라는 질문에는 병원 이름이 적힌 예약 메모가 정답이다. 준비물만 적힌 메모는 같은 사건이어도 검색 정답이 아니다. 문서 묶기 학습에는 같은 사건의 두 메모와 주제만 비슷한 다른 사건의 메모를 넣는다. 모델의 작은 추가 가중치인 LoRA를 조정해 정답 쌍의 임베딩이 오답 쌍보다 가깝도록 한다.

학습에 쓰지 않은 개발 자료로 묶음의 거리 기준과 학습 시점을 고른다. 고정 평가 결과로 이 선택을 바꾸지 않는다. 묶음 이름은 형태소 분석으로 찾은 원문 명사 중 여러 메모에 나온 말로 정한다. 이름의 출처인 메모 ID도 남긴다. 이 두 작업이 태그를 붙이는 간단한 규칙이다. 태그가 실제 묶음을 잘 설명하는지는 별도 사람 검토가 필요하다.

평가를 실행하면 `artifacts/evaluation/tuned.tag_review.csv`에 크기가 다른 묶음 최대 30개의 태그와 예시 메모가 나온다. 이 표에서 태그가 묶음을 잘 설명하는지 살핀다. 이 검토는 관련·무관 문서 쌍의 F1 점수를 대신하지 않는다.

## 실행

아래 명령은 모두 이 디렉터리에서 실행한다. Python 3.12, `uv`, Ollama가 필요하다. 패키지 버전은 `uv.lock`에 고정한다. `artifacts/`에는 합성 자료와 체크포인트가 저장되며 Git에는 올라가지 않는다.

```bash
uv sync --frozen
ollama pull qwen3:8b
.venv/bin/python generate.py --split train --max-events 50 --output artifacts/raw_events.jsonl --max-hours 12
.venv/bin/python generate.py --output artifacts/raw_events.jsonl --max-hours 12
.venv/bin/python repair_duplicates.py --max-hours 12
ollama stop qwen3:8b
.venv/bin/python curate.py
.venv/bin/python prepare.py
.venv/bin/python validate.py
.venv/bin/python train.py
.venv/bin/python select.py
.venv/bin/python evaluate.py --output artifacts/evaluation/base.json
.venv/bin/python evaluate.py --adapter artifacts/trained/adapter --output artifacts/evaluation/tuned.json
```

첫 50개 사건은 생성 속도·형식을 확인하는 예비 실행이다. 같은 출력 파일에서 이어 만들 수 있다. 8B 모델이 실행되지 않거나 전체 생성 예상 시간이 12시간을 넘으면 `qwen3:4b`로 처음부터 새 파일에 만든다. 두 모델의 자료를 합치지 않는다. 생성과 학습의 실행 시간은 합계 24시간 이내로 기록한다. `repair_duplicates.py`는 서로 다른 사건에서 문장이 똑같이 나온 경우 뒤에 생성된 사건만 다시 쓴다. 이전 문장과 재생성 설정은 사건 행에 남긴다. 두 모델이 동시에 메모리에 남지 않도록 생성 후 `ollama stop`을 실행한다. 단서 제거 규칙은 `curate.py`에서 관리한다. `curation.json`에는 1차 보정본과 단서 제거본의 해시를 남긴다.

## 정답 검수와 비공개 보관

`artifacts/dataset/review.csv`에는 고정 평가 질문 140개와 문서 쌍 100개가 들어 있다. 사람이 원문을 확인한 항목의 `accepted`에 `yes`를 적는다. 틀린 정답은 자료를 고쳐 새 검수본을 만든다. 240개 항목이 모두 확인되기 전에는 정식 합격 판정을 내리지 않는다.

질문 행은 `question_or_left`의 질문을 읽고 `evidence_or_right`에 실린 같은 사건의 메모 4개를 대조한다. `gold`의 근거가 실제로 첫 메모에 있으며 다른 메모에는 같은 답이 없는지 확인한다. 문서 쌍 행은 두 메모가 같은 사건이면 `True`, 다른 사건이면 `False`가 맞는지 확인한다. 잘못된 행은 `comment`에 이유를 적고 `accepted`를 비워 둔다. `gold` 칸만 고치지 말고 원본 사건을 수정한 뒤 검수표를 다시 만들어야 한다.

```bash
.venv/bin/python review.py
.venv/bin/python evaluate.py --output artifacts/evaluation/base.json
.venv/bin/python evaluate.py --adapter artifacts/trained/adapter --output artifacts/evaluation/tuned.json
.venv/bin/python publish.py
```

사람 검수 전 비공개 보관본을 올리려면 `publish.py --allow-provisional`을 실행한다. 이때 Dataset Card와 Model Card에 잠정 상태를 표시한다. 잠정 보관본은 공식 품질 판정에 사용할 수 없다. 평가 정답 240개를 확인하고 `review.py`를 통과한 자료는 `publish.py`를 추가 옵션 없이 실행해 검수 완료 revision으로 올린다.

`publish.py`는 합성 데이터와 LoRA 어댑터를 서로 다른 비공개 Hugging Face Hub 저장소에 올린다. LoRA는 기반 모델의 일부에 작은 학습 값을 덧붙이는 방식이다. 전체 revision은 저장소의 특정 버전을 가리키는 식별값이다. 이 값과 데이터 파일 해시를 확인한다. 실제 사용자 메모·질문·로그는 생성 입력이나 업로드 대상에 넣지 않는다. Hub 인증 정보는 개발 PC에서만 사용한다.

## 현재 확인 상태

- 가상 사건 525개에서 메모 2,100건을 만들었다. 분할·중복·근거 검사를 통과했으며, 중복 사건 8개는 재생성했다.
- EmbeddingGemma 2 텍스트 인코더를 250단계 학습하고 2회차 체크포인트를 개발 자료로 골랐다. MPS는 Mac GPU 계산 경로다.
- 고정 평가에서 잠정 검색 Recall@5는 134/140(95.7%)이다. 문서 쌍 표본 F1은 0.990이지만 전체 사건 쌍 F1은 0.529다. 해석과 한계는 [잠정 결과](RESULTS.md)에 기록했다.
- 데이터와 LoRA 어댑터를 검수 대기 상태로 비공개 Hub에 올린 뒤 다시 내려받아 해시와 모델 평가를 대조했다. 정답 240개와 태그 30개의 사람 검수는 남아 있다. Android 앱은 시험하지 않았다.

현재 저장소: [합성 데이터](https://huggingface.co/datasets/heecchoi/slm-wiki-embeddinggemma2-text-synthetic-v1/tree/ddba19537f56966c5fbe352707b93e71f57ccdf0) · [LoRA 어댑터](https://huggingface.co/heecchoi/slm-wiki-embeddinggemma2-ko-lora-v1/tree/f51442c47345d2f3e9113c5ad77057e42d14ea59). 두 저장소 모두 비공개다.

[잠정 결과](RESULTS.md) · [POC 한 페이지 요약](../README.md) · [상세 PRD](../SLM_Wiki_PRD.md)
