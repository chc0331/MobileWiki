# 한국어 문서 검색·묶음 임베딩 실험

이 실험은 메모의 표현이 달라도 필요한 원문을 찾고, 같은 사건의 메모를 모을 수 있는지 확인한다. 예를 들어 “연차 허가 받은 기록”으로 “휴가 신청 승인” 메모를 찾는다. 임베딩(embedding)은 문서의 뜻을 숫자 배열로 표현한 것이다. 이 숫자가 비슷한 문서를 찾는 데 쓰인다. 묶음의 이름은 메모에 실제로 나온 낱말에서 고른다. 이 단계는 M2 Mac의 텍스트 문서만 다루며 Android 앱의 성능을 판정하지 않는다.

## 자료가 만들어지는 순서

1. 코드가 가상의 가족·날짜·사건·확인할 사실을 정한다. 같은 사건에는 메모 4개를 배정하고, 사건 카드의 가상 인물 이름을 메모 제목에 붙인다.
2. 로컬 Qwen3 8B가 사건 카드로 한국어 메모와 검색 질문을 쓴다. 답은 첫 메모에만 두고 나머지 메모는 같은 사건의 준비·후속 기록으로 만든다. 다른 메모에 답이 반복되면 그 문장을 제거한다. 근거 구절이 원문과 일치하는지 코드가 검사한다. 자동 수정 사항은 사건별 `repairs`에 남긴다.
3. 가족·사건 단위로 학습 800건, 개발 300건, 고정 평가 1,000건을 나눈다. 질문의 절반은 원문과 다른 표현으로 바꾼다. 질문에는 가상 인물 이름을 넣되 날짜를 일률적으로 덧붙이지 않는다. 정답을 가까이, 비슷한 오답을 멀리 놓는 학습 세 쌍을 검색용 200개와 묶음용 800개 만든다.
4. 원본 EmbeddingGemma 2와 미세 조정한 텍스트 전용 모델을 같은 자료로 비교한다. 개발 자료로 군집 기준값과 모델 체크포인트를 고른 뒤, 고정 평가 자료를 한 번 평가한다.

질문 상위 5개 결과에 근거 문서가 있으면 검색 성공으로 센다. 목표는 고정 평가 질문 140개의 성공률 90% 이상이다. 같은 사건의 다른 메모와 근거 메모를 구별하는지 보려고 첫 결과의 정답률도 따로 기록한다. 문서 묶기는 관련·무관 문서 쌍 100개에서 F1 0.80 이상을 목표로 한다. F1은 잘못 묶은 경우와 놓친 경우를 함께 반영하는 점수다. 사람이 평가 정답을 원문과 대조하기 전의 결과는 잠정치다. PDF 추출, 생성형 답변, 스마트폰 저장 공간과 속도는 이 결과에 포함되지 않는다.

가족과 사건은 분할 사이에 겹치지 않지만, 10가지 주제와 일부 정답 표현은 반복된다. 따라서 이 평가는 처음 보는 사건을 찾는 능력을 확인한다. 새로운 주제나 실제 사람이 쓰는 메모에서도 같은 점수가 나오는지는 확인하지 못한다.

## 모델이 배우는 관계

검색 학습에는 질문, 답이 든 첫 메모, 같은 사건이지만 답이 없는 다른 메모를 한 묶음으로 넣는다. 예를 들어 “소아과 어디로 가지?”라는 질문에는 병원 이름이 적힌 예약 메모가 정답이다. 준비물만 적힌 메모는 같은 사건이어도 검색 정답이 아니다. 문서 묶기 학습에는 같은 사건의 두 메모와 주제만 비슷한 다른 사건의 메모를 넣는다. 모델의 작은 추가 가중치인 LoRA를 조정해 정답 쌍의 임베딩이 오답 쌍보다 가깝도록 한다.

학습에 쓰지 않은 개발 자료로 묶음의 거리 기준과 학습 시점을 고른다. 고정 평가 자료는 마지막에 한 번 확인한다. 묶음 이름은 형태소 분석으로 찾은 원문 명사 중 여러 메모에 나온 말로 정한다. 이름의 출처인 메모 ID도 남긴다. 이는 태그를 붙이는 간단한 규칙이다. 태그가 실제 묶음을 잘 설명하는지는 별도 사람 검토가 필요하다.

평가를 실행하면 `artifacts/evaluation/tuned.tag_review.csv`에 크기가 다른 묶음 최대 30개의 태그와 예시 메모가 나온다. 이 표에서 태그가 묶음을 잘 설명하는지 살핀다. 이 검토는 관련·무관 문서 쌍의 F1 점수를 대신하지 않는다.

## 실행

아래 명령은 모두 이 디렉터리에서 실행한다. Python 3.12, `uv`, Ollama가 필요하다. 패키지 버전은 `uv.lock`에 고정한다. `artifacts/`에는 합성 자료와 체크포인트가 저장되며 Git에는 올라가지 않는다.

```bash
uv sync --frozen
ollama pull qwen3:8b
.venv/bin/python generate.py --split train --max-events 50 --output artifacts/raw_events.jsonl --max-hours 12
.venv/bin/python generate.py --output artifacts/raw_events.jsonl --max-hours 12
ollama stop qwen3:8b
.venv/bin/python prepare.py
.venv/bin/python validate.py
.venv/bin/python train.py
.venv/bin/python select.py
.venv/bin/python evaluate.py --output artifacts/evaluation/base.json
.venv/bin/python evaluate.py --adapter artifacts/trained/adapter --output artifacts/evaluation/tuned.json
```

첫 50개 사건은 생성 속도·형식을 확인하는 예비 실행이다. 같은 출력 파일에서 이어 만들 수 있다. 8B 모델이 실행되지 않거나 전체 생성 예상 시간이 12시간을 넘으면 `qwen3:4b`로 처음부터 새 파일에 만든다. 두 모델의 자료를 합치지 않는다. 생성과 학습의 실행 시간은 합계 24시간 이내로 기록한다. 두 모델이 동시에 메모리에 남지 않도록 생성 후 `ollama stop`을 실행한다.

## 정답 검수와 비공개 보관

`artifacts/dataset/review.csv`에는 고정 평가 질문 140개와 문서 쌍 100개가 들어 있다. 사람이 원문을 확인한 항목의 `accepted`에 `yes`를 적는다. 틀린 정답은 자료를 고쳐 새 검수본을 만든다. 240개 항목이 모두 확인되기 전에는 정식 합격 판정을 내리지 않는다.

```bash
.venv/bin/python review.py
.venv/bin/python evaluate.py --output artifacts/evaluation/base.json
.venv/bin/python evaluate.py --adapter artifacts/trained/adapter --output artifacts/evaluation/tuned.json
.venv/bin/python publish.py
```

`publish.py`는 검수된 합성 데이터와 LoRA 어댑터를 서로 다른 비공개 Hugging Face Hub 저장소에 올린다. LoRA는 기반 모델의 일부에 작은 학습 값을 덧붙이는 방식이다. 전체 revision은 저장소의 특정 버전을 가리키는 식별값이다. 이 값과 데이터 파일 해시를 확인한다. 실제 사용자 메모·질문·로그는 생성 입력이나 업로드 대상에 넣지 않는다. Hub 인증 정보는 개발 PC에서만 사용한다.

## 현재 확인 상태

- EmbeddingGemma 2 텍스트 인코더의 MPS(Mac GPU 계산 경로) 추론과 LoRA 학습 1단계, 어댑터 저장·재로딩을 확인했다.
- Qwen3 8B의 로컬 메모 4건·질문 1건 생성과 근거 보정을 확인했다.
- 전체 자료의 생성·학습·고정 평가·사람 검수·Hub 업로드는 아직 완료되지 않았다.

[POC 한 페이지 요약](../README.md) · [상세 PRD](../SLM_Wiki_PRD.md)
