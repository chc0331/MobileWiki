# 한국어 문서 검색·묶음 임베딩 실험

이 실험은 메모의 표현이 달라도 필요한 원문을 찾고 같은 사건의 메모를 모을 수 있는지 확인한다. 예를 들어 “연차 허가 받은 기록”으로 “휴가 신청 승인” 메모를 찾는다. 임베딩(embedding)은 문서의 뜻을 숫자 배열로 표현한 것이다. 이 숫자가 비슷한 문서를 찾는 데 쓰인다. 묶음의 이름은 메모에 실제로 나온 낱말에서 고른다. 이 단계는 M2 Mac의 텍스트 문서만 다루며 Android 앱의 성능을 판정하지 않는다.

## v3: 사건 맥락과 군집 후처리

`context-hardneg-v3`는 같은 날짜·주제의 다른 사건도 구별하는지 확인한다. 사건마다 상황과 목적을 정하고 네 메모가 이를 서로 다른 표현으로 공유하게 한다. 정답 메모는 Stage 1(첫 생성 단계)에서 먼저 만든다. 주변 메모 세 개와 질문은 Stage 2(다음 생성 단계)에서 각각 독립 요청으로 만든다. Stage 2에는 정답값·정답 메모·이전 응답과 이력을 전달하지 않는다. 검증을 통과하지 못하면 해당 단계를 다시 생성한다. 정답 문장을 옮기거나 일반 문구로 빈 내용을 채우지 않는다. 상황과 목적은 동의 표현 후보를 허용한다. 같은 두 단어를 모든 입력에 그대로 넣도록 강제하지 않는다.

정답과 같은 주제의 알려진 후보는 띄어쓰기가 달라도 검사한다. 예를 들어 `엔진 오일`도 정답 후보의 단서인 `엔진오일`과 비교한다. 답 메모에 다른 후보를 더하거나 주변 메모·질문에서 후보를 드러내면 거부한다. 이 검사는 공백을 뺀 문자열로 비교만 하며 저장된 원문을 수정하지 않는다. 지정된 정답 문자열이 답 메모에 있어야 한다는 조건은 그대로다. 근거 구절은 기존 날짜 표기 정규화를 거친 뒤 답 메모의 연속된 부분과 일치해야 한다.

소아과 예약의 합성 카드는 사람을 환자로 정한다. 이 범주의 제목·본문·질문에서 반려견·반려묘·강아지·고양이·동물병원·수의사 등 알려진 동물 관련 표현을 거르며 집 수리·연차의 반려동물 배경은 유지한다. 알려진 낱말을 찾는 검사이므로 처음 보는 표현이나 문맥 전체의 의미 오류를 모두 찾는다고 보장하지 않는다.

새 자료도 사건 525개와 메모 2,100개다. 각 분할의 사건 60%를 같은 주제·날짜의 서로 다른 5가족 사건으로 묶어 배치한다. 학습에는 검색 triplet(질문·정답·오답) 400개와 군집 triplet 800개를 쓴다. 군집 오답은 같은 가족의 다른 사건 A 400개, 같은 주제이면서 가족과 정답값이 다른 사건 B 240개, 같은 정답값의 다른 가족 사건 C 160개다. C 유형은 기준 메모와 오답 메모의 실제 문장에 같은 정답값이 있어야 한다. 검색은 같은 사건의 답 없는 메모와 다른 사건 메모를 함께 오답 예시로 사용한다.

군집 입력에서만 알려진 가상 인물 이름을 가린다. 검색 입력과 원문에는 이름이 남는다. 현재 이름 처리는 합성 카드에 알려진 인물을 사용하므로 실제 메모에서 이름을 자동으로 찾는 성능까지 입증하는 것은 아니다.

군집 벡터의 평균 중심화·상위 주성분 제거를 개발 자료에서 비교한다. 본문 날짜와 상호 근접 이웃 거리, average·complete 연결도 같은 자료에서 비교한다. 평균과 주성분은 학습 문서에서만 계산한다. 날짜를 확정할 수 없는 본문에는 날짜 페널티를 주지 않는다. 사건·가족 ID와 정답 날짜는 학습 예시 구성·검증·오류 진단에 사용하며 군집 거리 계산에는 넣지 않는다. 태그는 원문 명사의 군집별 TF-IDF 점수로 고르며 출처와 후보를 넓힌 단계를 기록한다.

기반 모델은 revision `f9972ac22e696a105c482202646979e33e4b5a1e`로 고정한다. q/v·마진 0.1과 q/k/v/o·마진 0.1·0.2·0.3을 각각 2 epoch 학습한다. 개발 자료에서 검색 Recall@5 0.90 이상을 만족하는 후보를 추린다. 그중 전체 사건 쌍 F1이 가장 높은 후보를 고른다. 시험 자료에서 검색 Recall@5 0.90과 전체 사건 쌍 F1 0.80을 함께 목표로 한다. 사람 검수 전 판정은 잠정치다.

## v2 고정 자료의 추가 학습 비교

새 v3 자료 생성과 분리해 v2 원문과 평가 정답을 고정한 통제 실험도 실행한다. 통제 실험은 비교할 조건을 정해 두고 변경 효과를 확인하는 실험이다. `phase2.py`는 문서·질문·평가 쌍·검수표 파일을 그대로 복사하고 학습용 triplet만 바꾼다. 원본 manifest와 학습 관계도 별도 파일로 보존한다. 새 파생 revision은 `shortcut-hardneg-v2-cluster-phase2`이며, 새 v3 자료에서 측정한 성능으로 해석하지 않는다.

질문마다 같은 사건의 답 없는 메모와 같은 가족의 다른 사건을 오답으로 넣어 검색 학습 400개를 만든다. 군집 학습은 메모 800개를 각각 한 번씩 기준 메모로 사용한다. A 400개는 같은 가족의 다른 사건, B 240개는 같은 주제이면서 가족과 정답값이 다른 사건, C 160개는 정답값은 같고 가족은 다른 사건이다. C의 기준 메모와 오답 메모에는 실제 제목이나 본문에 그 값이 있어야 한다. 평가 쌍의 기존 분류와 정답은 바꾸지 않는다.

파생 manifest에 `clustering_deidentify: true`를 지정하면 군집 입력의 알려진 가상 인물 이름을 가린다. 학습·어려운 오답 선별·개발 선택·평가·문자 기준선이 같은 설정을 읽는다. 검색 입력과 보관한 원문은 유지한다. 입력이 실제로 달라지면 학습 입력 해시도 바뀌므로 이전 학습 결과를 그대로 재사용할 수 없다.

이 디렉터리에서 「v3 실행」의 환경 준비를 마친 뒤 다음 명령을 사용한다. 현재 실험에는 기존 누적 생성 장부가 있다. 아래 명령은 이 실험을 이어 실행하는 예다. `--max-hours 23`은 주 실행의 누적 상한 예시다. 준비·문자 기준선·회귀 확인에 쓴 시간을 합쳐 24시간을 넘기지 않도록 실제 상한을 줄인다. 이미 만든 출력 폴더는 덮어쓰지 않는다. 자료가 있으면 생성 단계를 건너뛰고 검증과 재사용 조건부터 확인한다.

```bash
.venv/bin/python phase2.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --output artifacts/context-hardneg-v3/phase2_v2/dataset
.venv/bin/python mine.py --dataset artifacts/context-hardneg-v3/phase2_v2/dataset --output artifacts/context-hardneg-v3/phase2_v2/mined_dataset --adapter artifacts/shortcut-hardneg-v2-balanced/trained/adapter --generation-budget artifacts/context-hardneg-v3/generation_budget.json --max-hours 23
.venv/bin/python validate.py --dataset artifacts/context-hardneg-v3/phase2_v2/mined_dataset
.venv/bin/python run_experiment.py --dataset artifacts/context-hardneg-v3/phase2_v2/mined_dataset --output artifacts/context-hardneg-v3/phase2_v2/training --generation-budget artifacts/context-hardneg-v3/generation_budget.json --max-hours 23 --epochs 2 --evaluate --visualize
```

네 후보의 개발 선택과 시험 평가 조건은 v3 경로와 같다. 기존 v2 모델에도 이름 가리기와 개발 선택 절차를 동일하게 적용해 추가로 비교한다. 선택한 새 가중치와 고정 군집 설정을 v1·v2 자료에 적용한 회귀 결과는 별도로 기록한다. 실행 결과와 새 v3 자료의 생성 실패는 [실험 기록](RESULTS_v3.md)에서 구분해 확인한다.

## v3 실행

Python 3.12, `uv`, Ollama가 필요하다. 학습 명령의 기본 장치는 Mac GPU 계산 경로인 MPS다. 모든 명령은 이 디렉터리에서 실행한다. `--inexact`는 기존 변환용 추가 패키지를 유지한다. 첫 50개 사건의 속도·형식·누출 검사를 확인한 뒤 같은 파일에 전체 생성을 이어 간다. 생성 단계가 끝나면 Ollama를 종료한 후 임베딩 모델을 불러온다.

아래 명령은 전체 재현 절차다. 현재 `p2-transfer` 실행은 「168개 사건의 의미 점검과 수정」에 설명한 것처럼 학습·개발 자료를 먼저 완성한다. 모델 선택을 고정한 뒤 시험 자료를 생성한다.

```bash
uv sync --frozen --group viz --inexact
ollama pull qwen3.5:9b-q4_K_M
.venv/bin/python generate.py --revision context-hardneg-v3 --teacher qwen3.5:9b-q4_K_M --output artifacts/context-hardneg-v3/raw_events.jsonl --max-events 50 --max-hours 23
.venv/bin/python generate.py --revision context-hardneg-v3 --teacher qwen3.5:9b-q4_K_M --output artifacts/context-hardneg-v3/raw_events.jsonl --max-hours 23
.venv/bin/python repair_duplicates.py --input artifacts/context-hardneg-v3/raw_events.jsonl --max-hours 23 --allow-test-repair
ollama stop qwen3.5:9b-q4_K_M
.venv/bin/python curate.py --input artifacts/context-hardneg-v3/raw_events.jsonl --output artifacts/context-hardneg-v3/curated_events.jsonl --manifest artifacts/context-hardneg-v3/curation.json
.venv/bin/python prepare.py --input artifacts/context-hardneg-v3/curated_events.jsonl --output artifacts/context-hardneg-v3/dataset
.venv/bin/python validate.py --dataset artifacts/context-hardneg-v3/dataset
.venv/bin/python mine.py --dataset artifacts/context-hardneg-v3/dataset --output artifacts/context-hardneg-v3/mined_dataset --adapter artifacts/shortcut-hardneg-v2-balanced/trained/adapter --generation-budget artifacts/context-hardneg-v3/generation_budget.json --max-hours 23
.venv/bin/python validate.py --dataset artifacts/context-hardneg-v3/mined_dataset
.venv/bin/python lexical_baseline.py --dataset artifacts/context-hardneg-v3/mined_dataset --output artifacts/context-hardneg-v3/lexical_baseline.json
.venv/bin/python run_experiment.py --dataset artifacts/context-hardneg-v3/mined_dataset --output artifacts/context-hardneg-v3/training --generation-budget artifacts/context-hardneg-v3/generation_budget.json --max-hours 23 --evaluate --visualize
```

Stage 1의 첫 시도는 기본 버전 3 입력으로 작성한다. 직전 시도가 알려진 검증 오류로 실패했으면 3·5·7회차에는 초안과 이전 오류 안내가 없는 기본 버전 3으로 새로 작성한다. 이때 시도 기록과 최종 메타데이터에 `restart_without_feedback: true`를 남기고 조건을 엄격히 확인한다.

2·4·6·8회차에 같은 요청의 직전 시도가 검증에 실패했고 원시 응답 파일과 해시가 남아 있으면 `retry_hint_version: 4`를 사용한다. 파일의 UTF-8 바이트로 SHA-256 해시를 계산해 직전 실패 기록과 일치하는지 확인한다. 모델이 쓴 내용(`message.content`)만 JSON 문자열로 인용해 수정 대상 초안으로 제시한다. 교사 모델이 카드 조건에 맞는 새 응답을 작성하며 코드가 초안에 사실을 넣거나 문장을 고치지 않는다. 직전 시도가 중단됐거나 짝수 회차에서 필요한 원시 응답이 없으면 기존 버전 3과 정해진 오류 안내를 사용한다.

각 요청은 새 시스템 메시지와 사용자 메시지 한 쌍으로 보낸다. Stage 2의 재시도는 `retry_hint_version: 3`과 정해진 오류 안내를 유지하며 정답·초안·이력을 받지 않는다. Stage 2에서는 새 작성 마커도 거부한다. 기존 재시도 버전 1~4의 프롬프트와 해시는 유지한다. 교대 정책은 새 시도에만 적용하며 과거 캐시나 실패 횟수를 고치거나 초기화하지 않는다. 난수·온도, 한 공간 8회·사건별 일반 복구 2블록, 누적 24시간 한도도 그대로다. 초안과 새 응답은 해시와 함께 보존한다.

v3 생성은 독립 요청 하나가 여덟 번 실패하면 중단한다. 같은 요청에서 여덟 번을 모두 실패했다면 사용하지 않은 `--seed-offset 1000`으로 다시 실행한다. 이 값은 생성 난수와 캐시 저장 공간을 바꾼다. 이미 완료된 사건은 검증한 뒤 건너뛰며 이전 캐시와 응답도 보존한다. 해당 공간도 실패하면 `2000`처럼 다른 값을 쓴다. 생성 시간은 기존 예산 파일에 계속 누적되며 이 옵션으로 시간 한도나 교사·자료 버전 검사를 우회할 수 없다.

```bash
.venv/bin/python generate.py --revision context-hardneg-v3 --teacher qwen3.5:9b-q4_K_M --output artifacts/context-hardneg-v3/raw_events.jsonl --seed-offset 1000 --max-hours 23
```

`lexical_baseline.py`는 학습 문서에서만 2~5글자 조각의 어휘와 TF-IDF 가중치를 계산한다. 개발 자료에서 군집 설정을 고른 뒤 시험 자료에 적용한다. 단어의 겹침만으로 묶이는 정도를 임베딩 결과와 비교하기 위한 기준선이다.

위 오답 예시 선별 명령은 기존 로컬 v2 어댑터를 사용한다. `--adapter`를 생략하면 고정 기반 모델을 선별기로 사용한다. 선별에는 학습 문서만 넣고 유형별 조건 안에서 유사도가 높은 오답을 고른다. 원래 데이터셋은 보존하고 `mined_dataset`에 결과와 사용한 모델의 해시를 기록한다.

`run_experiment.py`는 네 학습 후보와 개발 자료의 선택을 순서대로 실행한다. 선택을 고정한 뒤 `--evaluate`가 기준 모델·기준 모델 후처리·선택 모델을 시험하고 `--visualize`가 탐색 화면용 캐시를 만든다. 완료된 같은 실험은 이어서 실행할 수 있다. 기존 학습 결과는 학습 입력·조건·학습 코드가 다르면 재사용하지 않는다. 선택 코드가 바뀌면 개발 자료의 선택을 다시 실행해 어댑터와 군집 설정을 갱신한다. 이미 저장한 시험 결과는 조건과 해시가 일치할 때만 재사용한다. 실패와 중단 시간도 실행 예산에 포함한다. 주 생성·학습·평가에는 23시간을 두고 문자 기준선과 이전 revision 회귀 확인에는 남은 1시간을 사용한다. 이 시간을 합쳐 24시간을 넘기지 않는다. v3는 생성 예산 파일을 데이터셋 또는 원본 자료 위치에서 찾는다. 파일이 없으면 학습을 시작하지 않는다. 데이터 파일의 해시와 실제 학습 입력도 확인한다. 데이터·코드·설정을 묶어 식별하는 해시(fingerprint)가 이전 학습 결과에 없으면 새 실행에 재사용하지 않는다. 결과는 [v3 실험 결과](RESULTS_v3.md)에 기록한다.

새 자료의 사람 검수표는 `artifacts/context-hardneg-v3/mined_dataset/review.csv`다. 검색 질문 140개와 문서 쌍 100개를 원문과 대조한다. 질문이 실제 사실 관계를 묻는지, 다른 메모도 정답이 될 수 있는지 함께 확인한다. 태그는 `artifacts/context-hardneg-v3/training/evaluation/tuned.tag_review.csv`에서 최대 30개를 확인한다. 자동 검사 통과만으로 의미와 정답의 유일성이 보장되지는 않는다.

### 168개 사건의 의미 점검과 수정

자동 검사를 통과한 문장에도 증상을 바꾸거나 자료의 용도를 잘못 연결한 경우가 있었다. 학생 모델을 학습하기 전에 처음 완성한 168개 사건의 메모 네 개·질문·근거를 카드와 대조해 AI 보조 검토를 했다. 수정 대상은 13개 사건의 문서·질문 출력 14개로 정했다. 해석이 애매한 10개 사건은 확정 오류와 구분해 기록했다.

같은 Q4 교사가 수정 대상 출력만 다시 만들었다. 원본은 보존하고 후보 파일을 따로 검토했다. 13개 후보 사건 모두 핵심 정답 관계 검토를 통과한 뒤 해시가 일치하는 후보만 한 번에 반영했다. `train-0018`은 첫 후보에도 다른 준비물이 들어가 두 번째 후보를 만들었다. 같은 난수·캐시 공간에서 다음 시도를 이어 갔으며 실제 완료 상태와 이전 응답을 보존했다. 실패 상태를 새로 붙이거나 시도 횟수를 초기화하지 않았다.

검토 기록은 `artifacts/context-hardneg-v3/` 아래에 있다. 전체 검토는 `debug/semantic-audit-prefix168-a.json`·`debug/semantic-audit-prefix168-b.json`, 수정 계획은 `debug/semantic-repair-prefix168-plan.json`에 보존했다. 현재 후보의 최종 판단과 반영 해시는 `semantic_candidates/review-all-current.json`, `semantic_candidates/state.json`에서 확인한다. 이 판단은 잠정 AI 검토이며 사람 검수가 아니다. 불확실한 사례와 어색한 표현이 남아 있어 최종 자료 품질은 미판정이다.

학습 사건 200개와 개발 사건 75개를 생성하고 AI로 전체 의미를 잠정 검토했다. 필요한 출력 17개를 재생성해 확인한 뒤, 고정 기반 모델에서 q/k/v/o·마진 0.2 설정 하나를 새로 학습했다. 개발 자료로 300단계 체크포인트와 군집 거리 기준값 0.4를 골라 고정했다. 개발 검색 Recall@5는 70/75(93.3%), 전체 사건 쌍 F1은 1.000이다. 이전 v1·v2 자료의 검색 Recall@5는 각각 53/140(37.9%)와 54/140(38.6%)로 낮았다. 새 v3 시험 평가에서 LoRA 모델은 질문 250개 중 210개를 상위 5개에 포함해 Recall@5 0.840, 전체 사건 쌍 F1 0.832를 기록했다. 기반 모델은 같은 자료에서 각각 0.872와 0.483을 기록했다. 검색 목표 0.90에는 미달했고 군집 목표 0.80은 넘었다. `test-0010`의 시간 순서 오류와 사람 검수 미완료로 모든 시험 지표는 잠정 진단이다. 자세한 비교는 [v3 실험 결과](RESULTS_v3.md)에 있다.

v3 [합성 데이터셋](https://huggingface.co/datasets/heecchoi/slm-wiki-embeddinggemma2-text-synthetic-v3/tree/27e1536b4bcb769158e12ed6bc566e8dcfd5071b)과 [시험 결과를 반영한 LoRA 어댑터](https://huggingface.co/heecchoi/slm-wiki-embeddinggemma2-ko-lora-v3/tree/2dbd069c9cdf56723372718f5e3b8d4102213ef3)를 Hugging Face 비공개 저장소에 올렸다. 내려받은 데이터셋은 전체 자동 검증을 다시 통과했고 모델 파일의 해시도 모두 일치했다. 두 보관본은 잠정 상태다. 시험 자료의 의미 검수와 사람 검수는 남아 있다.

시험 생성의 누적 실행 한도는 사용자 승인으로 24시간에서 32시간으로 늘렸다. `test-0012`가 일반 복구 한도에서 실패해 이 사건에만 추가 난수 공간 하나를 사용했다. 이전 실패 응답과 예외 기록은 보존했고, 모델 가중치와 개발 자료에서 고른 설정은 바꾸지 않았다.

## 로컬 군집 탐색 화면

```bash
uv sync --frozen --group viz --inexact
.venv/bin/streamlit run dashboard.py
```

브라우저에서 `http://127.0.0.1:8501`을 연다. 패키지와 시각화 캐시를 준비하면 오프라인에서도 사용할 수 있다. 화면을 조작할 때 임베딩 모델을 다시 실행하지 않는다. 점을 클릭하면 메모를 고른다. 여러 점은 지도 도구 모음의 사각형·올가미 선택으로 고른다. 서버는 현재 PC에서만 접속하도록 설정하며 사용 통계 수집은 꺼 둔다.

| 화면 | 확인할 내용 |
| --- | --- |
| 군집 지도 | PCA·t-SNE 지도에서 점이나 영역 선택 → 메모 원문·실제 사건·예측 묶음. 선택 묶음의 거리 히트맵과 태그 출처 |
| 오결합·과분할 | 다른 사건을 합친 경우와 같은 사건을 나눈 경우. 같은 날짜·주제·가족·정답값 조건으로 원문 대조 |
| 모델 비교 | 같은 데이터에서 문서 선택을 함께 강조하고 묶음 구성과 지표 비교. 서로 다른 revision은 독립된 지도에서 확인 |
| 개발 실험 | 개발 자료에서 보정·거리·연결·재분할 설정을 바꾸고 결과 확인. 시험 자료는 저장된 고정 결과 표시 |

2차원 좌표는 문서를 살펴보기 위한 투영이다. 점 사이의 화면상 거리로 실제 군집 품질을 판정하지 않는다. 지표와 거리 히트맵은 원래 벡터 공간에서 계산한다. 모델별 좌표 축도 다르므로 같은 문서가 화면에서 이동한 거리를 성능 향상으로 해석하지 않는다.

기존 평가를 탐색하려면 한 번 캐시를 만든다. 평가 때 사용한 모델의 해시와 데이터 파일의 해시를 확인한 뒤 벡터를 계산한다. 아래 명령으로 현재 v2 결과의 캐시를 만든다. 대시보드를 새로고침한 뒤 평가 실행 목록에서 선택한다.

```bash
.venv/bin/python visualize.py export --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --evaluation artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.json --output artifacts/shortcut-hardneg-v2-balanced/visualization/tuned
```

시각화 묶음에는 모델·자료의 해시, 문서 행 순서, 군집 설정과 학습 때 고정한 보정 통계를 저장한다. 다른 revision에서 같은 문서 ID가 다시 쓰여도 서로 연결하지 않는다. 본문에서 추출한 날짜와 합성 정답 메타데이터의 날짜를 화면에서 구분한다.

## v2 자료가 만들어지는 순서

1. 코드가 가상의 가족·날짜·사건·확인할 사실을 정한다. 정답 문서 위치는 주제·분할별로 d0~d3에 균형 배정한다. 사건의 절반은 가상 인물 이름을 쓰고 나머지 절반은 주어를 생략한다. 이름을 제목 앞에 일괄적으로 붙이지 않는다.
2. 로컬 Qwen3 8B가 사건 카드로 한국어 메모와 검색 질문을 쓴다. 문서와 질문에는 ISO 날짜와 정확한 한국어 날짜 표기를 섞는다. 상대 날짜와 대략적인 날짜는 쓰지 않는다. 지정된 문서 한 곳에만 답을 두고 나머지 메모는 같은 사건의 준비·후속 기록으로 만든다. 근거 구절이 지정 문서에 있는지 코드가 검사한다. 이어서 답을 짐작하게 하는 단서도 다른 메모에서 제거한다. 생성 단계의 1차 보정본과 단서 제거본을 따로 보관하고 두 단계의 수정 사항을 각각 `repairs`와 `curation_repairs`에 남긴다. 교사 모델의 보정 전 응답 전문은 따로 저장하지 않는다.
3. 가족·사건 단위로 학습 800건, 개발 300건, 고정 평가 1,000건을 나눈다. 고정 평가에는 검색 질문 140개와 문서 쌍 100개를 둔다. 검색용 세 쌍 200개와 군집용 세 쌍 800개를 만든다. 군집 음성 예시는 같은 가족의 다른 사건 400개, 같은 주제의 다른 가족 사건 240개, 날짜가 30일 이내이거나 사실값을 공유하는 다른 주제·가족 사건 160개로 구성한다.
4. 원본 EmbeddingGemma 2와 미세 조정한 텍스트 전용 모델을 같은 자료로 비교한다. 개발 자료로 군집 기준값과 모델 체크포인트를 고른다. 고정 평가 자료는 모델 선택에 사용하지 않는다.

질문 상위 5개 결과에 근거 문서가 있으면 검색 성공으로 센다. 목표는 고정 평가 질문 140개의 성공률 90% 이상이다. 같은 사건의 다른 메모와 근거 메모를 구별하는지 보려고 첫 결과의 정답률도 따로 기록한다. 문서 묶기는 관련·무관 문서 쌍 100개에서 F1 0.80 이상을 목표로 한다. F1은 잘못 묶은 경우와 놓친 경우를 함께 반영하는 점수다. 사람이 평가 정답을 원문과 대조하기 전의 결과는 잠정치다. PDF 추출, 생성형 답변, 스마트폰 저장 공간과 속도는 이 결과에 포함되지 않는다.

가족과 사건은 분할 사이에 겹치지 않지만 10가지 주제와 일부 정답 표현은 반복된다. 따라서 이 평가는 처음 보는 사건을 찾는 능력을 확인한다. 새로운 주제나 실제 사람이 쓰는 메모에서도 같은 점수가 나오는지는 확인하지 못한다.

이전 합성 자료에는 인물 이름과 첫 메모 위치를 이용하는 지름길이 있었다. 새 revision에서는 이름 유무와 정답 위치를 분할·주제별로 균형화한다. 이름만 찾는 기준선과 d0만 우선하는 기준선도 따로 기록해 남은 지름길을 확인한다. 상위 5개 성공률만으로 의미 검색 능력을 판정하지 않는다.

## v2 모델이 배우는 관계

검색 학습에는 질문, 정답 위치가 지정된 메모, 같은 사건이지만 답이 없는 다른 메모를 한 묶음으로 넣는다. 예를 들어 “소아과 어디로 가지?”라는 질문에는 병원 이름이 적힌 예약 메모가 정답이다. 준비물만 적힌 메모는 같은 사건이어도 검색 정답이 아니다. 문서 묶기 학습에는 같은 사건의 두 메모와 A/B/C 유형별 다른 사건 메모를 넣는다. 같은 가족의 다른 사건을 포함해 같은 인물 이름이 여러 사건을 하나로 합치지 않도록 학습한다. 모델의 작은 추가 가중치인 LoRA를 조정해 정답 쌍의 임베딩이 오답 쌍보다 가깝도록 한다.

학습에 쓰지 않은 개발 자료로 묶음의 거리 기준과 학습 시점을 고른다. 고정 평가 결과로 이 선택을 바꾸지 않는다. 묶음 이름은 형태소 분석으로 찾은 원문 명사 중 여러 메모에 나온 말을 우선 사용한다. 공통 후보가 없으면 대표 메모의 명사로 후보를 넓힌다. 이름의 출처인 메모 ID도 남긴다. 이 두 작업이 태그를 붙이는 간단한 규칙이다. 태그가 실제 묶음을 잘 설명하는지는 별도 사람 검토가 필요하다.

평가를 실행하면 `artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.tag_review.csv`에 크기가 다른 묶음 최대 30개의 태그와 예시 메모가 나온다. 이 표에서 태그가 묶음을 잘 설명하는지 살핀다. 이 검토는 관련·무관 문서 쌍의 F1 점수를 대신하지 않는다.

## v2 재현 실행

아래 명령은 모두 이 디렉터리에서 실행한다. Python 3.12, `uv`, Ollama가 필요하다. 패키지 버전은 `uv.lock`에 고정한다. `artifacts/`에는 합성 자료와 체크포인트가 저장되며 Git에는 올라가지 않는다.

```bash
uv sync --frozen --inexact
ollama pull qwen3:8b
.venv/bin/python generate.py --split train --max-events 50 --output artifacts/shortcut-hardneg-v2-balanced/raw_events.jsonl --max-hours 12
.venv/bin/python generate.py --output artifacts/shortcut-hardneg-v2-balanced/raw_events.jsonl --max-hours 12
.venv/bin/python repair_duplicates.py --input artifacts/shortcut-hardneg-v2-balanced/raw_events.jsonl --max-hours 12 --allow-test-repair
ollama stop qwen3:8b
.venv/bin/python curate.py --input artifacts/shortcut-hardneg-v2-balanced/raw_events.jsonl --output artifacts/shortcut-hardneg-v2-balanced/curated_events.jsonl --manifest artifacts/shortcut-hardneg-v2-balanced/curation.json
.venv/bin/python prepare.py --input artifacts/shortcut-hardneg-v2-balanced/curated_events.jsonl --output artifacts/shortcut-hardneg-v2-balanced/dataset
.venv/bin/python validate.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset
.venv/bin/python train.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --output artifacts/shortcut-hardneg-v2-balanced/trained
.venv/bin/python select.py --run artifacts/shortcut-hardneg-v2-balanced/trained --dataset artifacts/shortcut-hardneg-v2-balanced/dataset
.venv/bin/python evaluate.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --output artifacts/shortcut-hardneg-v2-balanced/evaluation/base.json
.venv/bin/python evaluate.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --adapter artifacts/shortcut-hardneg-v2-balanced/trained/adapter --output artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.json
```

첫 50개 사건은 생성 속도·형식을 확인하는 예비 실행이다. 같은 새 출력 파일에서 이어 만들 수 있다. 8B 모델이 실행되지 않거나 전체 생성 예상 시간이 12시간을 넘으면 `qwen3:4b`로 별도 파일을 처음부터 만든다. 두 모델의 자료를 합치지 않는다. `repair_duplicates.py`의 `--allow-test-repair`는 아직 사람 검수 전인 새 revision에만 사용한다. 중복 문장을 이름 접두어로 덮지 않고 해당 사건을 다시 생성한다. 생성과 학습의 실행 시간은 합계 24시간 이내로 기록한다. 두 모델이 동시에 메모리에 남지 않도록 생성 후 `ollama stop`을 실행한다. 단서 제거 규칙은 `curate.py`에서 관리한다. `curation.json`에는 1차 보정본과 단서 제거본의 해시를 남긴다.

## v2 정답 검수와 비공개 보관

`artifacts/shortcut-hardneg-v2-balanced/dataset/review.csv`에는 고정 평가 질문 140개와 문서 쌍 100개가 들어 있다. 사람이 원문을 확인한 항목의 `accepted`에 `yes`를 적는다. 틀린 정답은 자료를 고쳐 새 검수본을 만든다. 240개 항목이 모두 확인되기 전에는 정식 합격 판정을 내리지 않는다.

질문 행은 `question_or_left`의 질문을 읽고 `evidence_or_right`에 실린 같은 사건의 메모 4개를 대조한다. `gold`의 근거가 지정 위치의 문서에 있으며 다른 메모에는 같은 답이 없는지 확인한다. 문서 쌍 행은 두 메모가 같은 사건이면 `True`, 다른 사건이면 `False`가 맞는지 확인한다. 잘못된 행은 `comment`에 이유를 적고 `accepted`를 비워 둔다. `gold` 칸만 고치지 말고 원본 사건을 수정한 뒤 검수표를 다시 만들어야 한다.

```bash
.venv/bin/python review.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset
.venv/bin/python evaluate.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --output artifacts/shortcut-hardneg-v2-balanced/evaluation/base.json
.venv/bin/python evaluate.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --adapter artifacts/shortcut-hardneg-v2-balanced/trained/adapter --output artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.json
.venv/bin/python publish.py --dataset artifacts/shortcut-hardneg-v2-balanced/dataset --curated artifacts/shortcut-hardneg-v2-balanced/curated_events.jsonl --teacher-raw artifacts/shortcut-hardneg-v2-balanced/raw_events.jsonl --adapter artifacts/shortcut-hardneg-v2-balanced/trained/adapter --base-eval artifacts/shortcut-hardneg-v2-balanced/evaluation/base.json --tuned-eval artifacts/shortcut-hardneg-v2-balanced/evaluation/tuned.json --stage artifacts/shortcut-hardneg-v2-balanced/publish_stage
```

사람 검수 전 비공개 보관본을 올리려면 `publish.py --allow-provisional`을 실행한다. 이때 Dataset Card와 Model Card에 잠정 상태를 표시한다. 잠정 보관본은 공식 품질 판정에 사용할 수 없다. 평가 정답 240개를 확인하고 `review.py`를 통과한 자료는 `publish.py`를 추가 옵션 없이 실행해 검수 완료 revision으로 올린다.

`publish.py`는 합성 데이터와 LoRA 어댑터를 서로 다른 비공개 Hugging Face Hub 저장소에 올린다. LoRA는 기반 모델의 일부에 작은 학습 값을 덧붙이는 방식이다. 전체 revision은 저장소의 특정 버전을 가리키는 식별값이다. 이 값과 데이터 파일의 해시를 확인한다. 실제 사용자 메모·질문·로그는 생성 입력이나 업로드 대상에 넣지 않는다. Hub 인증 정보는 개발 PC에서만 사용한다.

## 이전 revision 기준선

- 가상 사건 525개에서 메모 2,100건을 만들었다. 분할·중복·근거 검사를 통과했으며 중복 사건 8개는 재생성했다.
- EmbeddingGemma 2 텍스트 인코더를 250단계 학습하고 2회차 체크포인트를 개발 자료로 골랐다. MPS는 Mac GPU 계산 경로다.
- 고정 평가에서 잠정 검색 Recall@5는 134/140(95.7%)이다. 문서 쌍 표본 F1은 0.990이지만 전체 사건 쌍 F1은 0.529다. 해석과 한계는 [이전 revision 잠정 결과](RESULTS.md)에 기록했다.
- 이 revision의 합성 데이터와 LoRA 어댑터는 비공개 Hub에 보관했다. 이 결과는 아래 개선 revision과 구분한다.

## 지름길 제거·Hard Negative 개선 revision

새 revision은 답이 들어가는 문서 위치, 인물 이름 사용 여부, 날짜 표기를 주제·분할별로 균형화한다. 군집 학습에는 같은 가족의 다른 사건을 포함해 인물 이름만으로 사건을 합치지 않도록 A/B/C 유형의 음성 예시를 50:30:20 비율로 넣는다. 생성·학습·평가 결과는 [개선 revision 결과](RESULTS_v2.md)에 기록한다. 사람 검수 전 지표는 잠정치로 표시한다.

현재 저장소: [합성 데이터](https://huggingface.co/datasets/heecchoi/slm-wiki-embeddinggemma2-text-synthetic-v1/tree/ddba19537f56966c5fbe352707b93e71f57ccdf0) · [LoRA 어댑터](https://huggingface.co/heecchoi/slm-wiki-embeddinggemma2-ko-lora-v1/tree/f51442c47345d2f3e9113c5ad77057e42d14ea59). 두 저장소 모두 비공개다.

[이전 revision 결과](RESULTS.md) · [v2 결과](RESULTS_v2.md) · [v3 결과](RESULTS_v3.md) · [POC 한 페이지 요약](../README.md) · [상세 PRD](../SLM_Wiki_PRD.md)

<!-- HUMANIZE-SUMMARY -->
<!--
{
  "route": "standard",
  "original_chars": 13182,
  "final_chars": 13190,
  "grade": "B",
  "self_check": "6/6 통과",
  "categories": [
    "A-13",
    "A-18",
    "C-11"
  ],
  "highlights": [
    "기존 학습 결과의 재사용 조건을 주어부터 읽도록 정리했다.",
    "모델·데이터 파일의 해시와 메타데이터의 날짜에 필요한 조사를 보탰다.",
    "연결어미 뒤 쉼표 2개를 덜어냈다."
  ],
  "preserved": [
    "기술 용어·수치·코드·명령어·옵션·해시·링크",
    "한다체·제목·절 구성·평가 조건·잠정 상태와 한계"
  ],
  "remaining_findings": [
    "절차 동사와 기술 용어의 반복은 재현 안내에 필요해 보존했다.",
    "A-7 자동 탐지 후보는 실제 계산·생성을 뜻해 보존했다."
  ],
  "change_rate": "0.21% — 게이트 OK, 수렴",
  "follow_up": {
    "scope": "v3 생성 재시도 안내",
    "run_id": "2026-10-29-002-a6ea",
    "route": "light",
    "change_rate": "0%",
    "gate": "OK, exit0",
    "grade": "B",
    "self_check": "6/6 통과"
  },
  "phase2_follow_up": {
    "scope": "v2 고정자료 phase2 실행 안내",
    "run_id": "2026-10-29-003-76d1",
    "route": "light",
    "change_rate": "1.93%",
    "grade": "B",
    "self_check": "6/6 통과",
    "gate": "OK, exit0"
  },
  "v4_follow_up": {
    "scope": "v3 자기 Stage 1 초안 재생성과 알려진 정답 후보 검사 안내",
    "run_id": "2026-10-29-004-059b",
    "route": "light",
    "change_rate": "1.05%",
    "grade": "B",
    "self_check": "6/6 통과",
    "gate": "OK, exit0"
  },
  "human_clinic_follow_up": {
    "scope": "합성 소아과 환자의 사람 전제와 범주 한정 동물 표현 검사 및 한계",
    "run_id": "2026-10-29-007-df9c",
    "route": "light",
    "change_rate": "0%",
    "grade": "B",
    "self_check": "6/6 통과",
    "gate": "OK — exit 0",
    "findings": []
  },
  "retry_schedule_follow_up": {
    "scope": "새 Stage 1 시도의 기본 작성과 자기 초안 교대 및 출처 마커",
    "run_id": "2026-10-29-009-9829",
    "route": "light",
    "change_rate": "0%",
    "grade": "B",
    "self_check": "6/6 통과",
    "gate": "OK — exit 0",
    "findings": []
  },
  "semantic_gate_follow_up": {
    "run_id": "2026-10-29-011-a7b9",
    "route": "light",
    "genre": "report",
    "strength": "conservative",
    "change_rate": "0.00%",
    "gate": "OK — exit 0",
    "gate_source": "verify_gates.py JSON change_rate.rate",
    "grade": "B",
    "self_check": "6/6 통과",
    "findings": [],
    "preserved": [
      "사실·수치·해시·기록 경로·명령·서법·잠정 상태·단계별 실행 순서"
    ],
    "categories": [],
    "highlights": [
      "수정 없음. 이미 좋은 글."
    ],
    "remaining_warnings": [],
    "scope": "168개 전체 AI 의미 검토와 13사건·14출력 재생성 및 275개에서 학생 학습 전 정지하는 실행 절차",
    "full_document_prior_body_preserved": true
  },
  "v3_training_follow_up": {
    "run_id": "2026-10-29-013-18f2",
    "route": "heavy",
    "genre": "report",
    "strength": "conservative",
    "change_rate": "0.00%",
    "gate": "OK — exit 0",
    "grade": "B",
    "self_check": "6/6 통과",
    "scope": "275개 사건 잠정 검수, 새 v3 모델 개발 선택과 v1·v2 회귀",
    "categories": [],
    "highlights": [
      "기존 문장은 보존했다"
    ],
    "preserved": [
      "수치·날짜·revision·경로·링크·미판정·사람 검수 전 잠정 상태"
    ],
    "remaining_findings": []
  },
  "budget_extension_follow_up": {
    "run_id": "2026-10-29-015-2be0",
    "route": "heavy",
    "genre": "report",
    "strength": "conservative",
    "change_rate": "0.00%",
    "gate": "OK — exit 0",
    "grade": "B",
    "self_check": "6/6 통과",
    "scope": "누적 32시간 승인, test-0012 예외 복구와 초반 시험 자료 잠정 검토",
    "categories": [],
    "highlights": [
      "기존 문장은 보존했다"
    ],
    "preserved": [
      "24·32시간, 24건 실패, 공간 3000·5회차, 검토 10·2·1, 모델 선택 동결, 시험 미판정"
    ],
    "remaining_findings": []
  },
  "generation_completion_follow_up": {
    "run_id": "2026-10-30-002-a649",
    "route": "light",
    "genre": "report",
    "change_rate": "14.61%",
    "gate": "OK — exit 0",
    "grade": "B",
    "self_check": "6/6 통과",
    "scope": "시험 자료 생성·자동 검증 완료 상태 문장",
    "highlights": [
      "반복된 완료 상태를 합치고 모델 시험 평가 미실행을 분리"
    ],
    "preserved": [
      "시험 사건 250개, 자동 검증 통과, 의미·사람 검수 대기, 시험 성능 미판정"
    ],
    "remaining_findings": []
  }
}
-->
