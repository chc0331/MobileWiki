# Android 메모 검색·묶음 POC

Android 12 이상 ARM64 기기에서 개인 메모의 임베딩 검색, 낱말 검색, 결합 검색과 메모 묶음을 확인하는 로컬 POC입니다. 앱에는 인터넷 권한이 없고, 메모·평가 자료·피드백은 앱 전용 저장소에 남습니다. Android 자동 백업과 기기 이전 대상에서도 제외합니다.

## 7일 사용 안내

1. `개인` 공간에 평소 메모를 몇 개 작성하고 제목과 본문을 수정·삭제해 봅니다.
2. `검색` 화면에서 같은 질문을 임베딩·낱말·결합 방식으로 찾아봅니다. 검색 결과가 관련 있으면 👍, 아니면 👎를 남깁니다.
3. `묶음` 화면에서 자동으로 모인 메모를 확인합니다. 잘못 합쳐졌거나 나뉜 경우 👎 또는 👍를 남깁니다.
4. `합성 샘플` 공간에서 고정 평가 자료와 분리된 예시 메모를 불러와 같은 흐름을 확인합니다.
5. `평가` 화면에서 고정 자료 지표와 기기 성능을 직접 실행합니다. 자동 정답은 사람 검수 전 잠정치로 표시됩니다.
6. 7일 동안 매일 실제 메모를 검색하고 묶음 결과를 검토합니다. 날짜별 의견 수와 마지막 지연 측정은 `최근 7일 사용 요약`에서 확인합니다.
7. 필요할 때만 `측정 보고서 내보내기`를 눌러 기기·모델·데이터 revision과 지표를 저장합니다. 보고서에는 메모 본문이나 검색어가 들어가지 않습니다.

## 빌드

모델 변환 도구는 `poc/embedding/.venv`의 Python 환경과 로컬 학습 어댑터를 사용합니다. 고정 평가 자료와 모델을 만든 뒤 Android 폴더에서 APK를 빌드합니다.

```bash
python3 poc/android/tools/package_eval_data.py
PYTHONPATH=poc/embedding poc/embedding/.venv/bin/python poc/android/tools/export_embedding_model.py --precision int8_embedding
cd poc/android
./gradlew assembleDebug
```

APK 위치:

```text
poc/android/app/build/outputs/apk/debug/app-debug.apk
```

APK에는 ONNX 모델과 토크나이저가 포함됩니다. 모델 파일은 크기 때문에 Git에서 제외합니다. 앱은 ONNX Runtime CPU를 사용해 비행기 모드에서 실행하며, 모델 파일과 토크나이저의 SHA-256, 기준 모델 revision, 선택된 어댑터 checkpoint를 확인합니다. 검증된 학습 모델을 불러오지 못하면 기본 모델로 대체하지 않습니다.

LiteRT 변환본은 기준 임베딩과 일치하지 않아 사용하지 않습니다. Android용 ONNX Runtime CPU 모델은 행별 int8 토큰 임베딩과 float32 Transformer로 내보냈고, Python ONNX Runtime에서 패딩·동적 길이 입력 모두 기준 임베딩과 코사인 유사도 0.999 이상인지 확인했습니다. ONNX 모델과 토크나이저를 비공개 Hub 저장소 revision `dbb3bdd252f9034681cb65efee5d7f4b8cbef7bf`에 올렸고, artifact revision과 파일 해시를 기록한 최신 저장소 revision은 `526a59e58e244680fd076a58e646cdd3f4efd1bc`입니다. 앱 manifest에는 소스 모델·artifact revision과 파일 해시를 기록합니다.

## 확인하지 않은 항목

연결된 실기기나 에뮬레이터가 없는 환경에서는 설치 후 오프라인 동작, 실제 CPU 지연, 추가 PSS, Android 백업·이전 동작을 검증할 수 없습니다. POC가 생성하는 결과는 사용자 검토와 대상 기기 측정 전까지 잠정 또는 미판정입니다.
