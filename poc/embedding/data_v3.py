"""Versioned, context-rich synthetic cards and blind two-stage generation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from common import (
    DATE_PATTERN, DATE_STYLES, DOCUMENTS_PER_EVENT, EventCard, FAMILY_OFFSETS,
    FIRST_SYLLABLES, LAST_SYLLABLES, RELATIONSHIP_PATTERN, RELATIVE_DATE_PATTERN,
    SEED, SPLIT_EVENTS, SURNAMES, TOPICS, format_event_date, normalize_event_dates,
    read_jsonl, sha256, verified_dataset_manifest, write_jsonl,
)

REVISION = "context-hardneg-v3"
PROMPT_VERSION = "qwen-event-v3-blind-context-r12"
CURATION_VERSION = "validate-context-v3"
STYLES = ("개조식 요약", "체크리스트", "구어체 공유 메시지", "서술형 기록")
RELATIONSHIP_MENTION_PATTERN = re.compile(r"(?<![가-힣])" + RELATIONSHIP_PATTERN.pattern)
INTERNAL_MARKER_PATTERN = re.compile(r"event_id|family_id|fact_label|fact_value|context_keywords|secondary_context|answer_doc_index|identity_style|date_style|body_date|situation_any|purpose_any|\bJSON\b", re.IGNORECASE)
INSTRUCTION_ECHO_PATTERN = re.compile(r"배경\s*활동|중심\s*사건|상황\s*후보|목적\s*후보|공유\s*메시지처럼|개조식\s*요약|사실값|구체값|장소로만\s*언급")
RELATIVE_DAY_PATTERN = re.compile(r"(?<![가-힣])(?:그\s*)?(?:다음|지난|전)\s*날(?!짜)|이튿날")
CLINIC_ANIMAL_PATTERN = re.compile(r"반려\s*(?:견|묘|동물)|강아지|고양이|동물\s*병원|수의사")

# Values are fictional or ordinary objects. Short cues deliberately exclude topic words.
FACTS = (
    (("푸른숲 소아과", "푸른숲"), ("햇살 소아과", "햇살"), ("봄빛 소아과", "봄빛"), ("달빛 소아과", "달빛"), ("새싹 소아과", "새싹"), ("다온 소아과", "다온"), ("별나무 소아과", "별나무"), ("온유 소아과", "온유"), ("맑은샘 소아과", "맑은샘"), ("해솔 소아과", "해솔"), ("초록별 소아과", "초록별"), ("산들 소아과", "산들")),
    (("실내화", "실내화"), ("여벌 양말", "양말"), ("그림책", "그림책"), ("손수건", "손수건"), ("물통", "물통"), ("색연필", "색연필"), ("앞치마", "앞치마"), ("수건", "수건"), ("우비", "우비"), ("도시락", "도시락"), ("스케치북", "스케치북"), ("모자", "모자")),
    (("바람길 펜션", "바람길"), ("하늘마루 호텔", "하늘마루"), ("파도빛 게스트하우스", "파도빛"), ("솔바람 민박", "솔바람"), ("노을정원 숙소", "노을정원"), ("달그림 호텔", "달그림"), ("은하수 펜션", "은하수"), ("물결숲 민박", "물결숲"), ("해담 게스트하우스", "해담"), ("푸른뜰 펜션", "푸른뜰"), ("산마루 호텔", "산마루"), ("별채움 민박", "별채움")),
    (("주간 보고서", "주간 보고서"), ("고객 문의 목록", "고객 문의 목록"), ("회의 자료", "회의 자료"), ("계약 검토표", "계약 검토표"), ("납품 일정표", "납품 일정표"), ("재고 현황표", "재고 현황표"), ("견적 비교표", "견적 비교표"), ("업무 진행표", "업무 진행표"), ("출장 정산서", "출장 정산서"), ("작업 요청서", "작업 요청서"), ("행사 운영안", "행사 운영안"), ("교육 안내문", "교육 안내문")),
    (("서쪽 보건소", "서쪽"), ("중앙 의원", "중앙"), ("동네 보건지소", "보건지소"), ("나래 의원", "나래"), ("보람 보건소", "보람"), ("해든 의원", "해든"), ("이음 보건소", "이음"), ("가온 의원", "가온"), ("새봄 보건센터", "새봄"), ("한결 의원", "한결"), ("소담 보건센터", "소담"), ("도란 의원", "도란")),
    (("학교 강당", "강당"), ("운동장", "운동장"), ("도서관", "도서관"), ("음악실", "음악실"), ("미술실", "미술실"), ("체육관", "체육관"), ("과학실", "과학실"), ("시청각실", "시청각실"), ("다목적실", "다목적실"), ("회의실", "회의실"), ("급식실", "급식실"), ("교실", "교실")),
    (("와이퍼 교체", "와이퍼"), ("타이어 점검", "타이어"), ("엔진오일 교환", "엔진오일"), ("배터리 점검", "배터리"), ("전조등 교체", "전조등"), ("에어컨 필터 교체", "필터"), ("브레이크 점검", "브레이크"), ("냉각수 보충", "냉각수"), ("방향지시등 수리", "방향지시등"), ("안전벨트 점검", "안전벨트"), ("사이드미러 조정", "사이드미러"), ("경적 수리", "경적")),
    (("동네 도서관", "도서관"), ("하천 산책로", "산책로"), ("시민 공원", "공원"), ("느티나무 광장", "느티나무"), ("마을 회관", "회관"), ("문화 센터", "문화 센터"), ("벚꽃 쉼터", "벚꽃"), ("돌담 카페", "돌담"), ("햇무리 서점", "햇무리"), ("숲길 정자", "정자"), ("강변 나루터", "나루터"), ("고요한 찻집", "찻집")),
    (("주방 수도꼭지", "수도꼭지"), ("현관 조명", "조명"), ("욕실 환풍기", "환풍기"), ("방문 손잡이", "손잡이"), ("베란다 방충망", "방충망"), ("거실 콘센트", "콘센트"), ("창문 경첩", "경첩"), ("싱크대 배수관", "배수관"), ("수납장 선반", "선반"), ("현관 도어록", "도어록"), ("커튼 봉", "커튼 봉"), ("욕실 거울", "거울")),
    (("예산 초안", "예산 초안"), ("발표 슬라이드", "슬라이드"), ("회의록", "회의록"), ("일정 조정안", "일정 조정안"), ("설문 결과표", "설문 결과표"), ("운영 계획서", "운영 계획서"), ("비용 산출표", "비용 산출표"), ("성과 요약서", "성과 요약서"), ("기능 검토서", "기능 검토서"), ("고객 제안서", "고객 제안서"), ("실험 기록표", "실험 기록표"), ("채용 안내서", "채용 안내서")),
)

# Each situation is used in several events; combinations, rather than a copied ID,
# distinguish events. Context never contains the answer pool or hidden identifiers.
SITUATIONS = (
    (("발표 연습 중 목소리가 자주 잠기는 상황", "발표", "목소리"), ("달리기 뒤 기침이 이어진 상황", "달리기", "기침"), ("밤에 코막힘 때문에 잠을 설친 상황", "코막힘", "잠"), ("새 학기 전에 성장 상태를 확인하려는 상황", "학기", "성장"), ("소풍을 앞두고 귀의 불편함을 확인하려는 상황", "소풍", "귀"), ("식사 뒤 배가 불편하다는 이야기를 들은 상황", "식사", "배"), ("그림 그리기 중 손목이 불편했던 상황", "그림", "손목"), ("물놀이 뒤 피부 상태를 살펴보려는 상황", "물놀이", "피부")),
    (("씨앗을 관찰하는 활동을 앞둔 상황", "씨앗", "관찰"), ("인형극 역할을 맡게 된 상황", "인형극", "역할"), ("비 오는 날 바깥 활동이 잡힌 상황", "바깥", "우천활동"), ("나뭇잎을 모으는 활동이 예정된 상황", "나뭇잎", "모으기"), ("생일 축하 순서를 준비하는 상황", "생일", "축하"), ("종이로 마을을 만드는 활동을 앞둔 상황", "종이", "마을"), ("시장 놀이의 가게 역할을 맡은 상황", "시장", "가게"), ("동물 움직임을 흉내 내는 활동이 예정된 상황", "동물", "움직임")),
    (("철새를 관찰하는 여행을 준비하는 상황", "철새", "관찰"), ("야간 별자리 구경을 계획한 상황", "야간", "별자리"), ("도자기 체험을 함께 예약하려는 상황", "도자기", "체험"), ("옛 성곽을 걸어 보려는 상황", "성곽", "걷기"), ("해돋이를 보려는 여행을 준비하는 상황", "해돋이", "여행"), ("과수원 수확 체험을 계획한 상황", "과수원", "수확"), ("배편으로 섬을 둘러보려는 상황", "섬", "배편"), ("온천에 들르는 짧은 여행을 준비하는 상황", "온천", "여행")),
    (("이사 준비로 자리를 비우려는 상황", "이사", "준비"), ("자격 시험 응시로 쉬려는 상황", "자격", "시험"), ("공연 관람 때문에 일정을 조정하는 상황", "공연", "관람"), ("반려동물 돌봄으로 하루 쉬려는 상황", "반려동물", "돌봄"), ("집중 휴식을 위해 연차를 내는 상황", "집중", "휴식"), ("등산 모임 참여로 자리를 비우려는 상황", "등산", "모임"), ("가구 배송을 받아야 하는 상황", "가구", "배송"), ("학습 강좌에 참여하려는 상황", "학습", "강좌")),
    (("해외 출장을 앞두고 접종 일정을 확인하는 상황", "해외", "출장"), ("단체 생활을 시작하기 전에 접종을 확인하는 상황", "단체", "생활"), ("지난 접종 기록을 분실해 확인하려는 상황", "분실", "기록"), ("주말 봉사 참여 전에 일정을 확인하는 상황", "주말", "봉사"), ("장기 여행을 준비하며 접종을 확인하는 상황", "장기", "여행"), ("숙소 합숙을 앞두고 일정을 확인하는 상황", "숙소", "합숙"), ("새 직장 안내를 받고 접종을 확인하는 상황", "직장", "안내"), ("지역 행사 참여 전에 접종을 확인하는 상황", "지역", "참여")),
    (("합창 발표 순서를 맡은 상황", "합창", "발표"), ("독서 토론의 사회를 맡은 상황", "독서", "토론"), ("사진 전시 작품을 내기로 한 상황", "사진", "전시"), ("과학 발표를 시연하려는 상황", "과학", "시연"), ("줄넘기 순서를 준비하는 상황", "줄넘기", "순서"), ("연극 소품을 옮겨야 하는 상황", "연극", "소품"), ("진로 체험 질문을 준비하는 상황", "진로", "질문"), ("학교 축제 안내를 맡은 상황", "축제", "안내")),
    (("장거리 운전을 앞두고 정비를 예약한 상황", "장거리", "운전"), ("중고차 인수를 앞두고 상태를 확인하는 상황", "중고차", "인수"), ("주차 중 이상한 소리를 들은 상황", "주차", "소리"), ("야간 이동을 앞두고 상태를 확인하는 상황", "야간", "이동"), ("눈길 여행 전에 정비를 예약한 상황", "눈길", "여행"), ("출근 경로가 길어져 점검을 예약한 상황", "출근", "경로"), ("오래 보관한 차를 다시 쓰려는 상황", "보관", "재운행"), ("캠핑 물품을 싣기 전에 상태를 확인하는 상황", "캠핑", "물품")),
    (("보드게임을 가져와 함께 놀기로 한 상황", "보드게임", "놀이"), ("사진을 서로 보여 주기로 한 상황", "사진", "보여"), ("독서 감상을 나누기로 한 상황", "독서", "감상"), ("걷기 모임의 출발을 준비하는 상황", "걷기", "출발"), ("작은 선물을 주고받기로 한 상황", "선물", "교환"), ("취미 수업을 함께 알아보기로 한 상황", "취미", "수업"), ("여행 이야기를 나누기로 한 상황", "여행", "이야기"), ("오래된 편지를 함께 읽기로 한 상황", "편지", "읽기")),
    (("손님 방문 전에 집을 정리하려는 상황", "손님", "방문"), ("이사 전에 집 상태를 정리하는 상황", "이사", "상태"), ("재택 작업 공간을 바꾸려는 상황", "재택", "작업"), ("큰 청소를 하다 불편을 발견한 상황", "청소", "불편"), ("휴가 전에 집을 비울 준비를 하는 상황", "휴가", "비울"), ("반려동물 공간을 정리하려는 상황", "반려동물", "공간"), ("오래된 가구 배치를 바꾸는 상황", "가구", "배치"), ("겨울 준비로 집 안을 살펴보는 상황", "겨울", "준비")),
    (("신규 서비스 검토 모임을 앞둔 상황", "신규", "서비스"), ("고객 설명회를 준비하는 상황", "고객", "설명회"), ("분기 성과를 논의하려는 상황", "분기", "성과"), ("신입 교육 순서를 맡은 상황", "신입", "교육"), ("공급 업체와 조건을 논의하는 상황", "공급", "업체"), ("사내 행사 역할을 정하는 상황", "사내", "행사"), ("실험 결과를 공유하려는 상황", "실험", "결과"), ("업무 절차를 줄이는 모임을 준비하는 상황", "절차", "줄이기")),
)
PURPOSES = (
    ("오전 일정이 끝나자마자 이동할 예정이라 동선을 짧게 잡는다", "동선"),
    ("대중교통을 이용하므로 환승 시간을 넉넉히 둔다", "환승"),
    ("연락이 늦어질 수 있어 미리 내용을 공유한다", "연락"),
    ("짐을 적게 가져가야 해서 작은 가방 하나로 준비한다", "가방"),
    ("오후에 다른 약속이 있어 마치는 시간을 확인한다", "오후"),
    ("길을 모르는 동행자에게 이동 경로를 설명해 준다", "동행"),
    ("우천 가능성을 생각해 이동 계획을 다시 확인한다", "우천"),
    ("처음 하는 일정이라 빠진 준비가 없는지 적어 둔다", "처음"),
)

# Alternative design cue families: alternatives preserve meaning without requiring a
# shared literal pair in every note and question.
SITUATION_CUES = (
    (
        ('발표', '발성 연습', '목소리 잠', '목이 잠'),
        ('달리기', '조깅', '뛴 뒤', '뛰고 난'),
        ('코막힘', '코가 막', '막힌 코', '코가 답답'),
        ('학기', '새 학년', '성장 확인', '성장 점검'),
        ('소풍', '나들이 전 귀', '귀가 불편', '귀 불편'),
        ('식사', '먹은 뒤 배', '배가 불편', '복부 불편'),
        ('그림', '드로잉', '손목이 불편', '손목 불편'),
        ('물놀이', '수영 뒤', '피부 상태', '물에서 논 뒤'),
    ),
    (
        ('씨앗', '씨의 모습', '종자 관찰'),
        ('인형극', '인형 공연', '인형 연기'),
        ('바깥', '야외 활동', '실외 활동'),
        ('나뭇잎', '잎사귀', '잎 모으기'),
        ('생일', '태어난 날', '축하 순서'),
        ('종이', '마을 만들', '마을 모형'),
        ('시장', '가게 놀이', '장보기 놀이', '상점 역할'),
        ('동물', '움직임 흉내', '몸짓 따라', '동작 흉내'),
    ),
    (
        ('철새', '새 관찰', '조류 관찰', '이동하는 새'),
        ('별자리', '별 구경', '밤하늘 관찰'),
        ('도자기', '도예', '도자기 체험'),
        ('성곽', '성벽', '옛 성 둘레', '옛 성길'),
        ('해돋이', '일출', '해 뜨는', '아침 해'),
        ('과수원', '과일 수확', '열매 따기'),
        ('섬', '배편', '배를 타고'),
        ('온천', '온천욕'),
    ),
    (
        ('이사', '집 옮기', '거처 이동'),
        ('자격 시험', '자격증 시험', '시험 응시'),
        ('공연', '무대 관람', '공연장 관람'),
        ('반려동물', '동물 돌봄'),
        ('집중 휴식', '휴식', '재충전', '온전히 쉬'),
        ('등산', '산행', '산에 오르', '산 걷기'),
        ('가구', '집기 배송', '가재도구 배달'),
        ('학습 강좌', '강좌', '수강'),
    ),
    (
        ('해외 출장', '외국 출장', '국외 출장'),
        ('단체 생활', '공동생활', '여럿이 지내'),
        ('분실', '기록을 잃', '기록이 없어', '이력을 못 찾'),
        ('주말 봉사', '자원봉사', '봉사 참여', '자원 활동'),
        ('장기 여행', '긴 여행', '오랜 여행', '장기간 여행'),
        ('숙소 합숙', '합숙', '함께 묵', '공동 숙박'),
        ('직장', '새 회사', '입사', '직장 합류'),
        ('지역 행사', '동네 행사'),
    ),
    (
        ('합창', '함께 노래', '모둠 노래', '단체 노래'),
        ('독서 토론', '책 토론', '읽은 책 토론'),
        ('사진', '사진전', '사진 전시', '촬영 작품'),
        ('과학', '실험 시연', '과학 시연', '과학 발표'),
        ('줄넘기', '줄을 넘'),
        ('연극', '무대 소품', '극 소품', '공연 소품'),
        ('진로', '직업 체험', '직업 탐색', '미래 직업'),
        ('축제', '학교 잔치'),
    ),
    (
        ('장거리', '먼 길 운전', '긴 거리 운행', '멀리 운전'),
        ('중고차', '중고 차량', '차를 인수'),
        ('주차', '차를 세울 때', '주차장 소리'),
        ('야간', '밤 운전', '어두운 길', '밤길 이동'),
        ('눈길', '눈 오는 길', '눈 덮인 길'),
        ('출근', '통근', '출퇴근 거리', '회사 가는 길'),
        ('보관', '오래 세워', '장기 주차', '한동안 안 탄'),
        ('캠핑', '야영', '야영 준비'),
    ),
    (
        ('보드게임', '탁상 놀이', '판놀이'),
        ('사진', '찍은 사진', '촬영물 공유', '사진 구경'),
        ('독서', '책 감상', '독후감', '읽은 책 이야기'),
        ('걷기', '도보 모임', '산책 출발', '걷는 모임'),
        ('선물', '선물 교환', '작은 선물', '주고받을 물건'),
        ('취미', '취미 수업', '문화 강좌', '함께 배울 수업'),
        ('여행 이야기', '여행 얘기', '여행담'),
        ('편지', '서신'),
    ),
    (
        ('손님', '방문객', '손님 맞이', '누군가 방문'),
        ('이사', '집 옮기', '새집 이동', '거처 이동'),
        ('재택', '집에서 일', '홈오피스', '집 안 작업실'),
        ('청소', '대청소'),
        ('휴가', '집 비우기', '휴가 떠나', '빈집 준비'),
        ('반려동물', '동물 공간', '함께 사는 동물'),
        ('가구 배치', '배치 변경', '가구 재배치'),
        ('겨울', '월동', '추위 대비', '겨울나기'),
    ),
    (
        ('신규 서비스', '새 서비스', '서비스 검토'),
        ('고객 설명회', '대고객 안내', '설명회 준비'),
        ('분기 성과', '분기 실적', '분기별 결과'),
        ('신입', '입사자 교육', '새 직원 교육'),
        ('공급 업체', '납품 업체', '협력사 조건'),
        ('사내 행사', '회사 행사', '직장 행사'),
        ('실험', '테스트 결과'),
        ('절차 간소화', '업무 흐름 개선', '일하는 순서', '프로세스 개선'),
    ),
)
PURPOSE_CUES = (
    ('동선', '이동 거리', '가까운 경로', '짧은 이동'),
    ('환승', '갈아타', '갈아 타', '차편을 바꿔'),
    ('연락', '미리 공유', '사전에 알려', '먼저 알려'),
    ('가방', '짐을 줄', '짐을 적게', '짐을 최소'),
    ('오후', '다음 약속', '뒤의 약속', '끝나고 약속'),
    ('동행', '함께 가는 사람', '길 안내', '경로 설명'),
    ('우천', '비가 올', '비 예보', '비 올 때'),
    ('처음', '첫 일정', '첫 참여', '낯선 일정'),
)

QUERY_INTENTS = {
    "소아과 예약": {"kind": "where", "terms": ("병원", "소아과", "진료", "장소"), "instruction": "진료를 받을 병원이나 장소가 어디인지 묻는다. '어디' 또는 '어느 병원/장소'를 사용한다. 준비물·날짜를 묻지 않는다.", "example": "진료 장소가 어디였지?"},
    "어린이집 준비": {"kind": "what", "terms": ("준비물", "챙", "가져", "물건"), "instruction": "어린이집이나 선생님이 안내·공지·지정한 준비물 한 가지가 무엇인지 묻는다. 안내 출처를 질문에 드러낸다.", "example": "어린이집에서 안내받은 준비물이 무엇이었지?"},
    "가족 여행": {"kind": "where", "terms": ("숙소", "묵", "숙박", "머물"), "instruction": "여행에서 묵을 숙소가 어디인지 묻는다. '숙소/묵/숙박/머물' 중 하나와 '어디' 또는 '어느 숙소'를 반드시 사용한다. 무엇을 준비할지 묻는 질문은 금지한다.", "example": "어느 숙소에서 묵기로 했지?"},
    "연차 계획": {"kind": "what", "terms": ("인계", "넘길", "넘기", "맡길"), "instruction": "쉬기 전에 동료에게 인계하거나 넘길 자료가 무엇인지 묻는다. '인계/넘길'과 '자료/문서'를 함께 사용한다.", "example": "인계할 자료가 무엇이었지?"},
    "예방접종 예약": {"kind": "where", "terms": ("접종", "방문", "보건", "의원", "장소"), "instruction": "예방접종을 받으러 방문할 곳이 어디인지 묻는다. '어디' 또는 '어느 장소/의원'을 사용한다. 접종 종류나 날짜를 묻지 않는다.", "example": "접종을 받을 장소가 어디였지?"},
    "학교 행사": {"kind": "where", "terms": ("행사", "열리", "진행", "장소"), "instruction": "학교 행사가 열릴 장소가 어디인지 묻는다. '어디' 또는 '어느 장소/곳'을 사용한다.", "example": "행사가 열리는 장소가 어디였지?"},
    "차량 정비": {"kind": "what", "terms": ("정비", "손보", "점검", "교환", "교체", "수리", "항목"), "instruction": "차에서 정비하거나 손볼 항목이 무엇인지 묻는다. '무엇/뭐/어떤 항목'을 사용한다. 장소·준비물을 묻지 않는다.", "example": "차에서 정비할 항목이 무엇이었지?"},
    "주말 약속": {"kind": "where", "terms": ("만나", "만남", "모이", "약속", "장소"), "instruction": "약속에서 만나기로 한 장소가 어디인지 묻는다. '어디' 또는 '어느 장소/곳'을 사용한다.", "example": "만나기로 한 장소가 어디였지?"},
    "집 수리": {"kind": "what", "terms": ("고치", "고칠", "수리", "대상", "손보"), "instruction": "집에서 수리하거나 고칠 대상이 무엇인지 묻는다. '무엇/뭐/어떤 대상'을 사용한다.", "example": "집에서 고칠 대상이 무엇이었지?"},
    "회사 일정": {"kind": "what", "terms": ("자료", "문서"), "instruction": "회사 일정에 준비해 갈 자료나 문서가 무엇인지 묻는다. '자료/문서'와 '무엇/뭐/어떤'을 함께 사용한다.", "example": "준비할 자료가 무엇이었지?"},
}

CENTRAL_EVENTS = {
    "소아과 예약": ("소아과에서 진료를 받을 예약이다. 배경 활동 중 생긴 불편함을 진료에서 확인한다.", ("진료", "진찰")),
    "어린이집 준비": ("어린이집 활동에 안내받은 준비물을 챙기는 사건이다. 배경 활동과 이동 준비는 별도 사정이다.", ("어린이집", "등원")),
    "가족 여행": ("가족 여행에서 묵을 숙소를 정한 사건이다. 여행 중 활동과 이동 사정은 숙소를 찾는 배경이다.", ("여행", "숙소", "숙박")),
    "연차 계획": ("배경에 적힌 개인 사정으로 연차를 쓰며, 쉬기 전에 동료에게 업무 자료를 인계한다.", ("연차", "인계", "휴가")),
    "예방접종 예약": ("예방접종을 받을 장소를 예약한 사건이다. 배경 활동은 접종 일정을 확인하는 사정이다.", ("접종",)),
    "학교 행사": ("학교 행사가 열릴 행사 장소를 확인한다. 배경 활동은 이 행사에서 맡은 역할이다.", ("학교", "행사")),
    "차량 정비": ("차량에서 정비할 항목을 확인하는 사건이다. 운전이나 이동 사정은 정비가 필요한 배경이다.", ("정비", "점검", "차량")),
    "주말 약속": ("사람과 만나기로 한 장소를 확인하는 사건이다. 배경 활동은 만난 뒤 함께 할 일이다.", ("약속", "만나", "만남", "모임")),
    "집 수리": ("집에서 고칠 대상을 확인하는 사건이다. 배경 활동은 집 수리가 필요한 사정이다.", ("수리", "고치", "고칠")),
    "회사 일정": ("회사 일정에 준비해 갈 자료를 확인하는 사건이다. 배경 활동은 그 자료가 필요한 업무다.", ("회사", "회의", "업무")),
}
NOTICE_CUES = ("안내", "공지", "지정", "알려", "챙겨 오라고", "챙겨오라고", "가져오라고", "가져 오라고")


def validate_query_intent(query: str, category: str) -> None:
    intent = QUERY_INTENTS[category]
    if category == "어린이집 준비" and '준비물' not in re.split(r'[,，]', query)[-1]:
        raise ValueError("Daycare question focus must ask for the notified preparation item")
    places = r"(?:곳|장소|병원|소아과|숙소|의원|보건소|보건센터|공간)"
    interrogative = bool(re.search(rf"어디|(?:어느|어떤)\s*{places}", query)) if intent["kind"] == "where" else bool(re.search(r"무엇|뭐|어떤|어느", query))
    if not interrogative or not any(term in query for term in intent["terms"]):
        raise ValueError("Question asks the wrong fact relation or intent")
    if category == "연차 계획" and not any(term in query for term in ("자료", "문서", "업무")):
        raise ValueError("Question asks the wrong fact relation or intent")
    notice_query = re.sub(r"(?:길|경로)\s*안내", "", query)
    if category == "어린이집 준비" and not any(term in notice_query for term in NOTICE_CUES):
        raise ValueError("Question asks the wrong fact relation or intent: daycare-notified item required")
    if category == "어린이집 준비" and re.search(r"(?:길\s*안내|경로\s*설명|환승|연락|동선)\s*용", query):
        raise ValueError("Question turns a travel constraint into the item's purpose")


def validate_question_uniqueness(queries: list[dict]) -> None:
    seen = {}
    for query in queries:
        text = query.get("text", query.get("query", ""))
        normalized = DATE_PATTERN.sub("<DATE>", text)
        if query.get("person"):
            normalized = normalized.replace(query["person"], "<PERSON>")
        key = (query["split"], re.sub(r"[\W_]+", "", normalized).casefold())
        event_id = query["event_id"]
        if key in seen and seen[key] != event_id:
            raise ValueError(f"Duplicate or ambiguous same-split questions: {seen[key]}/{event_id}")
        seen[key] = event_id


def _context_pairs(topic: int, split: str, count: int) -> list[tuple[int, int]]:
    rng = random.Random(_seed(topic, "latin-contexts"))
    situations, purposes, blocks = list(range(8)), list(range(8)), list(range(8))
    rng.shuffle(situations)
    rng.shuffle(purposes)
    rng.shuffle(blocks)
    def block(number: int):
        values = [(situations[situation], purposes[(situation + blocks[number]) % 8]) for situation in range(8)]
        random.Random(_seed(topic, "latin-order", number)).shuffle(values)
        return values
    available = {"train": block(0) + block(1) + block(6)[:4], "dev": block(2), "test": block(3) + block(4) + block(5) + block(7)[:1]}[split]
    if len(available) < count:
        raise ValueError("Not enough disjoint Latin context pairs")
    return available[:count]


@dataclass(frozen=True)
class ContextCard(EventCard):
    card_revision: str
    secondary_context: str
    context_keywords: tuple[str, ...]
    situation_cues: tuple[str, ...]
    purpose_cues: tuple[str, ...]
    document_styles: tuple[str, ...]
    answer_cues: tuple[str, ...]
    collision_group: str | None

    def as_dict(self) -> dict:
        output = asdict(self)
        for field in ("document_date_styles", "context_keywords", "situation_cues", "purpose_cues", "document_styles", "answer_cues"):
            output[field] = list(output[field])
        return output


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, REVISION, *parts))).encode()).digest()[:8], "big")


@lru_cache(maxsize=None)
def _topics_for(split: str) -> tuple[int, ...]:
    count = SPLIT_EVENTS[split]
    families = count // 5
    targets = [count // 10 + (category < count % 10) for category in range(10)]
    repeated = [family % 10 for family in range(families)]
    for attempt in range(5000):
        rng = random.Random(_seed(split, "topics", attempt))
        remaining = targets.copy()
        assignments = []
        for category in repeated:
            remaining[category] -= 2
            assignments.append([category, category])
        family_order = list(range(families))
        rng.shuffle(family_order)
        valid = True
        for family in family_order:
            for _ in range(3):
                choices = [category for category in range(10) if remaining[category] > 0 and category not in assignments[family]]
                if not choices:
                    valid = False
                    break
                category = max(choices, key=lambda category: (remaining[category] + rng.random() * 0.8, rng.random()))
                assignments[family].append(category)
                remaining[category] -= 1
            if not valid:
                break
        if not valid or any(remaining):
            continue
        if any(sum(category in values for values in assignments) < 5 for category in range(10)):
            continue
        for values in assignments:
            rng.shuffle(values)
        return tuple(category for values in assignments for category in values)
    raise RuntimeError(f"Unable to assign balanced family topics for {split}")


@lru_cache(maxsize=None)
def _cards_for(split: str) -> tuple[ContextCard, ...]:
    topics = _topics_for(split)
    members = {topic: [i for i, value in enumerate(topics) if value == topic] for topic in range(10)}
    fields: dict[int, dict] = {index: {} for index in range(len(topics))}
    group_count = int(len(topics) * 0.6) // 5
    topic_order = list(range(10))
    random.Random(_seed(split, "collision-topics")).shuffle(topic_order)
    group_alloc = {topic: 0 for topic in range(10)}
    for n in range(group_count):
        choices = [topic for topic in topic_order if group_alloc[topic] < len(members[topic]) // 5]
        topic = min(choices, key=lambda value: (group_alloc[value], topic_order.index(value)))
        group_alloc[topic] += 1
    weekend_topic = next(topic for topic, item in enumerate(TOPICS) if item[0] == "주말 약속")
    planning_order = [topic for topic in members if topic != weekend_topic] + [weekend_topic]
    for topic in planning_order:
        indices = members[topic]
        ordered = sorted(indices, key=lambda i: _seed(split, topic, "member", i))
        # Pick repeated values across different families, reserving 160 C-eligible train events.
        pair_count = 8 if split == "train" else len(indices) // 2
        available = ordered.copy()
        fact_values = FACTS[topic]
        fact_number = 0
        for _ in range(pair_count):
            left = available.pop(0)
            right = next(index for index in available if index // 5 != left // 5)
            available.remove(right)
            for index in (left, right):
                fields[index]["fact"] = fact_number
            fact_number += 1
        for index in available:
            fields[index]["fact"] = fact_number % len(fact_values)
            fact_number += 1
        date_rng = random.Random(_seed(split, topic, "dates"))
        days = list(range(650))
        date_rng.shuffle(days)
        if topic == weekend_topic:
            days = [offset for offset in days if (date(2025, 1, 1) + timedelta(days=offset)).weekday() >= 5]
        by_family: dict[int, list[int]] = defaultdict(list)
        for index in ordered:
            by_family[index // 5].append(index)
        date_number = 0
        def next_date(bucket: list[int]) -> str:
            nonlocal date_number
            while date_number < len(days):
                day = (date(2025, 1, 1) + timedelta(days=days[date_number])).isoformat()
                date_number += 1
                if topic != weekend_topic or all(
                    fields[sibling].get("date") != day
                    for index in bucket for sibling in range(index // 5 * 5, index // 5 * 5 + 5)
                    if sibling not in bucket
                ):
                    return day
            raise RuntimeError("No date remains for the event-family bucket")
        for group in range(group_alloc[topic]):
            family_choices = sorted(
                (family for family, values in by_family.items() if values),
                key=lambda family: (-len(by_family[family]), _seed(split, topic, group, family)),
            )
            if len(family_choices) < 5:
                raise RuntimeError("Collision group needs five distinct families")
            bucket = [by_family[family].pop(0) for family in family_choices[:5]]
            day = next_date(bucket)
            for index in bucket:
                fields[index].update(date=day, collision_group=f"{split}-collision-{topic}-{group}")
        for values in by_family.values():
            for index in values:
                fields[index].update(date=next_date([index]), collision_group=None)
        def assign(domain: str, values: tuple):
            sequence = (list(values) * ((len(ordered) + len(values) - 1) // len(values)))[:len(ordered)]
            random.Random(_seed(split, topic, domain)).shuffle(sequence)
            for index, value in zip(ordered, sequence):
                fields[index][domain] = value
        assign("answer", (0, 1, 2, 3))
        assign("identity", ("named", "implicit"))
        assign("query_date", ("none", "none", "iso", "korean_full"))
        for slot in range(4):
            assign(f"date_style_{slot}", DATE_STYLES)
        context_pairs = _context_pairs(topic, split, len(indices))
        for rank, index in enumerate(sorted(indices)):
            situation_index, purpose_index = context_pairs[rank]
            situation, word1, word2 = SITUATIONS[topic][situation_index]
            purpose, word3 = PURPOSES[purpose_index]
            fields[index]["context"] = f"{situation}. {purpose}."
            fields[index]["keywords"] = (word1, word2, word3)
            fields[index]["situation_cues"] = SITUATION_CUES[topic][situation_index]
            fields[index]["purpose_cues"] = PURPOSE_CUES[purpose_index]
    output = []
    for index, topic in enumerate(topics):
        item = fields[index]
        family_number = FAMILY_OFFSETS[split] + index // 5
        person = SURNAMES[family_number % 10] + FIRST_SYLLABLES[(family_number // 10) % 10] + LAST_SYLLABLES[(family_number // 100) % 10]
        styles = list(STYLES)
        random.Random(_seed(split, index, "styles")).shuffle(styles)
        fact, cue = FACTS[topic][item["fact"]]
        if cue in item["context"]:
            raise ValueError(f"Secondary context leaks the answer: {split}/{index}/{cue}")
        output.append(ContextCard(
            event_id=f"{split}-{index:04d}", split=split, family_id=f"{split}-family-{index // 5:03d}", person=person,
            category=TOPICS[topic][0], date=item["date"], fact_label="어린이집에서 안내받은 준비물" if topic == 1 else TOPICS[topic][1], fact_value=fact,
            answer_doc_index=item["answer"], identity_style=item["identity"],
            document_date_styles=tuple(item[f"date_style_{slot}"] for slot in range(4)), query_date_style=item["query_date"],
            card_revision=REVISION, secondary_context=item["context"], context_keywords=item["keywords"],
            situation_cues=item["situation_cues"], purpose_cues=item["purpose_cues"],
            document_styles=tuple(styles), answer_cues=(cue,), collision_group=item["collision_group"],
        ))
    return tuple(output)


def event_card_v3(split: str, index: int) -> ContextCard:
    if split not in SPLIT_EVENTS or not 0 <= index < SPLIT_EVENTS[split]:
        raise ValueError(f"Invalid v3 event {split}/{index}")
    return _cards_for(split)[index]


def stage1_payload(card: dict) -> dict:
    slot = card["answer_doc_index"]
    return {key: card[key] for key in ("category", "date", "fact_label", "fact_value", "secondary_context", "identity_style")} | {
        "person": card["person"] if card["identity_style"] == "named" else None,
        "slot": slot, "style": card["document_styles"][slot], "date_style": card["document_date_styles"][slot],
        "body_date": format_event_date(card["date"], card["document_date_styles"][slot]),
        "required_context": {"situation_any": card["situation_cues"], "purpose_any": card["purpose_cues"]},
        "central_event": CENTRAL_EVENTS[card["category"]][0],
    }


def stage2_payload(card: dict) -> dict:
    # Whitelist construction is intentional: never copy the full card and delete secrets.
    payload = {key: card[key] for key in ("category", "date", "fact_label", "secondary_context", "identity_style", "query_date_style")}
    payload["person"] = card["person"] if card["identity_style"] == "named" else None
    payload["required_context"] = {"situation_any": card["situation_cues"], "purpose_any": card["purpose_cues"]}
    payload["query_intent"] = QUERY_INTENTS[card["category"]]["instruction"]
    payload["central_event"] = CENTRAL_EVENTS[card["category"]][0]
    payload["documents"] = [
        {"slot": slot, "style": card["document_styles"][slot], "date_style": card["document_date_styles"][slot],
         "body_date": format_event_date(card["date"], card["document_date_styles"][slot])}
        for slot in range(4) if slot != card["answer_doc_index"]
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    if card["fact_value"] in serialized or any(cue in serialized for cue in card["answer_cues"]):
        raise ValueError("Blind stage payload contains the answer or an answer cue")
    return payload


def stage2_component_payload(payload: dict, component: str) -> dict:
    # Derive each fresh request only from the already blind parent whitelist.
    result = {key: payload[key] for key in ("category", "fact_label", "secondary_context", "identity_style", "person", "required_context", "central_event")}
    result["component"] = component
    if component == "query":
        result["query_date_style"] = payload["query_date_style"]
        result["query_intent"] = payload["query_intent"]
        if payload["query_date_style"] != "none":
            result["date"] = payload["date"]
    else:
        slot = int(component[1:])
        result["document"] = next(item for item in payload["documents"] if item["slot"] == slot)
    return result


def _wire_document_schema(item: dict, slot: int | None = None) -> dict:
    body = ({"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "minItems": 3, "maxItems": 3}
            if item["style"] in {"체크리스트", "개조식 요약"}
            else {"type": "string", "minLength": 40, "maxLength": 250})
    properties = {"text": body, "title": {"type": "string", "minLength": 1, "maxLength": 100}}
    if slot is not None:
        properties["slot"] = {"type": "integer", "const": slot}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def stage_schema(payload: dict, stage: int) -> dict:
    if stage == 1:
        return {"type": "object", "properties": {
            "document": _wire_document_schema(payload),
            "evidence": {"type": "string", "minLength": 1, "maxLength": 250},
        }, "required": ["document", "evidence"], "additionalProperties": False}
    if payload["component"] != "query":
        item = payload["document"]
        return {"type": "object", "properties": {"document": _wire_document_schema(item, item["slot"])},
                "required": ["document"], "additionalProperties": False}
    return {"type": "object", "properties": {"query": {"type": "string", "minLength": 8, "maxLength": 180}},
            "required": ["query"], "additionalProperties": False}


def _style_instruction(style: str) -> str:
    return {
        "체크리스트": "text는 체크 항목 세 개만 담는 JSON 문자열 배열이다. 각 항목은 반드시 '- [ ] '로 시작하고 내용이 있는 완성된 한 줄로 쓴다. 날짜는 첫 항목에 넣는다. 각 문자열 내부에 실제 줄바꿈을 넣지 않는다.",
        "개조식 요약": "text는 짧은 요약 항목 세 개만 담는 JSON 문자열 배열이다. 각 항목은 '- '로 시작하고 내용이 있는 완성된 한 줄로 쓴다. 날짜는 첫 항목에 넣는다. 각 문자열 내부에 실제 줄바꿈을 넣지 않는다.",
        "구어체 공유 메시지": "text 본문을 사람에게 보내는 자연스러운 공유 메시지 두세 문장으로 쓴다. 해요·했어·할게 같은 구어체를 사용한다.",
        "서술형 기록": "text 본문을 상황과 준비·후속 행동이 이어지는 서술형 문단으로 쓴다. 완성된 문장 세 개 정도를 사용한다.",
    }[style]


def _instructions(payload: dict, stage: int) -> str:
    if stage == 2:
        return _blind_component_instructions(payload)
    identity = "주어를 생략하고 인물 이름과 가족 관계 호칭을 쓰지 않는다." if payload["identity_style"] == "implicit" else f"이름 '{payload['person']}'을 본문에 자연스럽게 쓰되 제목에는 넣지 않는다."
    situation = " / ".join(payload["required_context"]["situation_any"])
    purpose = " / ".join(payload["required_context"]["purpose_any"])
    common = (
        "완전히 가상의 한국어 생활 메모를 작성하고 JSON만 출력한다. 중국어·한자·연락처·주민번호·의료 조언은 쓰지 않는다. "
        "각 메모 본문은 짧은 80~180자이며 지정한 형식으로 쓴다. 체크리스트와 개조식 요약의 text는 정확히 세 항목의 문자열 배열이고, 나머지 형식은 줄바꿈 없는 문자열이다. text를 먼저 완성하고 마지막에 이름 없는 title을 쓴다. 중심 사건의 행동을 메모와 질문에 드러낸다. 배경 상황은 이 사건이 필요한 사정이고, 이동·공유 목적은 일정 준비의 제약이다. 배경이나 이동 제약을 중심 사건으로 바꾸거나 새로운 원인으로 꾸미지 않는다. 진단·치료 조언·증상 예측·주어지지 않은 시각은 만들지 않는다. "
        "맥락을 버리고 '일정과 준비를 다시 확인한다' 같은 일반론만 쓰지 않는다. "
        f"상황 표현 후보 [{situation}]에서 하나 이상, 목적 표현 후보 [{purpose}]에서 하나 이상을 골라 뜻이 자연스럽게 이어지게 쓴다. 메모마다 다른 후보와 문장 구조를 사용하고 질문도 표현을 바꾼다. 특정 두 낱말을 모든 문서에 똑같이 반복하지 않는다. "
        f"중심 사건: {payload['central_event']} "
        "event_id 같은 내부 식별자나 새로운 사건 코드는 만들지 않는다. 상대 날짜 대신 정확한 사건 날짜를 쓰고, 지정한 날짜 표기를 지킨다. " + identity
    )
    if stage == 1:
        named_body = f"text 본문에 이름 '{payload['person']}'을 반드시 글자 그대로 한 번 넣는다. 이름을 title에만 쓰거나 생략하면 안 된다. " if payload["identity_style"] == "named" else "text와 title에서 인물 이름을 쓰지 않는다. "
        fact_relation = "어린이집 준비물은 어린이집이나 선생님이 안내·공지·지정해 알려 준 물건이다. 그 출처와 안내 사실을 정답 문장과 evidence에 함께 적는다. 개인적인 외출 준비물과 혼동하지 않는다. " if payload["category"] == "어린이집 준비" else ""
        if payload["category"] == "학교 행사":
            fact_relation = "fact_value는 학교 행사가 열리는 장소다. 정답 문장과 evidence에 그곳에서 행사가 열린다는 관계를 분명히 적는다. "
        return common + (
            f"\n답 메모 한 개를 작성한다. text 본문 안에 정확한 날짜 '{payload['body_date']}'를 반드시 한 번 이상 글자 그대로 넣는다. "
            "날짜를 title에만 쓰면 안 된다. fact_value를 text 본문 한 곳에 넣는다.\n입력: "
        ) + json.dumps(payload, ensure_ascii=False) + (
            f"\n최종 본문 조건: {_style_instruction(payload['style'])} "
            f"text 안에 '{payload['body_date']}'와 상황·목적 후보에서 고른 자연스러운 표현을 넣는다. "
            + named_body
            + fact_relation
            + "답은 fact_value의 문자열을 한 번 포함한다. "
            f"evidence에는 완성된 document.text에서 '{payload['fact_value']}' 문자열이 들어 있는 문장 또는 목록 한 항목을 그대로 복사한다. 첫 문장이 아니라 이 정답값이 든 문장을 고른다. 새로 요약하거나 조사·어미를 고치지 않는다. "
            "evidence의 모든 글자가 document.text의 연속된 부분과 정확히 같아야 한다."
        )


def _blind_component_instructions(payload: dict) -> str:
    situation = " / ".join(payload["required_context"]["situation_any"])
    purpose = " / ".join(payload["required_context"]["purpose_any"])
    common = (
        "가상의 한국어 생활 기록이다. 주어진 중심 사건을 유지하고 배경 활동이나 이동 사정으로 바꾸지 않는다. "
        "상황 후보와 목적 후보에서 각각 자연스러운 표현 하나 이상을 골라 쓴다. "
        "중국어·한자·의료 조언·새 사건 식별자·주어지지 않은 시각은 쓰지 않는다. "
        "fact_label의 값은 모른다. 그 값이나 다른 후보 장소·물건·자료 이름을 지어내지 않는다. "
        f"상황 후보: {situation}. 목적 후보: {purpose}.\n입력: " + json.dumps(payload, ensure_ascii=False)
    )
    if payload["component"] == "query":
        date_rule = "날짜는 쓰지 않는다." if payload["query_date_style"] == "none" else f"날짜 '{format_event_date(payload['date'], payload['query_date_style'])}'를 넣는다."
        identity = "이름과 가족 호칭을 쓰지 않는다." if payload["identity_style"] == "implicit" else f"이름 '{payload['person']}'을 넣는다."
        required_date = '날짜 없음' if payload['query_date_style'] == 'none' else format_event_date(payload['date'], payload['query_date_style'])
        required_identity = '주어 생략, 이름·가족 호칭 없음' if payload['identity_style'] == 'implicit' else payload['person']
        queried_fact = '동료에게 인계할 자료나 문서' if payload['category'] == '연차 계획' else payload['fact_label']
        return common + (
            "\n실제 사용자가 메모 검색창에 입력할 의문문 한 개를 query에 쓴다. 상황과 목적을 짧은 배경 구절로 연결하고 fact_label의 값 한 가지만 직접 묻는다. "
            "마지막은 '어디였지?', '무엇이었지?'처럼 의문형 어미와 물음표로 마친다. '묻는다', '확인한다', '궁금하다'처럼 질문을 설명하는 문장은 쓰지 않는다. "
            "질문을 반복하거나 두 번째 질문·부탁·답을 덧붙이지 않는다. 40~100자 정도로 자연스럽게 쓴다. "
            + payload["query_intent"] + " " + date_rule + " " + identity
            + "\n질문은 이동·연락·짐 준비의 사정을 말하는 짧은 배경 절로 시작한다. 첫 절에 목적 후보에서 고른 표현을 반드시 넣고 '~해야 하는데,'처럼 뒤의 질문과 구분한다. 이어서 상황을 짚고 중심 사건에서 모르는 사실 한 가지만 직접 묻는다. 이동 사정을 그 물건의 용도로 단정하지 않는다. 날짜와 이름 조건도 지킨다."
            + f"\n이번 질문에 모두 들어갈 정보: 첫 이동·연락 사정 [{' / '.join(payload['required_context']['purpose_any'])}] 중 하나; 별도의 활동 상황 [{' / '.join(payload['required_context']['situation_any'])}] 중 하나; 날짜 [{required_date}]; 인물 [{required_identity}]. 마지막에는 [{queried_fact}] 자체가 무엇인지 또는 어디인지 한 가지만 직접 묻는다."
        )
    item = payload["document"]
    identity = "이름과 가족 호칭을 쓰지 않는다." if payload["identity_style"] == "implicit" else f"이름 '{payload['person']}'은 text에만 넣고 title에는 쓰지 않는다."
    return common + (
        "\ndocument 한 개만 쓴다. 중심 사건의 준비·이동·공유·후속 확인 중 한 관점으로 80~180자 정도를 적는다. "
        f"정확한 날짜 '{item['body_date']}'를 text 첫 문장이나 첫 항목에 자연스럽게 넣는다. "
        "사실값은 별도 기록에서 확인할 상태로 두고 새 후보값을 열거하지 않는다. "
        f"구체값을 모르는 사항은 '{payload['fact_label']}'라는 일반명으로 지칭한다. "
        + _style_instruction(item["style"]) + " " + identity + " text를 완성한 뒤 짧은 title을 쓴다. JSON 객체 하나만 출력한다."
    )


def _parse_json(response: str) -> dict:
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Teacher response has no JSON object")
    return json.loads(response[start:end + 1])


def _matches_known_answer(text: str, candidates: list[str]) -> bool:
    compact = "".join(text.split())
    return any("".join(candidate.split()) in compact for candidate in candidates)


def _validate_document(document: dict, card: dict, slot: int, answer: bool) -> dict:
    title, text = document.get("title"), document.get("text")
    if isinstance(text, list):
        if card["document_styles"][slot] not in {"체크리스트", "개조식 요약"} or len(text) != 3 or any(not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line for line in text):
            raise ValueError("List document needs exactly three complete single-line items")
        text = "\n".join(text)
    if not isinstance(title, str) or not title.strip() or not isinstance(text, str):
        raise ValueError("Missing document title or text")
    title, text = title.strip(), text.strip()
    if card["category"] == "소아과 예약" and any(CLINIC_ANIMAL_PATTERN.search(part) for part in (title, text)):
        raise ValueError("Clinic fact relation changed to an animal patient or veterinarian")
    if len(title) > 100:
        raise ValueError("Title length outside 1–100 characters")
    if INTERNAL_MARKER_PATTERN.search(title + text):
        raise ValueError("Internal schema marker in document")
    if INSTRUCTION_ECHO_PATTERN.search(title + text):
        raise ValueError("Generation instruction echo in document")
    if not 40 <= len(text) <= 500:
        raise ValueError("Document length outside 40–500 characters")
    if re.search(r"(?:^|\n)\s*(?:[-*□☐✓✔]|[1-9][.)])\s*(?:\[\s*\])?\s*(?=\n|$)", text):
        raise ValueError("Empty list item in document")
    if re.search(r"[\u4e00-\u9fff]", title + text):
        raise ValueError("Chinese characters in document")
    if RELATIVE_DATE_PATTERN.search(title + text) or RELATIVE_DAY_PATTERN.search(title + text):
        raise ValueError("Document contains relative date")
    if card["person"] in title:
        raise ValueError("Person name in title")
    if card["identity_style"] == "implicit" and (card["person"] in title + text or RELATIONSHIP_MENTION_PATTERN.search(title + text)):
        raise ValueError("Identity shortcut in implicit event")
    text, count = normalize_event_dates(text, card["date"], card["document_date_styles"][slot])
    title, _ = normalize_event_dates(title, card["date"], card["document_date_styles"][slot])
    if not count:
        raise ValueError("Document is missing its absolute event date")
    if not any(cue in text for cue in card["situation_cues"]) or not any(cue in text for cue in card["purpose_cues"]):
        raise ValueError("Document is missing concrete shared event context")
    if not any(term in text for term in CENTRAL_EVENTS[card["category"]][1]):
        raise ValueError("Document is missing its central event action")
    if card["document_styles"][slot] == "체크리스트" and not re.search(r"(?:^|\n)\s*(?:[-*□☐✓✔]|[1-9][.)])", text):
        raise ValueError("Checklist document is missing list markers")
    if answer:
        if card["fact_value"] not in text:
            raise ValueError("Configured answer document does not contain its fact")
        topic = next(index for index, item in enumerate(TOPICS) if item[0] == card["category"])
        other_content = (title + "\n" + text).replace(card["fact_value"], "")
        candidates = [value for fact in FACTS[topic] if fact[0] != card["fact_value"] for value in fact]
        # Compact comparison shadows only; retain title/body boundaries and exclude the assigned fact.
        other_parts = ["".join(part.split()).replace("".join(card["fact_value"].split()), "") for part in (title, text)]
        if any(value in other_content for value in candidates) or any(_matches_known_answer(part, candidates) for part in other_parts):
            raise ValueError("Another concrete answer candidate was added to the answer document")
        if card["identity_style"] == "named" and card["person"] not in text:
            raise ValueError("Named answer document is missing its person")
    else:
        if card['category'] == '학교 행사' and re.search(r'행사\s*장소인', title + text):
            raise ValueError('A concrete answer relation was asserted in a blind distractor')
        topic = next(index for index, item in enumerate(TOPICS) if item[0] == card["category"])
        forbidden = [value for fact in FACTS[topic] for value in fact]
        if any(cue in title + text for cue in forbidden) or any(_matches_known_answer(part, forbidden) for part in (title, text)):
            raise ValueError("A concrete answer was guessed or leaked into a blind distractor")
    return {"title": title, "text": text}


def parse_stage1(response: str, card: dict) -> dict:
    result = _parse_json(response)
    document = _validate_document(result.get("document", {}), card, card["answer_doc_index"], True)
    evidence = result.get("evidence")
    if not isinstance(evidence, str):
        raise ValueError("Missing answer evidence")
    evidence, _ = normalize_event_dates(evidence.strip(), card["date"], card["document_date_styles"][card["answer_doc_index"]])
    if evidence not in document["text"] or card["fact_value"] not in evidence:
        raise ValueError("Answer evidence is not grounded in the answer document")
    if card["category"] == "소아과 예약":
        fact_sentences = [sentence for sentence in re.split(r"[.!?\n]", document["text"]) if card["fact_value"] in sentence]
        if not any(term in evidence for term in ("진료", "진찰")) or any(re.search(r"(?:발표\s*)?연습\s*장소", sentence) for sentence in fact_sentences):
            raise ValueError("Clinic fact relation is not a care appointment")
    if card["category"] == "어린이집 준비" and (not any(term in evidence for term in NOTICE_CUES) or not any(term in evidence for term in ("어린이집", "선생님", "교사"))):
        raise ValueError("Daycare fact relation is not a notified preparation item")
    if card["category"] == "학교 행사" and not any(term in evidence for term in ("행사", "열리", "개최", "진행", "장소")):
        raise ValueError("School venue fact relation is not an event location")
    return {"document": document, "evidence": evidence}


def parse_stage2(response: str, card: dict) -> dict:
    result = _parse_json(response)
    items = result.get("documents")
    expected = set(range(4)) - {card["answer_doc_index"]}
    if not isinstance(items, list) or len(items) != 3 or {item.get("slot") for item in items} != expected:
        raise ValueError("Blind stage must return exactly its three assigned slots")
    documents = [{"slot": item["slot"], **_validate_document(item, card, item["slot"], False)} for item in items]
    return {"documents": documents, "query": _validate_query(result.get("query"), card)}


def _validate_query(query, card: dict) -> str:
    if not isinstance(query, str) or not 8 <= len(query.strip()) <= 180:
        raise ValueError("Invalid Korean query")
    if re.search(r"[\u4e00-\u9fff]", query):
        raise ValueError("Chinese characters in question")
    query = query.strip()
    if card["category"] == "소아과 예약" and CLINIC_ANIMAL_PATTERN.search(query):
        raise ValueError("Clinic fact relation changed to an animal patient or veterinarian")
    if INTERNAL_MARKER_PATTERN.search(query):
        raise ValueError("Internal schema marker in question")
    if INSTRUCTION_ECHO_PATTERN.search(query):
        raise ValueError("Generation instruction echo in question")
    if query.count("?") + query.count("？") != 1 or not query.endswith(("?", "？")) or "\n" in query or "\r" in query:
        raise ValueError("Question contains multiple questions or lines")
    if re.search(r"(?:묻는다|확인한다|궁금하다)(?=\s*(?:[.!?？]|$))", query):
        raise ValueError("Question is a declarative description rather than a direct question")
    topic = next(index for index, item in enumerate(TOPICS) if item[0] == card["category"])
    if _matches_known_answer(query, [cue for fact in FACTS[topic] for cue in fact]):
        raise ValueError("Answer cue leaked into question")
    if RELATIVE_DATE_PATTERN.search(query) or RELATIVE_DAY_PATTERN.search(query):
        raise ValueError("Relative date in question")
    if card["query_date_style"] == "none":
        if DATE_PATTERN.search(query):
            raise ValueError("Date found in a no-date question")
    else:
        query, count = normalize_event_dates(query, card["date"], card["query_date_style"])
        if not count:
            raise ValueError("Question is missing its assigned date")
    if card["identity_style"] == "named" and card["person"] not in query:
        raise ValueError("Named question is missing its person")
    if card["identity_style"] == "implicit" and (card["person"] in query or RELATIONSHIP_MENTION_PATTERN.search(query)):
        raise ValueError("Identity shortcut in implicit question")
    if not any(cue in query for cue in card["situation_cues"]) or not any(cue in query for cue in card["purpose_cues"]):
        raise ValueError("Question is missing disambiguating event context")
    if not any(term in query for term in CENTRAL_EVENTS[card["category"]][1]):
        raise ValueError("Question is missing its central event action")
    validate_query_intent(query, card["category"])
    return query


def parse_stage2_component(response: str, card: dict, component: str) -> dict:
    result = _parse_json(response)
    if component == "query":
        if set(result) != {"query"}:
            raise ValueError("Blind query component must return only one query")
        return {"query": _validate_query(result["query"], card)}
    slot = int(component[1:])
    item = result.get("document", {})
    if set(result) != {"document"} or item.get("slot") != slot:
        raise ValueError("Blind document component returned a different slot")
    return {"document": {"slot": slot, **_validate_document(item, card, slot, False)}}


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


RETRY_HINTS = {
    "invalid_json": "JSON 객체를 끝까지 완성하고 객체 밖의 설명을 쓰지 않는다.",
    "missing_date": "지정된 날짜 문자열을 title뿐 아니라 각 text 본문에도 넣고 query의 날짜 조건을 따른다.",
    "missing_context": "상황 후보군과 목적 후보군에서 각각 하나 이상의 표현을 골라 각 text와 query에 자연스럽게 넣는다. 모든 메모에 같은 낱말을 반복할 필요는 없다.",
    "missing_name": "named인 경우 지정된 인물 이름을 text 본문과 query에 글자 그대로 넣는다.",
    "evidence_mismatch": "evidence는 fact_value 문자열이 들어 있는 document.text의 문장 또는 목록 한 항목을 고치지 말고 그대로 복사한다.",
    "name_in_title": "identity_style을 유지하며 title에는 인물 이름을 쓰지 않는다. named이면 이름은 text와 query에만 넣고 title은 상황이나 할 일을 짧게 적는다.",
    "wrong_event_relation": "중심 사건의 행동과 사실 관계를 유지한다. 정답 문장에는 병원에서 진료를 받거나 해당 장소에서 학교 행사가 열리는 등 지정된 사실 관계를 분명히 적는다.",
    "wrong_fact_origin": "정답 준비물은 어린이집이나 선생님이 안내·공지·지정해 알려 준 물건이다. 정답 문장과 evidence에 그 출처와 안내 사실을 함께 적는다.",
    "invalid_style": "체크리스트와 요약은 내용이 있는 정확히 세 항목의 문자열 배열이다. 빈 표시만 남기지 말고 체크리스트 항목은 '- [ ] '로 시작한다.",
    "wrong_query_intent": "query_intent에 지정된 사실 관계를 묻는다. 다른 준비물·날짜·시간 질문으로 바꾸지 않는다.",
    "answer_leak": "주어지지 않은 사실값을 추측하지 말고 구체적인 장소·물건·자료 이름을 새로 만들지 않는다.",
    "invalid_length": "text를 80~180자 정도로 짧게 쓰고 지정된 필수 내용을 포함한다.",
    "invalid_slots": "요청된 세 slot을 각각 한 번씩만 반환한다.",
    "identity_leak": "implicit인 경우 인물 이름과 가족 관계 호칭을 쓰지 않는다.",
    "invalid_content": "제공된 조건을 다시 확인하고 짧은 한국어 메모와 JSON 객체를 완성한다.",
    "invalid_query_shape": "query는 사용자가 직접 묻는 짧은 의문문 하나다. '어디였지?', '무엇이었지?'처럼 의문형 어미와 물음표로 끝낸다. '묻는다/확인한다/궁금하다'로 질문을 설명하지 않는다.",
    "internal_marker": "메모와 질문은 생활 기록을 읽는 사람을 위해 쓴다. 입력 항목명과 JSON 같은 형식 이름을 본문·제목·질문에 쓰지 않는다.",
    "instruction_echo": "생활 메모의 실제 준비와 행동을 적는다. '배경 활동/이동 사정/중심 사건/개조식 요약'처럼 생성 방법을 설명하는 문구는 쓰지 않는다.",
    "extra_answer": "주어진 fact_value 한 가지만 사실로 기록한다. 같은 fact_label의 다른 장소·준비물·정비 항목·자료는 새로 추가하지 않는다.",
    "wrong_query_purpose": "이동·연락 사정은 별도의 배경 구절로 쓴다. 준비물을 길 안내용·환승용 물건으로 단정하지 않는다. '길을 안내해야 하는데, 활동에서 선생님이 알려 준 준비물은 무엇이었지?'처럼 배경과 질문의 사실 관계를 나눠 쓴다.",
    "wrong_query_focus": "마지막 질문 부분은 '선생님이 안내한 준비물 한 가지가 무엇이었지?'처럼 준비물 자체를 묻는다. 마을 모형·공연·활동 내용이 무엇인지 묻는 질문으로 바꾸지 않는다.",
    "interrupted": "이전 요청이 중단되었다. 제공된 입력만 사용해 새 결과를 완성한다.",
    "chinese_characters": "중국어·한자 없이 자연스러운 한국어로 쓴다. 지정된 날짜·이름·사실 관계와 JSON 형식은 유지한다.",
    "relative_date": "오늘·내일·이번 주·다음 날 같은 상대 날짜를 쓰지 않는다. 정확한 사건 날짜가 이미 있어도 상대 날짜를 함께 쓰지 않는다.",
}
DRAFT_FIELD = "previous_response_raw_text"
RESTART_FIELD = "restart_without_feedback"
ACTIVE_RETRY_HINT_VERSION = 6


def _retry_code(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, json.JSONDecodeError) or "JSON" in str(error):
        return "invalid_json"
    message = str(error).lower()
    for fragment, code in (
        ("chinese characters", "chinese_characters"), ("relative date", "relative_date"),
        ("missing its absolute event date", "missing_date"), ("missing its assigned date", "missing_date"),
        ("no-date question", "missing_date"), ("unexpected date", "missing_date"),
        ("event context", "missing_context"), ("missing its person", "missing_name"),
        ("person name in title", "name_in_title"), ("central event", "wrong_event_relation"),
        ("clinic fact relation", "wrong_event_relation"), ("school venue fact relation", "wrong_event_relation"), ("empty list", "invalid_style"),
        ("daycare fact relation", "wrong_fact_origin"),
        ("list document", "invalid_style"), ("multiple questions", "invalid_query_shape"), ("declarative description", "invalid_query_shape"), ("different slot", "invalid_slots"),
        ("evidence", "evidence_mismatch"), ("checklist", "invalid_style"), ("wrong fact relation", "wrong_query_intent"),
        ("another concrete answer candidate", "extra_answer"),
        ("answer", "answer_leak"), ("length", "invalid_length"), ("slots", "invalid_slots"),
        ("identity shortcut", "identity_leak"),
        ("travel constraint into the item's purpose", "wrong_query_purpose"),
        ("daycare question focus", "wrong_query_focus"),
        ("internal schema marker", "internal_marker"),
        ("generation instruction echo", "instruction_echo"),
    ):
        if fragment in message:
            return code
    return "invalid_content"


def _write_cache(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def compose_event(card: dict, first: dict, second: dict, provenance: dict) -> dict:
    documents = [None] * 4
    documents[card["answer_doc_index"]] = first["document"]
    for document in second["documents"]:
        documents[document["slot"]] = {key: document[key] for key in ("title", "text")}
    row = {**card, "documents": documents, "query": second["query"], "evidence": first["evidence"], "generation_stages": provenance, "repairs": []}
    validate_event(row)
    return row


def validate_event(row: dict, provenance: bool = True) -> None:
    split, number = row["event_id"].split("-")
    card = event_card_v3(split, int(number)).as_dict()
    if any(row.get(key) != value for key, value in card.items()):
        raise ValueError(f"v3 card mismatch: {row['event_id']}")
    if len(row.get("documents", [])) != 4:
        raise ValueError("Event needs four documents")
    for slot, document in enumerate(row["documents"]):
        normalized = _validate_document(document, card, slot, slot == card["answer_doc_index"])
        if normalized != {"title": document["title"], "text": document["text"]}:
            raise ValueError("Stored document differs from its normalized assigned date/style")
    first = parse_stage1(json.dumps({"document": row["documents"][card["answer_doc_index"]], "evidence": row["evidence"]}, ensure_ascii=False), card)
    second = parse_stage2(json.dumps({"documents": [{"slot": slot, **document} for slot, document in enumerate(row["documents"]) if slot != card["answer_doc_index"]], "query": row["query"]}, ensure_ascii=False), card)
    if second["query"] != row["query"]:
        raise ValueError("Stored question differs from its assigned date format")
    keys = [" ".join(document["text"].split()) for document in row["documents"]]
    if len(set(keys)) != 4:
        raise ValueError("Duplicate document text in event")
    if provenance:
        stages = row.get("generation_stages", {})
        if DRAFT_FIELD in stages.get("stage2", {}):
            raise ValueError("A blind stage must not contain a previous-response draft")
        if RESTART_FIELD in stages.get("stage2", {}):
            raise ValueError("A blind stage must not contain a feedback restart marker")
        payload = stage2_payload(card)
        if stages.get("stage2", {}).get("input") != payload or stages["stage2"].get("input_sha256") != _json_hash(payload):
            raise ValueError("Blind stage provenance is missing or differs from the whitelist")
        _verify_request_metadata(stages.get("stage1", {}), stage1_payload(card), 1, first)
        components = stages["stage2"].get("components", {})
        expected = {f"d{item['slot']}" for item in payload["documents"]} | {"query"}
        if set(components) != expected:
            raise ValueError("Blind stage component provenance is missing or incomplete")
        for component in expected:
            result = ({"query": second["query"]} if component == "query" else
                      {"document": next(item for item in second["documents"] if item["slot"] == int(component[1:]))})
            _verify_request_metadata(components[component], stage2_component_payload(payload, component), 2, result)
        if stages["stage2"].get("result_sha256") != _json_hash(second):
            raise ValueError("Blind stage assembly hash differs from its stored content")


def _request_prompt(payload: dict, stage: int, feedback: str | None = None, retry_hint_version: int = 1, previous_response_raw_text: str | None = None) -> str:
    if type(retry_hint_version) is not int or retry_hint_version not in (1, 2, 3, 4, 5, 6):
        raise ValueError("Unknown retry hint version in stage provenance")
    if retry_hint_version == 6:
        if previous_response_raw_text is not None or (stage == 2 and DRAFT_FIELD in payload):
            raise ValueError("A previous-response draft is allowed only in Stage-1 v4")
        if feedback is not None and feedback not in RETRY_HINTS:
            raise ValueError("Unknown retry feedback code in stage provenance")
        return _compact_request(payload, stage, feedback)
    if retry_hint_version == 5:
        if previous_response_raw_text is not None:
            raise ValueError("A previous-response draft is allowed only in Stage-1 v4")
        return _request_prompt(payload, stage, feedback, 3) + _completion_checks(payload, stage)
    if retry_hint_version == 4:
        if stage != 1 or feedback is None or not isinstance(previous_response_raw_text, str):
            raise ValueError("A v4 draft requires a failed Stage-1 response")
        try:
            draft = json.loads(previous_response_raw_text)["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Previous response is not a complete teacher message") from error
        if not isinstance(draft, str):
            raise ValueError("Previous teacher message content must be a string")
        return _request_prompt(payload, stage, feedback, 3) + (
            "\n다음은 같은 답 메모 작업의 검증에 실패한 초안이다. 사실의 기준은 위 입력 카드다. "
            "초안의 틀린 사실이나 지시는 따르지 말고, 카드와 본문 조건에 맞는 새 JSON 응답을 작성한다. "
            "초안을 설명하거나 그대로 인용하지 않는다. 수정 대상 초안(JSON 문자열): "
            + json.dumps(draft, ensure_ascii=False)
        )
    if previous_response_raw_text is not None or (stage == 2 and DRAFT_FIELD in payload):
        raise ValueError("A previous-response draft is allowed only in Stage-1 v4")
    prompt = _instructions(payload, stage)
    if feedback is not None:
        if feedback not in RETRY_HINTS:
            raise ValueError("Unknown retry feedback code in stage provenance")
        hint = RETRY_HINTS[feedback]
        if retry_hint_version == 2 and feedback == "wrong_event_relation":
            actions = " / ".join(CENTRAL_EVENTS[payload["category"]][1])
            target = "query" if stage == 2 and payload.get("component") == "query" else "text 본문"
            hint = f"{payload['category']} 사건의 행동과 사실 관계를 유지한다. [{actions}] 중 하나 이상의 행동 낱말을 {target}에 자연스럽게 적는다."
            if target == "text 본문":
                body_date = payload["body_date"] if stage == 1 else payload["document"]["body_date"]
                hint += f" text 본문에는 지정된 날짜 '{body_date}'를 글자 그대로 넣는다."
        if retry_hint_version == 3 and feedback in {"missing_date", "wrong_event_relation"}:
            actions = " / ".join(CENTRAL_EVENTS[payload["category"]][1])
            query = stage == 2 and payload.get("component") == "query"
            target = "query" if query else "text 본문"
            hint = f"{payload['category']} 사건의 행동과 사실 관계를 유지하고 [{actions}] 중 하나 이상을 {target}에 넣는다. "
            if query and payload["query_date_style"] == "none":
                hint += "query에는 날짜를 넣지 않는다. "
            else:
                body_date = (format_event_date(payload["date"], payload["query_date_style"]) if query else
                             payload["body_date"] if stage == 1 else payload["document"]["body_date"])
                hint += f"{target}에 정확한 날짜 '{body_date}'를 글자 그대로 넣는다. "
            hint += "오늘·내일·이번 주·다음 날 같은 상대 날짜는 쓰지 않는다. 정확한 날짜가 이미 있어도 상대 날짜를 함께 쓰면 안 된다."
        prompt += f"\n이전 시도 검증 코드: {feedback}. {hint}"
        if stage == 2 and payload.get('component') == 'query' and feedback == 'missing_context':
            purpose = ' / '.join(payload['required_context']['purpose_any'])
            situation = ' / '.join(payload['required_context']['situation_any'])
            prompt += f"\n첫 배경 절에는 반드시 [{purpose}] 중 하나의 이동·연락 사정을 넣는다. [{situation}] 상황은 그 뒤에 별도 구절로 잇는다. 두 후보군은 서로 대신할 수 없다."
    return prompt


def _completion_checks(payload: dict, stage: int) -> str:
    query = stage == 2 and payload.get("component") == "query"
    target = "query" if query else "text 본문"
    actions = " / ".join(CENTRAL_EVENTS[payload["category"]][1])
    situation = " / ".join(payload["required_context"]["situation_any"])
    purpose = " / ".join(payload["required_context"]["purpose_any"])
    checks = (
        "\n출력 직전 다음 조건을 모두 확인한 새 JSON만 반환한다. 점검 내용은 출력하지 않는다. "
        "제목과 본문과 질문 어디에도 오늘·내일·어제·이번 주·다음 날 같은 상대 날짜를 쓰지 않는다. "
        f"{target}에 [{actions}] 중 하나 이상의 행동 표현과 [{situation}] 중 하나, [{purpose}] 중 하나를 모두 넣는다. "
        "활동 상황과 이동·연락 사정은 별도의 사실로 연결한다. "
    )
    if query and payload["query_date_style"] == "none":
        checks += "query에는 날짜를 넣지 않는다. "
    else:
        body_date = (format_event_date(payload["date"], payload["query_date_style"]) if query else
                     payload["body_date"] if stage == 1 else payload["document"]["body_date"])
        checks += f"{target}의 {'한 곳' if query else '첫 문장 또는 첫 항목'}에 날짜 '{body_date}'를 글자 그대로 넣는다. "
    if payload["identity_style"] == "named":
        checks += f"{target}에 전체 이름 '{payload['person']}'을 글자 그대로 넣고 title에는 이름을 넣지 않는다. "
    else:
        checks += "이름과 가족 관계 호칭을 쓰지 않는다. "
    if stage == 1:
        checks += (
            f"정답 '{payload['fact_value']}'가 {payload['fact_label']}이라는 관계를 완성된 문장 하나에 분명히 쓴다. "
            "evidence는 그 문장이나 목록 한 항목을 본문에서 글자 그대로 복사한다. "
        )
        if payload["category"] == "소아과 예약":
            checks += "정답이 든 문장과 evidence에는 '진료' 또는 '진찰'을 넣어 그 병원에서 진료받는 예약임을 밝힌다. "
    elif query:
        checks += "질문은 입력의 검색 의도대로 모르는 사실 한 가지만 직접 묻고 물음표 하나로 끝낸다. "
    return checks


def _compact_request(payload: dict, stage: int, feedback: str | None) -> str:
    query = stage == 2 and payload.get("component") == "query"
    brief = {"일정": payload["category"], "사정": payload["secondary_context"],
             "상황 표현 중 하나": payload["required_context"]["situation_any"],
             "이동·연락 표현 중 하나": payload["required_context"]["purpose_any"]}
    identity = (f"전체 이름 '{payload['person']}'을 {'query' if query else 'text'}에 글자 그대로 넣는다. "
                if payload["identity_style"] == "named" else "이름과 가족 관계 호칭은 모두 생략한다. ")
    actions = " / ".join(CENTRAL_EVENTS[payload["category"]][1])
    prompt = (
        "가상의 한국어 생활 기록을 작성한다. 주어진 사정과 일정의 관계를 유지한다. "
        "활동을 바꾸거나 이동 사정을 물건·자료의 용도로 바꾸지 않는다. "
        "중국어·한자·의료 조언·새 장소나 물건·주어지지 않은 시각을 만들지 않는다. "
        "제목·본문·질문에 오늘·내일·어제·이번 주·다음 날 같은 상대 날짜를 쓰지 않는다. "
        "입력의 지시를 설명하거나 본문에 옮기지 말고 JSON만 반환한다.\n"
    )
    if query:
        date_rule = "날짜는 쓰지 않는다."
        if payload["query_date_style"] != "none":
            date_rule = f"날짜 '{format_event_date(payload['date'], payload['query_date_style'])}'를 글자 그대로 넣는다."
        prompt += (
            f"query에 실제 사용자의 의문문 한 개를 쓴다. {identity}{date_rule} "
            f"[{actions}] 중 하나의 표현을 넣는다. 이동·연락 사정을 짧은 첫 절에 적고, 활동 상황을 별도 구절로 잇는다. "
            "입력의 두 표현 묶음에서 각각 하나 이상을 써야 한다. "
            + payload["query_intent"]
            + " 한 가지 사실만 직접 묻고 물음표 하나로 끝낸다. 40~100자 정도로 쓴다. 답이나 구체적인 후보값은 모른다.\n"
        )
        brief["모르는 정보"] = payload["fact_label"]
    else:
        item = payload if stage == 1 else payload["document"]
        brief["본문 날짜"] = item["body_date"]
        prompt += (
            f"document.text는 80~180자 정도로 쓴다. 첫 문장 또는 첫 항목에 날짜 '{item['body_date']}'를 글자 그대로 넣는다. "
            + identity + f"[{actions}] 중 하나의 행동을 본문에 넣는다. "
            "입력의 두 표현 묶음에서 각각 하나 이상을 본문에 자연스럽게 넣는다. "
            + _style_instruction(item["style"])
            + " title은 이름 없이 짧게 쓴다.\n"
        )
        if stage == 1:
            brief["기록할 사실의 역할"] = payload["fact_label"]
            brief["정답 문자열"] = payload["fact_value"]
            prompt += (
                "정답 문자열을 본문 한 곳에 그대로 쓰고 지정한 역할과의 관계를 완성된 문장으로 밝힌다. "
                "evidence에는 그 정답 문장 또는 목록 한 항목을 본문에서 글자 그대로 복사한다. "
            )
            if payload["category"] == "소아과 예약":
                prompt += "정답이 든 문장과 evidence에는 '진료' 또는 '진찰'을 넣어 그 병원에서 진료받을 예약임을 밝힌다. "
            elif payload["category"] == "어린이집 준비":
                prompt += "정답 문장과 evidence에 어린이집이나 선생님이 그 준비물을 안내했다는 출처와 사실을 함께 쓴다. "
            elif payload["category"] == "학교 행사":
                prompt += "정답 문장과 evidence에 해당 장소에서 학교 행사가 열린다는 관계를 쓴다. "
        else:
            brief["아직 모르는 정보"] = payload["fact_label"]
            prompt += f"구체적인 값은 모른다. '{payload['fact_label']}'라는 일반명으로만 지칭하고 별도 기록에서 확인할 상태로 둔다. "
    prompt += "\n입력: " + json.dumps(brief, ensure_ascii=False)
    if feedback is not None:
        hint = ("기록 작성법을 설명하지 말고 실제 일정과 준비 내용만 쓴다."
                if feedback == "instruction_echo" else RETRY_HINTS[feedback])
        prompt += f"\n이전 시도 검증 코드: {feedback}. {hint}"
    return prompt


def _draft_source(previous: dict | None, stage: int) -> Path:
    if stage != 1 or previous is None or previous.get("status") != "failed" or previous.get("error_code") not in RETRY_HINTS:
        raise ValueError("A v4 draft must follow the same Stage-1 job's failed attempt")
    source_path = previous.get("response_path")
    if not isinstance(source_path, str) or Path(source_path).is_absolute() or ".." in Path(source_path).parts or Path(source_path).name != f"stage1-{previous.get('attempt')}.json":
        raise ValueError("A v4 draft source must be its preceding Stage-1 response file")
    return Path(source_path)


def _verified_draft(record: dict, previous: dict | None, stage: int) -> str | None:
    if record.get("retry_hint_version", 1) != 4:
        if DRAFT_FIELD in record:
            raise ValueError("A previous-response draft is allowed only in Stage-1 v4")
        return None
    _draft_source(previous, stage)
    raw = record.get(DRAFT_FIELD)
    if not isinstance(raw, str) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != previous.get("response_sha256"):
        raise ValueError("Previous response raw UTF-8 hash differs from the preceding failed attempt")
    return raw


def _attempt_feedback(record: dict, previous: dict | None, stage: int, index: int) -> str | None:
    if RESTART_FIELD not in record:
        return previous.get("error_code") if previous else None
    attempt = record.get("attempt")
    if (record[RESTART_FIELD] is not True or stage != 1 or record.get("retry_hint_version", 1) != 3
            or type(attempt) is not int or attempt < 3 or attempt % 2 != 1 or index + 1 != attempt
            or previous is None or previous.get("status") != "failed" or previous.get("error_code") not in RETRY_HINTS
            or record.get("feedback_code") is not None or DRAFT_FIELD in record):
        raise ValueError("Invalid feedback restart marker in stage provenance")
    return None


def _verify_attempt_prompts(records: list[dict], payload: dict, stage: int) -> None:
    for index, record in enumerate(records):
        previous = records[index - 1] if index else None
        feedback = _attempt_feedback(record, previous, stage, index)
        draft = _verified_draft(record, previous, stage)
        prompt = _request_prompt(payload, stage, feedback, record.get("retry_hint_version", 1), draft)
        if record.get("feedback_code") != feedback or record.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError("Attempt prompt hash differs from its fresh request")


def _verify_request_metadata(metadata: dict, payload: dict, stage: int, result: dict) -> None:
    if metadata.get("input") != payload or metadata.get("input_sha256") != _json_hash(payload):
        raise ValueError("Request input provenance differs from its whitelist")
    codes = metadata.get("retry_error_codes", [])
    if any(code not in RETRY_HINTS for code in codes):
        raise ValueError("Unknown retry feedback code in stage provenance")
    schema_hash = _json_hash(stage_schema(payload, stage))
    records = metadata.get("attempt_log", [])
    if metadata.get("schema_sha256") != schema_hash or any(item.get("schema_sha256") != schema_hash for item in records):
        raise ValueError("Stage schema hash differs from the blind generation schema")
    if RESTART_FIELD in metadata:
        _attempt_feedback(metadata, records[-2] if len(records) >= 2 else None, stage, len(records) - 1)
    if records and (RESTART_FIELD in metadata) != (RESTART_FIELD in records[-1]):
        raise ValueError("Completed feedback restart marker differs from its final attempt")
    draft = _verified_draft(metadata, records[-2] if len(records) >= 2 else None, stage)
    if metadata.get("retry_hint_version", 1) == 4 and metadata.get(DRAFT_FIELD) != records[-1].get(DRAFT_FIELD):
        raise ValueError("Completed draft source differs from its final attempt")
    prompt = _request_prompt(payload, stage, metadata.get("feedback_code"), metadata.get("retry_hint_version", 1), draft)
    if metadata.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("Stage prompt hash differs from the safe retry protocol")
    _verify_attempt_prompts(records, payload, stage)
    if records and (records[-1].get("status") != "completed" or records[-1].get("feedback_code") != metadata.get("feedback_code") or records[-1].get("retry_hint_version", 1) != metadata.get("retry_hint_version", 1) or [item["error_code"] for item in records if "error_code" in item] != codes):
        raise ValueError("Completed request differs from its attempt history")
    if metadata.get("result_sha256") != _json_hash(result):
        raise ValueError("Generated result hash differs from its stored content")


def _request_options(card: dict, stage: int, name: str, attempt: int, seed_offset: int) -> dict:
    limit = 1000 if stage == 1 else 350 if name == "stage2-query" else 700
    seed_key = 1 if stage == 1 else name
    options = {"temperature": 0.45 + attempt * 0.025, "num_predict": limit, "repeat_penalty": 1.1, "presence_penalty": 0,
               "seed": _seed(card["event_id"], seed_key, attempt, seed_offset) % (2 ** 31)}
    if name == 'stage2-query':
        options.update(temperature=round(0.675 + attempt * 0.025, 3), repeat_penalty=1.0, top_p=0.8, top_k=20, min_p=0.0)
    return options


def generate_event(card: dict, teacher: str, revision: str, stage_dir: Path, budget_path: Path, max_hours: float, seed_offset: int = 0) -> dict:
    from budget import WallBudget
    from generate import ollama_request
    stage_dir.mkdir(parents=True, exist_ok=True)
    cache_name = card['event_id'] + (f'.regeneration-{seed_offset}' if seed_offset else '')
    cache_path = stage_dir / f"{cache_name}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    signature = {"card_sha256": _json_hash(card), "teacher_id": teacher, "teacher_revision": revision, "prompt_version": PROMPT_VERSION}
    if cache and any(cache.get(key) != value for key, value in signature.items()):
        if any(stage in cache for stage in ("stage1", "stage2")):
            raise ValueError(f"Incompatible completed stage cache for {card['event_id']}")
        archive = cache_path.with_name(f"{cache_path.stem}.stale-{time.time_ns()}.json")
        cache_path.replace(archive)
        cache = {}
    cache.update(signature)
    blind = stage2_payload(card)
    components = [f"d{item['slot']}" for item in blind["documents"]] + ["query"]
    jobs = [("stage1", 1, stage1_payload(card), None)] + [
        (f"stage2-{component}", 2, stage2_component_payload(blind, component), component) for component in components
    ]
    for name, stage, payload, component in jobs:
        def parse(response: str) -> dict:
            return parse_stage1(response, card) if stage == 1 else parse_stage2_component(response, card, component)
        schema = stage_schema(payload, stage)
        schema_hash = _json_hash(schema)
        if name in cache:
            parsed = parse(json.dumps(cache[name]["result"], ensure_ascii=False))
            _verify_request_metadata(cache[name], payload, stage, parsed)
            continue
        records = cache.setdefault("attempts", {}).setdefault(name, [])
        if any(record.get("schema_sha256") != schema_hash for record in records):
            raise ValueError(f"Cached {name} attempt schema hash differs from its generation schema")
        _verify_attempt_prompts(records, payload, stage)
        if len(records) >= 8:
            raise RuntimeError(f"Unable to generate {card['event_id']} {name}; eight attempts are already recorded")
        for attempt in range(len(records) + 1, 9):
            feedback = records[-1].get("error_code") if records else None
            version, retry_fields = ACTIVE_RETRY_HINT_VERSION, {}
            if version == 3 and stage == 1 and attempt >= 3 and attempt % 2 == 1 and records and records[-1].get("status") == "failed" and records[-1].get("error_code") in RETRY_HINTS:
                feedback = None
                retry_fields[RESTART_FIELD] = True
            elif version == 3 and stage == 1 and attempt % 2 == 0 and records and records[-1].get("status") == "failed" and records[-1].get("response_path") and records[-1].get("response_sha256"):
                previous = records[-1]
                retry_fields[DRAFT_FIELD] = (stage_dir / _draft_source(previous, stage)).read_bytes().decode("utf-8")
                version = 4
                _verified_draft({"retry_hint_version": version, **retry_fields}, previous, stage)
            prompt = _request_prompt(payload, stage, feedback, version, retry_fields.get(DRAFT_FIELD))
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            options = _request_options(card, stage, name, attempt, seed_offset)
            if version == 6:
                options["num_ctx"] = 2048
            record = {"attempt": attempt, "prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "feedback_code": feedback, "retry_hint_version": version, **retry_fields,
                      "options": options, "seed_offset": seed_offset, "status": "started", "started_at": datetime.now(timezone.utc).isoformat()}
            records.append(record)
            _write_cache(cache_path, cache)
            try:
                with WallBudget(budget_path, max_hours):
                    response = ollama_request("/api/chat", {
                        "model": teacher, "messages": [
                            {"role": "system", "content": "한국어 생활 메모 작성자다. 주어진 입력만 사용하고 JSON만 출력한다."},
                            {"role": "user", "content": prompt},
                        ], "format": schema, "think": False, "stream": False, "keep_alive": "10m", "options": options,
                    })
                    response_dir = stage_dir / f"{card['event_id']}.responses"
                    if seed_offset:
                        response_dir /= f"regeneration-{seed_offset}"
                    response_dir.mkdir(parents=True, exist_ok=True)
                    response_path = response_dir / f"{name}-{attempt}.json"
                    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
                    record.update(response_path=str(response_path.relative_to(stage_dir)), response_sha256=sha256(response_path))
                    _write_cache(cache_path, cache)
                    parsed = parse(response["message"]["content"])
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                record.update(status="failed", error_code=_retry_code(error))
                _write_cache(cache_path, cache)
                print(f"{card['event_id']} {name} attempt {attempt}: {error}", flush=True)
                if attempt == 8:
                    raise RuntimeError(f"Unable to generate {card['event_id']} {name}") from error
            except BaseException as error:
                record.update(status="interrupted_or_failed", error_code=_retry_code(error))
                _write_cache(cache_path, cache)
                raise
            else:
                record.update(status="completed")
                cache[name] = {"input": payload, "input_sha256": _json_hash(payload), "result": parsed, "result_sha256": _json_hash(parsed), "attempt": attempt,
                               "prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "feedback_code": feedback, "retry_hint_version": version, **retry_fields,
                               "retry_error_codes": [item["error_code"] for item in records if "error_code" in item],
                               "options": options, "seed_offset": seed_offset, "attempt_log": records, "generated_at": datetime.now(timezone.utc).isoformat()}
                _write_cache(cache_path, cache)
                break
    second = {"documents": [cache[f"stage2-{component}"]["result"]["document"] for component in components if component != "query"],
              "query": cache["stage2-query"]["result"]["query"]}
    component_metadata = {component: {key: value for key, value in cache[f"stage2-{component}"].items() if key != "result"} for component in components}
    assembled = {"input": blind, "input_sha256": _json_hash(blind), "components": component_metadata,
                 "result": second, "result_sha256": _json_hash(second)}
    if "stage2" in cache:
        if any(cache["stage2"].get(key) != value for key, value in assembled.items()):
            raise ValueError("Cached blind assembly differs from its independent components")
    else:
        cache["stage2"] = {**assembled, "generated_at": datetime.now(timezone.utc).isoformat()}
        _write_cache(cache_path, cache)
    provenance = {name: {key: value for key, value in cache[name].items() if key != "result"} for name in ("stage1", "stage2")}
    return compose_event(card, cache["stage1"]["result"], second, provenance)

def generate(args) -> None:
    from generate import teacher_digest
    teacher_revision = teacher_digest(args.teacher)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(path) if path.exists() else []
    for row in existing:
        if row.get("card_revision") != REVISION or row["teacher_id"] != args.teacher or row["teacher_revision"] != teacher_revision or row["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Output contains another data/teacher revision")
        validate_event(row)
    validate_question_uniqueness(existing)
    completed = {row["event_id"] for row in existing}
    selected = [event_card_v3(split, index) for split in ((args.split,) if args.split else SPLIT_EVENTS) for index in range(SPLIT_EVENTS[split])]
    if args.max_events:
        selected = selected[:args.max_events]
    print(f"Teacher: {args.teacher}@{teacher_revision}; revision: {REVISION}", flush=True)
    for number, card in enumerate(selected, 1):
        if card.event_id in completed:
            continue
        row = generate_event(card.as_dict(), args.teacher, teacher_revision, path.parent / "stages", path.parent / "generation_budget.json", args.max_hours, seed_offset=getattr(args, "seed_offset", 0))
        row.update(teacher_id=args.teacher, teacher_revision=teacher_revision, prompt_version=PROMPT_VERSION, generated_at=datetime.now(timezone.utc).isoformat())
        validate_question_uniqueness([*existing, row])
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        existing.append(row)
        print(f"{number}/{len(selected)} {card.event_id}", flush=True)


def curate_row(source: dict) -> dict:
    row = copy.deepcopy(source)
    validate_event(row)
    row["curation_version"] = CURATION_VERSION
    row["curation_repairs"] = []
    return row


def negative_pools(card: ContextCard, candidates: list[ContextCard]) -> dict[str, list[ContextCard]]:
    pools = {"A": [], "B": [], "C": []}
    for other in candidates:
        if other.event_id == card.event_id:
            continue
        if other.family_id == card.family_id:
            pools["A"].append(other)
        elif other.fact_value == card.fact_value:
            pools["C"].append(other)
        elif other.category == card.category:
            pools["B"].append(other)
    for kind in pools:
        pools[kind].sort(key=lambda other: (
            0 if kind == "B" and other.date == card.date else 1,
            0 if kind == "A" and other.category == card.category else 1,
            _seed(card.event_id, kind, other.event_id),
        ))
    return pools


def compile_rows(raw: list[dict], allow_partial: bool = False) -> dict[str, list[dict]]:
    validate_question_uniqueness(raw)
    by_event = {row["event_id"]: row for row in raw}
    if len(by_event) != len(raw):
        raise ValueError("Duplicate v3 event ID")
    all_cards = [card for split in SPLIT_EVENTS for card in _cards_for(split)]
    expected = {card.event_id for card in all_cards}
    if set(by_event) - expected or (not allow_partial and set(by_event) != expected):
        raise ValueError("Incomplete or unknown v3 event set")
    if len({(row["teacher_id"], row["teacher_revision"], row["prompt_version"], row.get("curation_version")) for row in raw}) != 1:
        raise ValueError("v3 rows use different teacher or curation revisions")
    documents, queries, pairs, triplets = [], [], [], []
    for card in all_cards:
        row = by_event.get(card.event_id)
        if row is None:
            continue
        validate_event(row)
        metadata = {key: card.as_dict()[key] for key in (
            "event_id", "family_id", "split", "category", "date", "fact_value", "person", "identity_style",
            "answer_doc_index", "card_revision", "secondary_context", "context_keywords", "situation_cues", "purpose_cues", "answer_cues", "collision_group",
        )}
        for slot, document in enumerate(row["documents"]):
            documents.append({**metadata, "doc_id": f"{card.event_id}-d{slot}", "date_style": card.document_date_styles[slot],
                              "document_style": card.document_styles[slot], "title": document["title"], "text": document["text"]})
        evidence_id = f"{card.event_id}-d{card.answer_doc_index}"
        queries.append({"query_id": f"{card.event_id}-q0", "event_id": card.event_id, "split": card.split,
                        "person": card.person, "identity_style": card.identity_style, "answer_doc_index": card.answer_doc_index,
                        "card_revision": REVISION, "date_style": card.query_date_style, "text": row["query"],
                        "query_source": "blind_context_teacher", "relevant_doc_ids": [evidence_id], "evidence_doc_id": evidence_id,
                        "evidence": row["evidence"], "reviewed": False})
    present = [card for card in all_cards if card.event_id in by_event]
    train = [card for card in present if card.split == "train"]
    for card in train:
        pools = negative_pools(card, train)
        within_slot = next(slot for slot in range(4) if slot != card.answer_doc_index)
        triplets.append({"task": "search", "anchor_id": f"{card.event_id}-q0", "positive_id": f"{card.event_id}-d{card.answer_doc_index}",
                         "negative_id": f"{card.event_id}-d{within_slot}", "negative_type": "within_event"})
        family_candidates = pools["A"]
        if not family_candidates:
            if not allow_partial:
                raise ValueError(f"No same-family search negative for {card.event_id}")
            family_candidates = [other for other in train if other.event_id != card.event_id]
        if family_candidates:
            other = family_candidates[0]
            triplets.append({"task": "search", "anchor_id": f"{card.event_id}-q0", "positive_id": f"{card.event_id}-d{card.answer_doc_index}",
                             "negative_id": f"{other.event_id}-d{other.answer_doc_index}", "negative_type": "same_family_other_event"})
        # All 160 repeated-fact train events contribute one C; the remaining 40 contribute a second B.
        full_train = list(_cards_for("train"))
        c_eligible = any(other.fact_value == card.fact_value and other.family_id != card.family_id for other in full_train)
        kinds = ["A", "A", "B", "C"] if c_eligible else ["A", "A", "B", "B"]
        random.Random(_seed(card.event_id, "negative-slots")).shuffle(kinds)
        if c_eligible:
            c_slot = kinds.index("C")
            kinds[c_slot], kinds[card.answer_doc_index] = kinds[card.answer_doc_index], kinds[c_slot]
        usage = Counter()
        for slot, kind in enumerate(kinds):
            pool = pools[kind]
            if not pool and allow_partial:
                pool = [other for other in train if other.event_id != card.event_id]
                kind = "partial"
            if not pool:
                raise ValueError(f"No {kind} negative for {card.event_id}")
            other = pool[usage[kind] % len(pool)]
            usage[kind] += 1
            negative_slot = other.answer_doc_index if kind == "C" else slot
            triplets.append({"task": "clustering", "anchor_id": f"{card.event_id}-d{slot}",
                             "positive_id": f"{card.event_id}-d{(slot + 1) % 4}", "negative_id": f"{other.event_id}-d{negative_slot}", "negative_type": kind})
    pair_slots = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for split in ("dev", "test"):
        candidates = [card for card in present if card.split == split]
        if len(candidates) < 50:
            continue
        ordered = sorted(candidates, key=lambda card: _seed(split, "pair-event", card.event_id))
        for number, card in enumerate(ordered[:50]):
            left, right = pair_slots[number % len(pair_slots)]
            pairs.append({"pair_id": f"{split}-related-{number:03d}", "split": split, "left_id": f"{card.event_id}-d{left}",
                          "right_id": f"{card.event_id}-d{right}", "related": True, "negative_type": None, "reviewed": False})
        for kind, amount in (("A", 20), ("B", 20), ("C", 10)):
            eligible = [card for card in ordered if negative_pools(card, candidates)[kind]]
            if kind == "B":
                eligible.sort(key=lambda card: (not any(other.date == card.date for other in negative_pools(card, candidates)[kind]), _seed(split, kind, card.event_id)))
            if not eligible:
                raise ValueError(f"No {kind} pair candidates in {split}")
            for number in range(amount):
                card = eligible[number % len(eligible)]
                pool = negative_pools(card, candidates)[kind]
                other = pool[0]
                left_slot = card.answer_doc_index if kind == "C" else number % 4
                right_slot = other.answer_doc_index if kind == "C" else number % 4
                pairs.append({"pair_id": f"{split}-unrelated-{kind}-{number:03d}", "split": split,
                              "left_id": f"{card.event_id}-d{left_slot}", "right_id": f"{other.event_id}-d{right_slot}",
                              "related": False, "negative_type": kind, "reviewed": False})
    return {"documents": documents, "queries": queries, "pairs": pairs, "triplets": triplets}


def validate_dataset(folder: Path, partial: bool = False) -> dict:
    manifest = verified_dataset_manifest(folder)
    documents, queries, pairs, triplets = (read_jsonl(folder / f"{name}.jsonl") for name in ("documents", "queries", "pairs", "triplets"))
    docs = {document["doc_id"]: document for document in documents}
    q_by_id = {query["query_id"]: query for query in queries}
    validate_question_uniqueness(queries)
    if len(docs) != len(documents) or len(q_by_id) != len(queries):
        raise ValueError("Duplicate document or question IDs")
    text_keys = [" ".join(document["text"].split()) for document in documents]
    if len(set(text_keys)) != len(text_keys):
        raise ValueError("Duplicate v3 document text")
    pii = re.compile(r"\b01[016789][- ]?\d{3,4}[- ]?\d{4}\b|\b\d{6}-?[1-4]\d{6}\b|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    if any(pii.search(document["title"] + document["text"]) for document in documents) or any(pii.search(query["text"] + query["evidence"]) for query in queries):
        raise ValueError("PII-shaped content in v3 data")
    grouped = defaultdict(list)
    for document in documents:
        grouped[document["event_id"]].append(document)
    if any(len(items) != 4 for items in grouped.values()):
        raise ValueError("Event has a document count other than four")
    for query in queries:
        split, index = query["event_id"].split("-")
        card = event_card_v3(split, int(index)).as_dict()
        for key in ("event_id", "split", "person", "identity_style", "answer_doc_index", "card_revision"):
            if query.get(key) != card[key]:
                raise ValueError(f"Question metadata differs from its v3 card: {query['query_id']}/{key}")
        if query["query_id"] != f"{card['event_id']}-q0" or query["date_style"] != card["query_date_style"]:
            raise ValueError("Question ID/date style mismatch")
        items = sorted(grouped[query["event_id"]], key=lambda document: document["doc_id"])
        row = {**card, "documents": [{"title": item["title"], "text": item["text"]} for item in items], "query": query["text"], "evidence": query["evidence"]}
        validate_event(row, provenance=False)
        evidence_id = f"{card['event_id']}-d{card['answer_doc_index']}"
        if query["evidence_doc_id"] != evidence_id or query["relevant_doc_ids"] != [evidence_id]:
            raise ValueError("Evidence reference differs from the assigned answer slot")
        for slot, item in enumerate(items):
            if item["doc_id"] != f"{card['event_id']}-d{slot}":
                raise ValueError("Document ID/slot mismatch")
            for key in ("family_id", "category", "date", "fact_value", "person", "identity_style", "answer_doc_index", "card_revision", "secondary_context", "context_keywords", "situation_cues", "purpose_cues", "answer_cues", "collision_group"):
                if item.get(key) != card[key]:
                    raise ValueError(f"Document metadata differs from its v3 card: {item['doc_id']}/{key}")
            if item["split"] != card["split"] or item["date_style"] != card["document_date_styles"][slot] or item["document_style"] != card["document_styles"][slot]:
                raise ValueError("Document split/style mismatch")
    cluster_counts, search_counts = Counter(), Counter()
    for triple in triplets:
        anchor = q_by_id[triple["anchor_id"]] if triple["task"] == "search" else docs[triple["anchor_id"]]
        positive, negative = docs[triple["positive_id"]], docs[triple["negative_id"]]
        if any(item["split"] != "train" for item in (anchor, positive, negative)):
            raise ValueError("Triplet crosses the training split")
        if triple["task"] == "search":
            if positive["doc_id"] != anchor["evidence_doc_id"] or negative["doc_id"] in anchor["relevant_doc_ids"]:
                raise ValueError("Invalid search triplet evidence")
            kind = "within_event" if negative["event_id"] == anchor["event_id"] else "same_family_other_event"
            search_counts[kind] += 1
            if kind != "within_event" and positive["family_id"] != negative["family_id"]:
                raise ValueError("Cross-event search negative must come from the same family")
        else:
            if positive["event_id"] != anchor["event_id"] or negative["event_id"] == anchor["event_id"]:
                raise ValueError("Invalid cluster positive/negative event relation")
            kind = triple.get("negative_type")
            cluster_counts[kind] += 1
            valid = {
                "A": positive["family_id"] == negative["family_id"],
                "B": positive["family_id"] != negative["family_id"] and positive["category"] == negative["category"] and positive["fact_value"] != negative["fact_value"],
                "C": positive["family_id"] != negative["family_id"] and positive["fact_value"] == negative["fact_value"],
                "partial": partial,
            }.get(kind, False)
            if not valid:
                raise ValueError(f"Invalid {kind} clustering negative")
            if kind == "C" and any(anchor["fact_value"] not in document["title"] + "\n" + document["text"] for document in (anchor, negative)):
                raise ValueError("C clustering anchor and negative must both contain the shared fact in text")
    for pair in pairs:
        left, right = docs[pair["left_id"]], docs[pair["right_id"]]
        if left["split"] != pair["split"] or right["split"] != pair["split"] or pair["related"] != (left["event_id"] == right["event_id"]):
            raise ValueError("Pair split or truth mismatch")
        if not pair["related"]:
            pool = negative_pools(event_card_v3(left["split"], int(left["event_id"].split("-")[1])), list(_cards_for(left["split"])))
            if right["event_id"] not in {candidate.event_id for candidate in pool[pair["negative_type"]]}:
                raise ValueError("Pair negative taxonomy mismatch")
            if pair["negative_type"] == "C" and any(left["fact_value"] not in document["title"] + "\n" + document["text"] for document in (left, right)):
                raise ValueError("C sampled pair must contain the shared fact in both texts")
    counts = Counter(document["split"] for document in documents)
    if not partial:
        if counts != Counter({split: count * 4 for split, count in SPLIT_EVENTS.items()}) or Counter(query["split"] for query in queries) != Counter(SPLIT_EVENTS):
            raise ValueError("Unexpected v3 split counts")
        if cluster_counts != Counter({"A": 400, "B": 240, "C": 160}) or search_counts != Counter({"within_event": 200, "same_family_other_event": 200}):
            raise ValueError("Unexpected v3 training negative counts")
        for split in ("dev", "test"):
            selected = [pair for pair in pairs if pair["split"] == split]
            if len(selected) != 100 or sum(pair["related"] for pair in selected) != 50 or Counter(pair["negative_type"] for pair in selected if not pair["related"]) != Counter({"A": 20, "B": 20, "C": 10}):
                raise ValueError("Unexpected stratified v3 pair counts")
            if not any(not pair["related"] and pair["negative_type"] == "B" and docs[pair["left_id"]]["date"] == docs[pair["right_id"]]["date"] for pair in selected):
                raise ValueError("Pair sample is missing same-date/topic negatives")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2).fit_transform(text_keys)
        similarity = (vectors @ vectors.T).tocoo()
        for left, right, score in zip(similarity.row, similarity.col, similarity.data):
            if left < right and documents[left]["split"] != documents[right]["split"] and score >= 0.98:
                raise ValueError(f"Near-duplicate across splits: {documents[left]['doc_id']}/{documents[right]['doc_id']}")
    if sha256(folder / "review.csv") != manifest["review_file_sha256"]:
        raise ValueError("Review sheet hash mismatch")
    return {"documents": dict(counts), "queries": len(queries), "pairs": len(pairs), "triplets": len(triplets), "clustering_negative_types": dict(cluster_counts), "search_negative_types": dict(search_counts)}


def repair_duplicates(args) -> None:
    from generate import teacher_digest
    from repair_duplicates import duplicate_events
    path = Path(args.input)
    rows = read_jsonl(path)
    teacher = rows[0]["teacher_id"]
    revision = teacher_digest(teacher)
    if any(row.get("card_revision") != REVISION or row["teacher_id"] != teacher or row["teacher_revision"] != revision or row["prompt_version"] != PROMPT_VERSION for row in rows):
        raise ValueError("v3 duplicate repair requires the original teacher/card revision")
    targets = duplicate_events(rows)
    for event_id in targets:
        if event_id.startswith("test-") and not args.allow_test_repair:
            raise ValueError("Test repair requires a new unreviewed revision")
        index = next(index for index, row in enumerate(rows) if row["event_id"] == event_id)
        original = rows[index]
        split, number = event_id.split("-")
        card = event_card_v3(split, int(number)).as_dict()
        history = original.get("regeneration_history", [])
        seed_offset = 1000 * (len(history) + 1)
        while (path.parent / "stages" / f"{event_id}.regeneration-{seed_offset}.json").exists() or (path.parent / "stages" / f"{event_id}.responses" / f"regeneration-{seed_offset}").exists():
            seed_offset += 1000
        new = generate_event(card, teacher, revision, path.parent / "stages", path.parent / "generation_budget.json", args.max_hours, seed_offset=seed_offset)
        history.append({key: original[key] for key in ("generated_at", "documents", "query", "evidence", "generation_stages")})
        rows[index] = {**original, **new, "generated_at": datetime.now(timezone.utc).isoformat(), "regeneration_history": history}
        temporary = path.with_suffix(path.suffix + ".tmp")
        write_jsonl(temporary, rows)
        temporary.replace(path)
        print(f"Regenerated v3 duplicate event {event_id}", flush=True)
    remaining = duplicate_events(rows)
    if remaining:
        raise ValueError(f"v3 cross-event duplicate text remains: {remaining}")
    print(f"Repaired {len(targets)} v3 events", flush=True)
