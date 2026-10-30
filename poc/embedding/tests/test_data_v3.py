import hashlib
import json
from collections import Counter, defaultdict

import pytest

from common import cards, clustering_text, deidentify_person, event_card, format_event_date, matching_doc_text, sha256, write_jsonl
from data_v3 import (
    CENTRAL_EVENTS, DRAFT_FIELD, FACTS, PROMPT_VERSION, PURPOSES, QUERY_INTENTS, RESTART_FIELD, REVISION, RETRY_HINTS, _instructions, _json_hash, _matches_known_answer, _request_prompt, _retry_code, _verify_request_metadata, compose_event, generate_event, negative_pools,
    parse_stage1, parse_stage2, parse_stage2_component, stage1_payload, stage2_payload, stage2_component_payload, stage_schema, validate_event, validate_query_intent, validate_question_uniqueness,
)
from prepare import compile_rows, write_review


def wire_document(card, slot, document):
    return {**document, "text": document["text"].split("\n")} if card.document_styles[slot] in {"체크리스트", "개조식 요약"} else document.copy()


def fixture_metadata(payload, stage, result):
    return {"input": payload, "input_sha256": _json_hash(payload),
            "prompt_sha256": hashlib.sha256(_instructions(payload, stage).encode()).hexdigest(),
            "schema_sha256": _json_hash(stage_schema(payload, stage)), "result_sha256": _json_hash(result),
            "feedback_code": None, "retry_error_codes": []}


def request_metadata_items(row):
    return [row["generation_stages"]["stage1"], *row["generation_stages"]["stage2"]["components"].values()]


def generation_responses(card, row):
    return [{"document": wire_document(card, card.answer_doc_index, row["documents"][card.answer_doc_index]), "evidence": row["evidence"]},
            *[{"document": {"slot": slot, **wire_document(card, slot, document)}} for slot, document in enumerate(row["documents"]) if slot != card.answer_doc_index],
            {"query": row["query"]}]


def sample_v3(card):
    value = card.as_dict()
    documents = []
    evidence = f"{card.fact_label}은 {card.fact_value}로 정했다."
    central = CENTRAL_EVENTS[card.category][1][0]
    for slot in range(4):
        situation = card.situation_cues[slot % len(card.situation_cues)]
        purpose = card.purpose_cues[slot % len(card.purpose_cues)]
        person = f"{card.person}은 " if card.identity_style == "named" else ""
        first = f"{format_event_date(card.date, card.document_date_styles[slot])} {person}{card.category}의 {central} 일정이다. {situation} 상황에서 {purpose} 계획을 확인했다."
        second = f"이 메모는 {slot + 1}번째 관점에서 구체적인 준비와 후속 진행을 적어 두었다."
        third = evidence if slot == card.answer_doc_index else "진행을 마친 뒤 확인할 내용을 다시 살펴보기로 했다."
        text = "\n".join("- " + line for line in (first, second, third)) if card.document_styles[slot] in {"체크리스트", "개조식 요약"} else " ".join((first, second, third))
        documents.append({"title": f"생활 일정 메모 {slot}", "text": text})
    query = f"{card.situation_cues[-1]} 상황에서 {card.purpose_cues[-1]} 계획으로 정한 {central} 일정의 " + QUERY_INTENTS[card.category]["example"]
    if card.identity_style == "named":
        query = f"{card.person}의 " + query
    if card.query_date_style != "none":
        query = f"{format_event_date(card.date, card.query_date_style)}에 " + query
    first = parse_stage1(json.dumps({"document": wire_document(card, card.answer_doc_index, documents[card.answer_doc_index]), "evidence": evidence}, ensure_ascii=False), value)
    second = parse_stage2(json.dumps({"documents": [{"slot": slot, **wire_document(card, slot, document)} for slot, document in enumerate(documents) if slot != card.answer_doc_index], "query": query}, ensure_ascii=False), value)
    blind = stage2_payload(value)
    component_results = {f"d{item['slot']}": {"document": item} for item in second["documents"]} | {"query": {"query": second["query"]}}
    provenance = {"stage1": fixture_metadata(stage1_payload(value), 1, first),
                  "stage2": {"input": blind, "input_sha256": _json_hash(blind), "result_sha256": _json_hash(second),
                             "components": {component: fixture_metadata(stage2_component_payload(blind, component), 2, result) for component, result in component_results.items()}}}

    row = compose_event(value, first, second, provenance)
    row.update(teacher_id="qwen3:8b", teacher_revision="a" * 64, prompt_version=PROMPT_VERSION, curation_version="validate-context-v3")
    return row


def test_v3_cards_have_controlled_collisions_and_repeated_family_topics():
    for split, expected in (("train", 200), ("dev", 75), ("test", 250)):
        values = list(cards(split, revision=REVISION))
        assert len(values) == expected
        collisions, families, categories = defaultdict(list), defaultdict(list), defaultdict(list)
        for card in values:
            families[card.family_id].append(card)
            categories[card.category].append(card)
            if card.collision_group:
                collisions[card.collision_group].append(card)
        assert sum(map(len, collisions.values())) == int(expected * 0.6)
        assert all(len(group) == 5 and len({c.family_id for c in group}) == 5 and len({c.date for c in group}) == 1 and len({c.secondary_context for c in group}) == 5 for group in collisions.values())
        for group in families.values():
            repeated = [topic for topic, count in Counter(card.category for card in group).items() if count >= 2]
            assert repeated
            for topic in repeated:
                dates = [card.date for card in group if card.category == topic]
                assert len(dates) == len(set(dates))
        for group in categories.values():
            for field in ("answer_doc_index", "identity_style"):
                counts = Counter(getattr(card, field) for card in group)
                assert max(counts.values()) - min(counts.values()) <= 1
    train = list(cards("train", revision=REVISION))
    assert sum(bool(negative_pools(card, train)["C"]) for card in train) == 160


def test_context_pairs_are_disjoint_with_balanced_situation_and_purpose_marginals():
    by_topic = defaultdict(dict)
    for split in ("train", "dev", "test"):
        for card in cards(split, revision=REVISION):
            by_topic[card.category].setdefault(split, []).append(card)
    for split_groups in by_topic.values():
        all_pairs = []
        for members in split_groups.values():
            pairs = [(card.context_keywords[0], card.context_keywords[2]) for card in members]
            assert len(set(pairs)) == len(pairs)
            all_pairs.extend(pairs)
            for position in (0, 1):
                frequencies = Counter(pair[position] for pair in pairs)
                frequencies_list = list(frequencies.values()) + [0] * (8 - len(frequencies))
                assert max(frequencies_list) - min(frequencies_list) <= 1
        assert len(set(all_pairs)) == len(all_pairs)


def test_v3_blind_payload_is_whitelisted_and_provenance_is_verified():
    card = event_card("train", 0, REVISION)
    value = card.as_dict()
    value["fact_value"] = "SECRET_ANSWER_SENTINEL"
    value["answer_cues"] = ["SECRET_CUE_SENTINEL"]
    value["documents"] = [{"text": "SECRET_ANSWER_SENTINEL"}]
    value["evidence"] = "SECRET_ANSWER_SENTINEL"
    payload = stage2_payload(value)
    assert "SECRET" not in json.dumps(payload)
    assert "fact_value" not in payload and "answer_cues" not in payload and "evidence" not in payload
    row = sample_v3(card)
    row["generation_stages"]["stage2"]["input"]["fact_value"] = card.fact_value
    with pytest.raises(ValueError, match="Blind stage provenance"):
        validate_event(row)


def test_generation_instructions_require_each_literal_date_in_body():
    card = event_card("train", 0, REVISION).as_dict()
    first = stage1_payload(card)
    assert f"text 본문 안에 정확한 날짜 '{first['body_date']}'" in _instructions(first, 1)
    assert "연속된 부분과 정확히 같아야" in _instructions(first, 1)
    blind = stage2_payload(card)
    for item in blind["documents"]:
        payload = stage2_component_payload(blind, f"d{item['slot']}")
        prompt = _instructions(payload, 2)
        assert f"정확한 날짜 '{item['body_date']}'" in prompt
        assert "document 한 개만" in prompt
        assert "query" not in payload and "documents" not in payload
    payload = stage2_component_payload(blind, "query")
    assert "document" not in payload and "documents" not in payload
    assert "의문문 한 개" in _instructions(payload, 2)
    assert "세 개만" in _instructions(stage2_component_payload(blind, "d0"), 2)


def test_compact_prompts_keep_event_constraints_and_blind_answer_boundaries():
    for card in cards(revision=REVISION):
        value = card.as_dict()
        blind = stage2_payload(value)
        payloads = [(1, stage1_payload(value)), *[
            (2, stage2_component_payload(blind, component))
            for component in [*[f"d{item['slot']}" for item in blind['documents']], 'query']]]
        for stage, payload in payloads:
            prompt = _request_prompt(payload, stage, None, 6)
            assert payload['secondary_context'] in prompt
            assert all(cue in prompt for group in payload['required_context'].values() for cue in group)
            assert all(action in prompt for action in CENTRAL_EVENTS[card.category][1])
            assert '배경 활동' not in prompt and '중심 사건' not in prompt
            if card.identity_style == 'named':
                assert card.person in prompt
            if stage == 1:
                assert card.fact_value in prompt and payload['body_date'] in prompt
            else:
                assert card.fact_value not in prompt
                if payload['component'] == 'query' and card.query_date_style == 'none':
                    assert card.date not in prompt
                elif payload['component'] != 'query':
                    assert payload['document']['body_date'] in prompt


def test_generation_schemas_use_only_shape_and_keep_blind_inputs_answer_free():
    from common import TOPICS
    def shape_only(value):
        if isinstance(value, dict):
            assert not {"pattern", "patternProperties", "prefixItems"}.intersection(value)
            for child in value.values():
                shape_only(child)
        elif isinstance(value, list):
            for child in value:
                shape_only(child)
    for card in cards(revision=REVISION):
        shape_only(stage_schema(stage1_payload(card.as_dict()), 1))
        blind = stage2_payload(card.as_dict())
        topic = next(index for index, item in enumerate(TOPICS) if item[0] == card.category)
        for component in [f"d{item['slot']}" for item in blind['documents']] + ['query']:
            payload = stage2_component_payload(blind, component)
            schema = stage_schema(payload, 2)
            encoded = json.dumps({"input": payload, "schema": schema}, ensure_ascii=False)
            assert all(cue not in encoded for fact in FACTS[topic] for cue in fact)
            assert "context_keywords" not in payload
            shape_only(schema)
            assert schema['additionalProperties'] is False
            if component == 'query':
                assert schema['properties']['query'] == {'type': 'string', 'minLength': 8, 'maxLength': 180}
            else:
                document = schema['properties']['document']
                assert document['properties']['slot']['const'] == int(component[1:])
                assert document['additionalProperties'] is False
    row = sample_v3(event_card('train', 0, REVISION))
    # Post-validation accepts genuine aliases without requiring one shared pair.
    validate_event(row)
    assert not all(row['context_keywords'][0] in item['text'] and row['context_keywords'][2] in item['text'] for item in row['documents'])


@pytest.mark.parametrize('ending', ['어디인지 묻는다.', '무엇인지 확인한다.', '무엇인지 궁금하다?', '어디였지', '무엇인지 묻는다. 무엇이었지?'])
def test_live_declarative_or_unfinished_queries_are_rejected(ending):
    card = event_card('train', 0, REVISION).as_dict()
    with pytest.raises(ValueError, match='Question') as error:
        parse_stage2_component(json.dumps({'query': '이 일정에서는 ' + ending}, ensure_ascii=False), card, 'query')
    assert _retry_code(error.value) == 'invalid_query_shape'


def test_school_fact_needs_an_event_location_relation_in_its_evidence():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '학교 행사')
    row = sample_v3(card)
    slot = card.answer_doc_index
    old = row['evidence']
    wrong = f'함께 가는 사람이 unfamiliar한 {card.fact_value} 위치 안내'
    document = {**row['documents'][slot], 'text': row['documents'][slot]['text'].replace(old, wrong)}
    with pytest.raises(ValueError, match='School venue fact relation') as error:
        parse_stage1(json.dumps({'document': wire_document(card, slot, document), 'evidence': wrong}, ensure_ascii=False), card.as_dict())
    assert _retry_code(error.value) == 'wrong_event_relation'


def test_v3_validation_rejects_missing_declared_hashes_before_reading_data(tmp_path):
    from data_v3 import validate_dataset
    (tmp_path / 'manifest.json').write_text(json.dumps({'dataset_revision': REVISION, 'files': {}}))
    with pytest.raises(ValueError, match='missing required file hashes'):
        validate_dataset(tmp_path, partial=True)


@pytest.mark.parametrize('component', ['d0', 'query'])
def test_live_input_field_names_cannot_become_memo_or_query_content(component):
    card = event_card('train', 0, REVISION)
    if component == 'query':
        response = {'query': 'fact_label로 지정된 숙소는 어디였지?'}
    else:
        row = sample_v3(card)
        document = row['documents'][0]
        document = {**document, 'text': document['text'] + ' fact_label로 지정된 숙소를 정리한다.'}
        response = {'document': {'slot': 0, **wire_document(card, 0, document)}}
    with pytest.raises(ValueError, match='Internal schema marker') as error:
        parse_stage2_component(json.dumps(response, ensure_ascii=False), card.as_dict(), component)
    assert _retry_code(error.value) == 'internal_marker'


def test_answer_rejects_additional_maintenance_facts_without_rejecting_its_own_substring():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '차량 정비' and card.fact_value == '타이어 점검')
    row = sample_v3(card)
    slot = card.answer_doc_index
    document = {**row['documents'][slot], 'text': row['documents'][slot]['text'] + ' 브레이크 패드와 배터리 연결 상태도 점검한다.'}
    with pytest.raises(ValueError, match='Another concrete answer candidate') as error:
        parse_stage1(json.dumps({'document': wire_document(card, slot, document), 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    assert _retry_code(error.value) == 'extra_answer'
    handkerchief = next(card for card in cards('train', revision=REVISION) if card.fact_value == '손수건')
    sample_v3(handkerchief)  # '수건' inside the configured answer is not a second item.


@pytest.mark.parametrize("whitespace", [" ", "  ", "\t", "\n", "\u00a0"])
def test_known_answer_matching_ignores_whitespace_only_in_comparison(whitespace):
    text = f"정비 항목 중 엔진{whitespace}오일 상태를 확인한다."
    assert _matches_known_answer(text, ["엔진오일"])
    assert text == f"정비 항목 중 엔진{whitespace}오일 상태를 확인한다."


@pytest.mark.parametrize("role", ["answer", "blind", "query"])
@pytest.mark.parametrize("whitespace", [" ", "\t", "\u00a0"])
def test_spaced_other_answer_is_rejected_in_documents_and_queries(role, whitespace):
    card = event_card("train", 127, REVISION)
    assert card.category == "차량 정비" and card.fact_value == "냉각수 보충"
    row = sample_v3(card)
    candidate = f"엔진{whitespace}오일"
    if role == "query":
        with pytest.raises(ValueError, match="Answer cue leaked"):
            parse_stage2_component(json.dumps({"query": candidate + " " + row["query"]}, ensure_ascii=False), card.as_dict(), "query")
        return
    slot = card.answer_doc_index if role == "answer" else next(index for index in range(4) if index != card.answer_doc_index)
    document = {**row["documents"][slot], "text": row["documents"][slot]["text"] + f" 정비 항목 중 {candidate} 상태를 먼저 확인한다."}
    if role == "answer":
        with pytest.raises(ValueError, match="Another concrete answer candidate"):
            parse_stage1(json.dumps({"document": wire_document(card, slot, document), "evidence": row["evidence"]}, ensure_ascii=False), card.as_dict())
    else:
        with pytest.raises(ValueError, match="guessed or leaked"):
            parse_stage2_component(json.dumps({"document": {"slot": slot, **wire_document(card, slot, document)}}, ensure_ascii=False), card.as_dict(), f"d{slot}")


def test_matching_does_not_join_title_to_body_or_confuse_the_assigned_fact():
    card = event_card("train", 127, REVISION)
    row = sample_v3(card)
    slot = next(index for index in range(4) if index != card.answer_doc_index and card.document_styles[index] not in {"체크리스트", "개조식 요약"})
    document = {"title": "엔 진", "text": "오 일 " + row["documents"][slot]["text"]}
    result = parse_stage2_component(json.dumps({"document": {"slot": slot, **document}}, ensure_ascii=False), card.as_dict(), f"d{slot}")
    assert result["document"]["text"] == document["text"]
    handkerchief = next(card for card in cards("train", revision=REVISION) if card.fact_value == "손수건")
    row = sample_v3(handkerchief)
    slot = handkerchief.answer_doc_index
    document = {**row["documents"][slot], "text": row["documents"][slot]["text"] + " 손 수 건으로 준비했다."}
    result = parse_stage1(json.dumps({"document": wire_document(handkerchief, slot, document), "evidence": row["evidence"]}, ensure_ascii=False), handkerchief.as_dict())
    assert result["document"]["text"] == document["text"] and result["evidence"] == row["evidence"]


def test_spaced_assigned_fact_does_not_replace_exact_gold_or_evidence():
    card = event_card("train", 127, REVISION)
    row = sample_v3(card)
    slot = card.answer_doc_index
    document = {**row["documents"][slot], "text": row["documents"][slot]["text"].replace(card.fact_value, "냉 각 수 보충")}
    with pytest.raises(ValueError, match="does not contain its fact"):
        parse_stage1(json.dumps({"document": wire_document(card, slot, document), "evidence": row["evidence"]}, ensure_ascii=False), card.as_dict())


def test_daycare_route_guidance_does_not_count_as_a_preparation_notice():
    with pytest.raises(ValueError, match='notified item'):
        validate_query_intent('종이로 마을 만들 활동을 앞둔 어린이집에서 길 안내해준 준비물은 무엇이었지?', '어린이집 준비')
    validate_query_intent('동행에게 길 안내를 하며 챙길 어린이집 선생님이 공지한 준비물은 무엇이었지?', '어린이집 준비')
    with pytest.raises(ValueError, match="item's purpose") as error:
        validate_query_intent('선생님이 지정한 종이 마을 활동 준비물 한 가지는 길 안내용 무엇이었지?', '어린이집 준비')
    assert _retry_code(error.value) == 'wrong_query_purpose'
    with pytest.raises(ValueError, match='question focus') as error:
        validate_query_intent('준비물 안내를 받고 동행에게 길을 설명해야 하는데, 어린이집에서 지정한 종이로 만드는 마을 모형은 무엇이었지?', '어린이집 준비')
    assert _retry_code(error.value) == 'wrong_query_focus'


@pytest.mark.parametrize('suffix, message, code', [
    (' 배경 활동과 이동 사정은 유지됩니다.', 'Generation instruction echo', 'instruction_echo'),
    (' 다음 날 이동 계획을 세웠다.', 'relative date', 'relative_date'),
])
def test_live_instruction_echo_and_next_day_drift_are_rejected(suffix, message, code):
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    document = {**row['documents'][0], 'text': row['documents'][0]['text'] + suffix}
    with pytest.raises(ValueError, match=message) as error:
        parse_stage2_component(json.dumps({'document': {'slot': 0, **wire_document(card, 0, document)}}, ensure_ascii=False), card.as_dict(), 'd0')
    assert _retry_code(error.value) == code


def test_live_dev0057_travel_circumstance_is_not_instruction_echo():
    card = event_card('dev', 57, REVISION).as_dict()
    # Exact failed d1 response from dev-0057, regeneration-1000 attempt 2.
    document = {'slot': 1, 'title': '2025-06-28 어린이집 준비 및 이동 계획', 'text': [
        '- 2025-06-28, 어린이집에서 안내받은 준비물을 챙기며 동물 움직임을 흉내 내는 활동에 대비한다.',
        '- 등원 시간에는 대중교통을 이용하므로 환승 시간을 넉넉히 두어 이동 사정을 고려해야 한다.',
        '- 이지은이 챙긴 물품과 동작 흉내를 위한 활동 계획은 어린이집 준비 일정에 맞춰 정리된다.',
    ]}
    parsed = parse_stage2_component(json.dumps({'document': document}, ensure_ascii=False), card, 'd1')
    assert parsed['document']['text'] == '\n'.join(document['text'])
    # No query was generated for this event; reuse its normal phrase in a direct question.
    query = '이지은이 2025-06-28 환승 시간을 넉넉히 두어 이동 사정을 고려해야 하는데, 동물 움직임을 흉내 내는 활동에 어린이집에서 안내받은 준비물은 무엇이었지?'
    assert parse_stage2_component(json.dumps({'query': query}, ensure_ascii=False), card, 'query') == {'query': query}


def test_live_dev0057_writing_instruction_echo_is_still_rejected():
    card = event_card('dev', 57, REVISION).as_dict()
    # Exact meta-writing sentence from regeneration-1000 d1 attempt 5.
    meta = '이 일은 2025-06-28 어린이집에서 안내받은 준비물을 챙기는 사건으로, 배경 활동과 이동 사정은 별도입니다.'
    document = {'slot': 1, 'title': '어린이집 준비 확인', 'text': [
        '- 2025-06-28 이지은이 어린이집에서 안내받은 준비물을 챙긴다.',
        '- 동물 움직임을 흉내 내는 활동을 위해 환승 시간을 넉넉히 둔다.',
        '- ' + meta,
    ]}
    with pytest.raises(ValueError, match='Generation instruction echo') as error:
        parse_stage2_component(json.dumps({'document': document}, ensure_ascii=False), card, 'd1')
    assert _retry_code(error.value) == 'instruction_echo'
    with pytest.raises(ValueError, match='Generation instruction echo'):
        parse_stage2_component(json.dumps({'query': meta + ' 준비물은 무엇이었지?'}, ensure_ascii=False), card, 'query')


@pytest.mark.parametrize('component', ['d0', 'query'])
@pytest.mark.parametrize('suffix, message, code', [
    (' 漢字', 'Chinese characters', 'chinese_characters'),
    (' 오늘 이동한다.', '[Rr]elative date', 'relative_date'),
    (' 다음 날 이동한다.', '[Rr]elative date', 'relative_date'),
])
def test_cjk_and_relative_dates_have_distinct_retry_codes(component, suffix, message, code):
    card = event_card('dev', 57, REVISION)
    row = sample_v3(card)
    if component == 'query':
        response = {'query': suffix.strip() + ' ' + row['query']}
    else:
        document = {**row['documents'][0], 'text': row['documents'][0]['text'] + suffix}
        response = {'document': {'slot': 0, **document}}
    with pytest.raises(ValueError, match=message) as error:
        parse_stage2_component(json.dumps(response, ensure_ascii=False), card.as_dict(), component)
    assert _retry_code(error.value) == code


@pytest.mark.parametrize('component', ['d0', 'query'])
def test_missing_absolute_date_keeps_its_existing_retry_code(component):
    card = event_card('dev', 57, REVISION)
    row = sample_v3(card)
    if component == 'query':
        response = {'query': row['query'].replace(card.date, '')}
    else:
        document = {**row['documents'][0], 'text': row['documents'][0]['text'].replace(card.date, '')}
        response = {'document': {'slot': 0, **document}}
    with pytest.raises(ValueError, match='missing its') as error:
        parse_stage2_component(json.dumps(response, ensure_ascii=False), card.as_dict(), component)
    assert _retry_code(error.value) == 'missing_date'


def test_relative_day_detection_does_not_split_driving_or_morning_words():
    from data_v3 import RELATIVE_DAY_PATTERN
    assert all(not RELATIVE_DAY_PATTERN.search(text) for text in ('운전날', '운전 날짜 확인', '오전날씨 확인'))
    assert all(RELATIVE_DAY_PATTERN.search(text) for text in ('다음 날 이동', '그전날에 확인', '이튿날 만남'))


def test_blind_school_notes_cannot_assign_an_unlisted_venue_or_echo_writing_rules():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '학교 행사')
    row = sample_v3(card)
    slot = next(slot for slot in range(4) if slot != card.answer_doc_index)
    for suffix, message in [(' 학교 행사 장소인 극장에서 소품을 정리한다.', 'answer relation'), (' 그냥 행사 장소로만 언급해야지.', 'instruction echo')]:
        document = {**row['documents'][slot], 'text': row['documents'][slot]['text'] + suffix}
        with pytest.raises(ValueError, match=message):
            parse_stage2_component(json.dumps({'document': {'slot': slot, **wire_document(card, slot, document)}}, ensure_ascii=False), card.as_dict(), f'd{slot}')
    document = {**row['documents'][slot], 'text': row['documents'][slot]['text'] + ' 학교 행사 장소는 아직 확인 중이다.'}
    parse_stage2_component(json.dumps({'document': {'slot': slot, **wire_document(card, slot, document)}}, ensure_ascii=False), card.as_dict(), f'd{slot}')


def test_v3_rejects_changed_stage_and_attempt_schema_provenance():
    row = sample_v3(event_card("train", 0, REVISION))
    for metadata in request_metadata_items(row):
        metadata["attempt_log"] = [{"schema_sha256": metadata["schema_sha256"], "prompt_sha256": metadata["prompt_sha256"], "feedback_code": None, "status": "completed"}]
    validate_event(row)
    for index in range(5):
        for change_attempt in (False, True):
            changed = json.loads(json.dumps(row))
            metadata = request_metadata_items(changed)[index]
            target = metadata["attempt_log"][0] if change_attempt else metadata
            target["schema_sha256"] = "0" * 64
            with pytest.raises(ValueError, match="Stage schema hash differs"):
                validate_event(changed)
    for field, message in (("input_sha256", "whitelist"), ("prompt_sha256", "prompt hash"), ("result_sha256", "result hash")):
        for index in range(5):
            changed = json.loads(json.dumps(row))
            request_metadata_items(changed)[index][field] = "0" * 64
            with pytest.raises(ValueError, match=message):
                validate_event(changed)


def test_v3_stage_cache_resumes_without_reusing_answer_conversation(tmp_path, monkeypatch):
    card = event_card("train", 0, REVISION)
    row = sample_v3(card)
    responses = generation_responses(card, row)
    requests = []
    def fake_request(endpoint, payload):
        requests.append(payload)
        return {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
    monkeypatch.setattr("generate.ollama_request", fake_request)
    generated = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, tmp_path / "stages", tmp_path / "budget.json", 24)
    assert len(requests) == 5
    assert [request["options"]["num_predict"] for request in requests] == [1000, 700, 700, 700, 350]
    blind = stage2_payload(card.as_dict())
    component_names = [f"d{item['slot']}" for item in blind["documents"]] + ["query"]
    payloads = [stage1_payload(card.as_dict()), *[stage2_component_payload(blind, component) for component in component_names]]
    for index, (request, payload, metadata) in enumerate(zip(requests, payloads, request_metadata_items(generated))):
        stage = 1 if index == 0 else 2
        assert len(request["messages"]) == 2
        assert request["format"] == stage_schema(payload, stage)
        assert metadata["schema_sha256"] == _json_hash(request["format"])
        assert metadata["retry_hint_version"] == 6
        assert metadata["options"] == request["options"]
        assert request["options"]["presence_penalty"] == 0
        assert "draft_num_predict" not in request["options"]
        for attempt in metadata["attempt_log"]:
            assert attempt["retry_hint_version"] == 6
            response_path = tmp_path / "stages" / attempt["response_path"]
            assert attempt["schema_sha256"] == metadata["schema_sha256"]
            assert attempt["response_sha256"] == sha256(response_path)
        if index:
            assert card.fact_value not in json.dumps(request, ensure_ascii=False)
            assert row["evidence"] not in json.dumps(request, ensure_ascii=False)
    resumed = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, tmp_path / "stages", tmp_path / "budget.json", 24)
    assert len(requests) == 5 and resumed == generated

    original_response = tmp_path / "stages" / generated["generation_stages"]["stage1"]["attempt_log"][0]["response_path"]
    original_content = original_response.read_text()
    original_cache = (tmp_path / 'stages' / f'{card.event_id}.json').read_bytes()
    requests.clear()
    regenerated = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, tmp_path / "stages", tmp_path / "budget.json", 24, seed_offset=1000)
    assert "regeneration-1000" in regenerated["generation_stages"]["stage1"]["attempt_log"][0]["response_path"]
    assert original_response.read_text() == original_content
    assert (tmp_path / 'stages' / f'{card.event_id}.json').read_bytes() == original_cache
    resumed_regeneration = generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, tmp_path / 'stages', tmp_path / 'budget.json', 24, seed_offset=1000)
    assert len(requests) == 5 and resumed_regeneration == regenerated


def test_empty_old_stage_cache_is_archived_without_discarding_source(tmp_path, monkeypatch):
    card = event_card("train", 0, REVISION).as_dict()
    stage_dir = tmp_path / "stages"
    stage_dir.mkdir()
    old = {"prompt_version": "old-prompt", "card_sha256": "old-card"}
    cache_path = stage_dir / f"{card['event_id']}.json"
    cache_path.write_text(json.dumps(old))
    def stop_before_request(*args, **kwargs):
        raise RuntimeError("request intentionally stopped")
    monkeypatch.setattr("generate.ollama_request", stop_before_request)
    with pytest.raises(RuntimeError, match="intentionally stopped"):
        generate_event(card, "qwen3:8b", "a" * 64, stage_dir, tmp_path / "budget.json", 24)
    archives = list(stage_dir.glob("*.stale-*.json"))
    assert len(archives) == 1 and json.loads(archives[0].read_text()) == old


@pytest.mark.parametrize("arguments, offset", [([], 0), (["--revision", REVISION, "--seed-offset", "2000"], 2000)])
def test_generation_cli_parses_optional_seed_offset_without_model_calls(monkeypatch, arguments, offset):
    import generate
    import sys

    captured = []
    monkeypatch.setattr(sys, "argv", ["generate.py", *arguments])
    monkeypatch.setattr(generate, "generate", captured.append)
    generate.main()
    assert captured[0].seed_offset == offset


@pytest.mark.parametrize("arguments, message", [
    (["--revision", REVISION, "--seed-offset", "-1"], "must be >= 0"),
    (["--seed-offset", "1000"], "supported only for context-hardneg-v3"),
])
def test_generation_cli_rejects_invalid_seed_offsets_before_generation(monkeypatch, capsys, arguments, message):
    import generate
    import sys

    monkeypatch.setattr(sys, "argv", ["generate.py", *arguments])
    monkeypatch.setattr(generate, "generate", lambda *args: pytest.fail("Invalid CLI arguments must not start generation"))
    with pytest.raises(SystemExit) as error:
        generate.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("offset", [0, 2000])
def test_generation_passes_seed_only_to_unfinished_events_and_validates_completed_rows(tmp_path, monkeypatch, offset):
    from argparse import Namespace
    import data_v3

    original = sample_v3(event_card("train", 0, REVISION))
    path = tmp_path / "raw_events.jsonl"
    write_jsonl(path, [original])
    before = path.read_bytes()
    monkeypatch.setattr("generate.teacher_digest", lambda *args: "a" * 64)
    calls = []
    def generated(card, *args, **kwargs):
        calls.append((card["event_id"], kwargs["seed_offset"]))
        return sample_v3(event_card("train", 1, REVISION))
    monkeypatch.setattr(data_v3, "generate_event", generated)
    args = Namespace(teacher="qwen3:8b", output=str(path), split="train", max_events=2, max_hours=24, seed_offset=offset)
    data_v3.generate(args)
    assert calls == [("train-0001", offset)]
    assert path.read_bytes().startswith(before)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["generation_stages"]["stage1"]["prompt_sha256"] = "0" * 64
    write_jsonl(path, rows)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Stage prompt hash"):
        data_v3.generate(args)
    assert calls == [("train-0001", offset)] and path.read_bytes() == before


def test_duplicate_repair_skips_occupied_cache_and_response_namespaces(tmp_path, monkeypatch):
    from argparse import Namespace
    import data_v3
    import repair_duplicates

    row = sample_v3(event_card("train", 0, REVISION))
    row["generated_at"] = "original-generation"
    path = tmp_path / "raw_events.jsonl"
    write_jsonl(path, [row])
    stages = tmp_path / "stages"
    stages.mkdir()
    occupied = [stages / f"{row['event_id']}.regeneration-{offset}.json" for offset in (1000, 3000)]
    response = stages / f"{row['event_id']}.responses/regeneration-2000/old.json"
    response.parent.mkdir(parents=True)
    for file in [*occupied, response]:
        file.write_text("preserve existing source")
    remaining = iter(([row["event_id"]], []))
    monkeypatch.setattr(repair_duplicates, "duplicate_events", lambda *args: next(remaining))
    monkeypatch.setattr("generate.teacher_digest", lambda *args: "a" * 64)
    offsets = []
    def generated(*args, **kwargs):
        offsets.append(kwargs["seed_offset"])
        return sample_v3(event_card("train", 0, REVISION))
    monkeypatch.setattr(data_v3, "generate_event", generated)
    data_v3.repair_duplicates(Namespace(input=str(path), max_hours=24, allow_test_repair=False))
    assert offsets == [4000]
    assert all(file.read_text() == "preserve existing source" for file in [*occupied, response])
    saved = json.loads(path.read_text())
    assert len(saved["regeneration_history"]) == 1
    assert saved["regeneration_history"][0]["documents"] == row["documents"]


def test_v3_rejects_guessed_nonassigned_answer_and_missing_purpose():
    card = event_card("train", 0, REVISION)
    row = sample_v3(card)
    slot = next(i for i in range(4) if i != card.answer_doc_index)
    row["documents"][slot]["text"] += " 파도빛 게스트하우스에 가기로 했다."
    with pytest.raises(ValueError, match="guessed or leaked"):
        validate_event(row)
    row = sample_v3(card)
    for cue in sorted(card.purpose_cues, key=len, reverse=True):
        row["documents"][slot]["text"] = row["documents"][slot]["text"].replace(cue, "일정 확인")
    with pytest.raises(ValueError, match="concrete shared event context"):
        validate_event(row)


def test_v3_rejects_generic_support_term_without_primary_context():
    card = next(card for card in cards("train", revision=REVISION) if card.context_keywords[0] == "해돋이")
    row = sample_v3(card)
    slot = next(index for index in range(4) if index != card.answer_doc_index)
    for cue in sorted(card.situation_cues, key=len, reverse=True):
        row["documents"][slot]["text"] = row["documents"][slot]["text"].replace(cue, card.context_keywords[1])
    assert "여행" in row["documents"][slot]["text"]
    with pytest.raises(ValueError, match="concrete shared event context"):
        validate_event(row)


def test_query_intent_rejects_live_wrong_lodging_question_and_ambiguous_duplicates():
    bad = "철새 관찰 동선을 짧게 잡기 위해 2025년 6월 14일에 무엇을 준비해야 하나요?"
    with pytest.raises(ValueError, match="wrong fact relation or intent"):
        validate_query_intent(bad, "가족 여행")
    validate_query_intent("철새 관찰 동선을 생각해 어느 숙소에서 묵기로 했지?", "가족 여행")
    queries = [
        {"split": "train", "event_id": "train-0000", "person": "김지은", "text": "김지은은 2025-06-14 철새 동선 여행에서 어느 숙소에 묵지?"},
        {"split": "train", "event_id": "train-0001", "person": "이가은", "text": "이가은은 2026년 7월 15일 철새 동선 여행에서 어느 숙소에 묵지?"},
    ]
    with pytest.raises(ValueError, match="Duplicate or ambiguous"):
        validate_question_uniqueness(queries)


def test_safe_retry_feedback_contains_only_enum_codes_and_fresh_blind_requests(tmp_path, monkeypatch):
    card = next(card for card in cards("train", revision=REVISION) if card.identity_style == "named" and card.category == "가족 여행")
    row = sample_v3(card)
    normal = generation_responses(card, row)
    wrong_first = json.loads(json.dumps(normal[0]))
    body = wrong_first["document"]["text"]
    wrong_first["document"]["text"] = [line.replace(card.person, "") for line in body] if isinstance(body, list) else body.replace(card.person, "")
    wrong_query = {"query": row["query"].replace(QUERY_INTENTS[card.category]["example"], "무엇을 준비해야 하지?")}
    responses = [wrong_first, *normal[:-1], wrong_query, normal[-1]]
    requests = []
    def fake_request(endpoint, payload):
        requests.append(payload)
        return {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
    monkeypatch.setattr("generate.ollama_request", fake_request)
    result = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, tmp_path / "stages", tmp_path / "budget.json", 24)
    assert len(requests) == 7
    assert result["generation_stages"]["stage1"]["retry_error_codes"] == ["missing_name"]
    assert result["generation_stages"]["stage2"]["components"]["query"]["retry_error_codes"] == ["wrong_query_intent"]
    failed_attempt = result["generation_stages"]["stage1"]["attempt_log"][0]
    response_path = tmp_path / "stages" / failed_attempt["response_path"]
    assert response_path.name == "stage1-1.json"
    assert json.loads(json.loads(response_path.read_text())["message"]["content"]) == wrong_first
    for metadata in request_metadata_items(result):
        for attempt in metadata["attempt_log"]:
            assert (tmp_path / "stages" / attempt["response_path"]).is_file()
    for request in requests[2:]:
        assert len(request["messages"]) == 2
        assert card.fact_value not in json.dumps(request, ensure_ascii=False)
        assert row["evidence"] not in json.dumps(request, ensure_ascii=False)
    assert _retry_code(ValueError("SECRET unexpected validation value")) in RETRY_HINTS
    assert json.loads((tmp_path / "budget.json").read_text())["elapsed_seconds"] >= 0


def test_contextual_retry_hints_preserve_legacy_prompts_and_blind_inputs():
    for card in cards(revision=REVISION):
        value = card.as_dict()
        first = stage1_payload(value)
        blind = stage2_payload(value)
        payloads = [(first, 1), *[(stage2_component_payload(blind, component), 2)
                    for component in [f"d{item['slot']}" for item in blind["documents"]] + ["query"]]]
        for payload, stage in payloads:
            old = _instructions(payload, stage) + f"\n이전 시도 검증 코드: wrong_event_relation. {RETRY_HINTS['wrong_event_relation']}"
            assert _request_prompt(payload, stage, "wrong_event_relation") == old
            assert _request_prompt(payload, stage, "wrong_event_relation", 1) == old
            prompt = _request_prompt(payload, stage, "wrong_event_relation", 2)
            hint = prompt.split("\n이전 시도 검증 코드:")[-1]
            assert all(term in hint for term in CENTRAL_EVENTS[card.category][1])
            if stage == 1 or payload["component"] != "query":
                body_date = first["body_date"] if stage == 1 else payload["document"]["body_date"]
                assert f"지정된 날짜 '{body_date}'" in hint
            else:
                assert "지정된 날짜" not in hint
                assert _request_prompt(payload, stage, "missing_context", 1) == _request_prompt(payload, stage, "missing_context", 2)
            if stage == 2:
                assert card.fact_value not in prompt
                assert all(cue not in prompt for cue in card.answer_cues)
            for feedback in ("missing_date", "wrong_event_relation"):
                current = _request_prompt(payload, stage, feedback, 3)
                recap = current.split("\n이전 시도 검증 코드:")[-1]
                assert all(term in recap for term in CENTRAL_EVENTS[card.category][1])
                assert "오늘·내일·이번 주·다음 날" in recap
                assert "이미 있어도 상대 날짜를 함께 쓰면 안 된다" in recap
                if stage == 1 or payload["component"] != "query":
                    body_date = first["body_date"] if stage == 1 else payload["document"]["body_date"]
                    assert f"정확한 날짜 '{body_date}'" in recap
                elif payload["query_date_style"] == "none":
                    assert "query에는 날짜를 넣지 않는다" in recap
                    assert "정확한 날짜 '" not in recap
                    assert card.date not in current
                else:
                    assert f"정확한 날짜 '{format_event_date(card.date, card.query_date_style)}'" in recap
                if stage == 2:
                    assert card.fact_value not in current
                    assert all(cue not in current for cue in card.answer_cues)
            if stage == 2 and payload["component"] == "query":
                assert _request_prompt(payload, stage, "missing_context", 2) == _request_prompt(payload, stage, "missing_context", 3)
        if card.category == "회사 일정":
            assert "병원에서 진료" not in _request_prompt(first, 1, "wrong_event_relation", 2)


def test_all_pre_v4_prompt_bytes_match_the_frozen_protocol_fingerprint():
    digest = hashlib.sha256()
    for card in cards(revision=REVISION):
        first, blind = stage1_payload(card.as_dict()), stage2_payload(card.as_dict())
        payloads = [(first, 1)] + [(stage2_component_payload(blind, component), 2)
                    for component in [f"d{item['slot']}" for item in blind["documents"]] + ["query"]]
        for payload, stage in payloads:
            for version in (1, 2, 3):
                for feedback in (None, "missing_date", "wrong_event_relation", "missing_context", "evidence_mismatch"):
                    digest.update(_request_prompt(payload, stage, feedback, version).encode())
                    digest.update(b"\x00")
    # 39,375 prompts measured from r12 plus its pre-v4 direct-question guard.
    assert digest.hexdigest() == "44020cb4be2263bc6bbd424f04dc876a7f613adceae820a7f8a735892518b374"


def test_all_legacy_hint_bytes_and_v1_through_v4_prompts_stay_unchanged():
    legacy = {code: hint for code, hint in RETRY_HINTS.items() if code not in {'chinese_characters', 'relative_date'}}
    assert len(legacy) == 22
    assert hashlib.sha256(json.dumps(legacy, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == '764d33be354319efca932e271727163dd82307a78808bfd99324ebf02f6d5dd5'
    selected = [event_card('dev', 57, REVISION)] + [next(card for card in cards(revision=REVISION) if card.category == category) for category in CENTRAL_EVENTS]
    draft = json.dumps({'message': {'content': '{"document": {}}'}}, ensure_ascii=False)
    digest = hashlib.sha256()
    for card in selected:
        first, blind = stage1_payload(card.as_dict()), stage2_payload(card.as_dict())
        payloads = [(first, 1)] + [(stage2_component_payload(blind, component), 2)
                    for component in [f"d{item['slot']}" for item in blind['documents']] + ['query']]
        for payload, stage in payloads:
            for version in (1, 2, 3):
                for feedback in sorted(legacy):
                    digest.update(_request_prompt(payload, stage, feedback, version).encode())
                    digest.update(b'\x00')
        for feedback in sorted(legacy):
            digest.update(_request_prompt(first, 1, feedback, 4, draft).encode())
            digest.update(b'\x00')
    # 3,872 prompts measured before the dev-0057 parser guard correction.
    assert digest.hexdigest() == 'fab758c50f37179a50756ad45c0e9b153a3dacb3aed495586922ef5ce5f1e9b2'


@pytest.mark.parametrize('code', ['chinese_characters', 'relative_date'])
def test_new_validation_retry_codes_work_in_fresh_v3_and_stage1_v4(code):
    card = event_card('dev', 57, REVISION).as_dict()
    first = stage1_payload(card)
    blind = stage2_component_payload(stage2_payload(card), 'query')
    for payload, stage in [(first, 1), (blind, 2)]:
        assert _request_prompt(payload, stage, code, 3) == _instructions(payload, stage) + f'\n이전 시도 검증 코드: {code}. {RETRY_HINTS[code]}'
    draft = json.dumps({'message': {'content': '{"document": {}}'}}, ensure_ascii=False)
    assert _request_prompt(first, 1, code, 4, draft).startswith(_request_prompt(first, 1, code, 3))
    assert PROMPT_VERSION == 'qwen-event-v3-blind-context-r12'


@pytest.fixture
def stage1_draft_case():
    card = event_card("train", 0, REVISION)
    row = sample_v3(card)
    payload = stage1_payload(card.as_dict())
    result = {"document": row["documents"][card.answer_doc_index], "evidence": row["evidence"]}
    raw = json.dumps({"model": "qwen3:8b", "message": {"content": '이전 "초안"\n새 줄'},
                      "outer_only": "ENVELOPE_ONLY"}, ensure_ascii=False, indent=3).replace("\n", "\r\n") + "\r\n"
    schema_hash = _json_hash(stage_schema(payload, 1))
    previous = {"attempt": 1, "status": "failed", "error_code": "invalid_json", "feedback_code": None,
                "retry_hint_version": 3, "schema_sha256": schema_hash,
                "prompt_sha256": hashlib.sha256(_request_prompt(payload, 1, None, 3).encode()).hexdigest(),
                "response_path": f"{card.event_id}.responses/stage1-1.json",
                "response_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    final = {"attempt": 2, "status": "completed", "feedback_code": "invalid_json", "retry_hint_version": 4,
             DRAFT_FIELD: raw, "schema_sha256": schema_hash,
             "prompt_sha256": hashlib.sha256(_request_prompt(payload, 1, "invalid_json", 4, raw).encode()).hexdigest()}
    metadata = {**fixture_metadata(payload, 1, result), "retry_hint_version": 4, DRAFT_FIELD: raw,
                "feedback_code": "invalid_json", "retry_error_codes": ["invalid_json"],
                "prompt_sha256": final["prompt_sha256"], "attempt_log": [previous, final]}
    return card, row, raw, metadata, result


def test_v4_uses_only_its_own_failed_raw_response_and_keeps_stage2_blind(tmp_path, monkeypatch, stage1_draft_case):
    monkeypatch.setattr('data_v3.ACTIVE_RETRY_HINT_VERSION', 3)
    card, row, raw, metadata, _ = stage1_draft_case
    folder = tmp_path / "stages"
    folder.mkdir()
    previous = metadata["attempt_log"][0]
    response_path = folder / previous["response_path"]
    response_path.parent.mkdir()
    response_path.write_bytes(raw.encode())
    cache_path = folder / f"{card.event_id}.json"
    cache_path.write_text(json.dumps({"card_sha256": _json_hash(card.as_dict()), "teacher_id": "qwen3:8b",
                                     "teacher_revision": "a" * 64, "prompt_version": PROMPT_VERSION,
                                     "attempts": {"stage1": [previous]}}))
    responses = iter(generation_responses(card, row))
    requests = []
    def request(endpoint, payload):
        requests.append(payload)
        return {"message": {"content": json.dumps(next(responses), ensure_ascii=False)}}
    monkeypatch.setattr("generate.ollama_request", request)
    generated = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24)
    stage1 = generated["generation_stages"]["stage1"]
    assert len(requests) == 5 and stage1["retry_hint_version"] == 4
    assert stage1[DRAFT_FIELD] == stage1["attempt_log"][-1][DRAFT_FIELD] == raw
    assert hashlib.sha256(raw.encode()).hexdigest() == stage1["attempt_log"][0]["response_sha256"]
    assert stage1["input"] == stage1_payload(card.as_dict())
    assert json.dumps(json.loads(raw)["message"]["content"], ensure_ascii=False) in requests[0]["messages"][1]["content"]
    assert "ENVELOPE_ONLY" not in requests[0]["messages"][1]["content"]
    assert generated["documents"] == row["documents"] and generated["evidence"] == row["evidence"]
    assert response_path.read_bytes() == raw.encode()
    blind = stage2_payload(card.as_dict())
    components = [f"d{item['slot']}" for item in blind["documents"]] + ["query"]
    for request, component in zip(requests[1:], components):
        encoded = json.dumps(request, ensure_ascii=False)
        assert card.fact_value not in encoded and DRAFT_FIELD not in encoded and "ENVELOPE_ONLY" not in encoded
        assert request["messages"][1]["content"] == _request_prompt(stage2_component_payload(blind, component), 2, None, 3)
    assert all([message["role"] for message in request["messages"]] == ["system", "user"] for request in requests)
    monkeypatch.setattr("generate.ollama_request", lambda *args: pytest.fail("A completed v4 cache must resume without teacher calls"))
    assert generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24) == generated
    validate_event(generated)


@pytest.mark.parametrize("tampering", ["raw", "previous_hash", "previous_status", "previous_job", "parent_path", "no_previous", "final_source", "attempt_source", "legacy_version"])
def test_v4_metadata_rejects_invalid_draft_provenance(stage1_draft_case, tampering):
    _, _, _, original, result = stage1_draft_case
    _verify_request_metadata(original, original["input"], 1, result)
    metadata = json.loads(json.dumps(original))
    previous, final = metadata["attempt_log"]
    if tampering == "raw":
        metadata[DRAFT_FIELD] += " "  # Same parsed JSON, different original UTF-8 bytes.
    elif tampering == "previous_hash":
        previous["response_sha256"] = "0" * 64
    elif tampering == "previous_status":
        previous["status"] = "completed"
    elif tampering == "previous_job":
        previous["response_path"] = "train-0000.responses/stage2-query-1.json"
    elif tampering == "parent_path":
        previous["response_path"] = "../stage1-1.json"
    elif tampering == "no_previous":
        metadata["attempt_log"] = [final]
    elif tampering == "final_source":
        metadata.pop(DRAFT_FIELD)
    elif tampering == "attempt_source":
        final.pop(DRAFT_FIELD)
    else:
        metadata["retry_hint_version"] = 3
    with pytest.raises(ValueError, match="draft|Draft|raw UTF-8|Stage-1"):
        _verify_request_metadata(metadata, metadata["input"], 1, result)


@pytest.mark.parametrize("raw", ["{broken", "{}", '{"message":{"content":null}}'])
def test_v4_rejects_incomplete_teacher_response_envelopes(stage1_draft_case, raw):
    _, _, _, metadata, _ = stage1_draft_case
    with pytest.raises(ValueError, match="teacher message|Teacher message|content must be"):
        _request_prompt(metadata["input"], 1, "invalid_json", 4, raw)


@pytest.mark.parametrize("value", [None, "", "SECRET_RESPONSE"])
def test_blind_stage_rejects_draft_fields_even_when_empty(stage1_draft_case, value):
    card, row, _, _, _ = stage1_draft_case
    payload = stage2_component_payload(stage2_payload(card.as_dict()), "query")
    metadata = fixture_metadata(payload, 2, {"query": row["query"]})
    for in_attempt in (False, True):
        changed = json.loads(json.dumps(metadata))
        target = changed
        if in_attempt:
            changed["attempt_log"] = [{"schema_sha256": changed["schema_sha256"], "prompt_sha256": changed["prompt_sha256"], "feedback_code": None, "status": "completed"}]
            target = changed["attempt_log"][0]
        target[DRAFT_FIELD] = value
        with pytest.raises(ValueError, match="only in Stage-1 v4"):
            _verify_request_metadata(changed, payload, 2, {"query": row["query"]})
    with pytest.raises(ValueError, match="Stage-1"):
        _request_prompt(payload, 2, "missing_context", 4, json.dumps({"message": {"content": "draft"}}))
    row["generation_stages"]["stage2"][DRAFT_FIELD] = value
    with pytest.raises(ValueError, match="blind stage"):
        validate_event(row)


def test_corrupt_previous_response_file_fails_before_a_new_teacher_call(tmp_path, monkeypatch, stage1_draft_case):
    monkeypatch.setattr('data_v3.ACTIVE_RETRY_HINT_VERSION', 3)
    card, _, raw, metadata, _ = stage1_draft_case
    folder = tmp_path / "stages"
    folder.mkdir()
    previous = metadata["attempt_log"][0]
    response = folder / previous["response_path"]
    response.parent.mkdir()
    response.write_bytes((raw + " ").encode())
    path = folder / f"{card.event_id}.json"
    path.write_text(json.dumps({"card_sha256": _json_hash(card.as_dict()), "teacher_id": "qwen3:8b", "teacher_revision": "a" * 64,
                               "prompt_version": PROMPT_VERSION, "attempts": {"stage1": [previous]}}))
    before = path.read_bytes()
    monkeypatch.setattr("generate.ollama_request", lambda *args: pytest.fail("Corrupt draft provenance must fail before a request"))
    with pytest.raises(ValueError, match="raw UTF-8 hash"):
        generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24)
    assert path.read_bytes() == before and not (tmp_path / "budget.json").exists()


@pytest.fixture
def stage1_restart_case(stage1_draft_case):
    card, row, raw, metadata, result = stage1_draft_case
    metadata = json.loads(json.dumps(metadata))
    second = metadata['attempt_log'][-1]
    second.update(status='failed', error_code='instruction_echo')
    third = {'attempt': 3, 'status': 'completed', 'feedback_code': None, 'retry_hint_version': 3,
             RESTART_FIELD: True, 'schema_sha256': metadata['schema_sha256'],
             'prompt_sha256': hashlib.sha256(_request_prompt(metadata['input'], 1, None, 3).encode()).hexdigest()}
    metadata.pop(DRAFT_FIELD)
    metadata.update(third, retry_error_codes=['invalid_json', 'instruction_echo'],
                    attempt_log=[*metadata['attempt_log'], third])
    return card, row, metadata, result


@pytest.mark.parametrize('tampering', ['false', 'one', 'null', 'string', 'even', 'first', 'ordinal', 'float', 'version',
                                     'previous_started', 'previous_interrupted', 'previous_unknown', 'feedback', 'draft',
                                     'missing_final', 'missing_attempt', 'missing_both', 'short_log', 'final_ordinal'])
def test_restart_marker_is_strict_and_cannot_hide_a_changed_retry_chain(stage1_restart_case, tampering):
    _, _, original, result = stage1_restart_case
    _verify_request_metadata(original, original['input'], 1, result)
    metadata = json.loads(json.dumps(original))
    final = metadata['attempt_log'][-1]
    if tampering in {'false', 'one', 'null', 'string'}:
        final[RESTART_FIELD] = {'false': False, 'one': 1, 'null': None, 'string': 'true'}[tampering]
    elif tampering in {'even', 'first', 'ordinal', 'float'}:
        final['attempt'] = {'even': 4, 'first': 1, 'ordinal': 5, 'float': 3.0}[tampering]
    elif tampering == 'version':
        final['retry_hint_version'] = 4
    elif tampering.startswith('previous_'):
        previous = metadata['attempt_log'][-2]
        if tampering == 'previous_unknown':
            previous['error_code'] = 'UNKNOWN'
        else:
            previous['status'] = 'started' if tampering == 'previous_started' else 'interrupted_or_failed'
    elif tampering == 'feedback':
        final['feedback_code'] = 'instruction_echo'
    elif tampering == 'draft':
        final[DRAFT_FIELD] = None
    elif tampering == 'missing_final':
        metadata.pop(RESTART_FIELD)
    elif tampering == 'missing_attempt':
        final.pop(RESTART_FIELD)
    elif tampering == 'missing_both':
        metadata.pop(RESTART_FIELD)
        final.pop(RESTART_FIELD)
    elif tampering == 'short_log':
        metadata['attempt_log'] = [final]
    else:
        metadata['attempt'] = 5
    with pytest.raises(ValueError, match='restart marker|Attempt prompt hash'):
        _verify_request_metadata(metadata, metadata['input'], 1, result)


@pytest.mark.parametrize('value', [True, False, None, 1, 'true'])
def test_blind_stage_rejects_restart_markers_in_every_metadata_layer(value):
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    for metadata in request_metadata_items(row)[1:]:
        payload = metadata['input']
        component = payload['component']
        result = {'query': row['query']} if component == 'query' else {'document': {'slot': payload['document']['slot'], **row['documents'][payload['document']['slot']]}}
        for in_attempt in (False, True):
            changed = json.loads(json.dumps(metadata))
            target = changed
            if in_attempt:
                changed['attempt_log'] = [{'schema_sha256': changed['schema_sha256'], 'prompt_sha256': changed['prompt_sha256'],
                                           'feedback_code': None, 'status': 'completed'}]
                target = changed['attempt_log'][0]
            target[RESTART_FIELD] = value
            with pytest.raises(ValueError, match='restart marker'):
                _verify_request_metadata(changed, payload, 2, result)
    row['generation_stages']['stage2'][RESTART_FIELD] = value
    with pytest.raises(ValueError, match='blind stage'):
        validate_event(row)


@pytest.mark.parametrize('successful_attempt', [3, 8, None])
@pytest.mark.parametrize('policy', [3, 5, 6])
def test_stage1_retry_policies_preserve_history_and_attempt_limits(tmp_path, monkeypatch, successful_attempt, policy):
    from data_v3 import _request_options
    monkeypatch.setattr('data_v3.ACTIVE_RETRY_HINT_VERSION', policy)
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    responses = iter(generation_responses(card, row))
    requests = []
    failures = (successful_attempt or 9) - 1
    def request(endpoint, payload):
        requests.append(payload)
        content = 'FAILED_DRAFT_SENTINEL' if len(requests) <= failures else json.dumps(next(responses), ensure_ascii=False)
        return {'message': {'content': content}}
    monkeypatch.setattr('generate.ollama_request', request)
    folder, budget = tmp_path / 'stages', tmp_path / 'budget.json'
    if successful_attempt is None:
        with pytest.raises(RuntimeError, match='Unable to generate'):
            generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
        records = json.loads((folder / f'{card.event_id}.json').read_text())['attempts']['stage1']
        assert len(requests) == len(records) == 8
        monkeypatch.setattr('generate.ollama_request', lambda *args: pytest.fail('The eight-attempt cap must remain'))
        with pytest.raises(RuntimeError, match='eight attempts are already recorded'):
            generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
    else:
        generated = generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
        stage1 = generated['generation_stages']['stage1']
        records = stage1['attempt_log']
        assert len(requests) == successful_attempt + 4
        assert stage1['retry_error_codes'] == ['invalid_json'] * failures
        assert (RESTART_FIELD in stage1) == (policy == 3 and successful_attempt == 3)
        for metadata, actual in zip(request_metadata_items(generated)[1:], requests[successful_attempt:]):
            assert RESTART_FIELD not in metadata and DRAFT_FIELD not in metadata and metadata['retry_hint_version'] == policy
            assert 'FAILED_DRAFT_SENTINEL' not in json.dumps(actual)
            assert card.fact_value not in json.dumps(actual, ensure_ascii=False)
        monkeypatch.setattr('generate.ollama_request', lambda *args: pytest.fail('Completed new cache must resume without calls'))
        assert generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24) == generated
    for record, actual in zip(records, requests):
        attempt = record['attempt']
        expected_options = _request_options(card.as_dict(), 1, 'stage1', attempt, 0)
        if policy == 6:
            expected_options['num_ctx'] = 2048
        assert record['options'] == actual['options'] == expected_options
        assert [message['role'] for message in actual['messages']] == ['system', 'user']
        if policy in (5, 6):
            feedback = 'invalid_json' if attempt > 1 else None
            assert record['retry_hint_version'] == policy and record['feedback_code'] == feedback
            assert RESTART_FIELD not in record and DRAFT_FIELD not in record
            assert 'FAILED_DRAFT_SENTINEL' not in actual['messages'][1]['content']
            assert actual['messages'][1]['content'] == _request_prompt(stage1_payload(card.as_dict()), 1, feedback, policy)
            continue
        assert record['retry_hint_version'] == (3 if attempt % 2 else 4)
        assert (RESTART_FIELD in record) == (attempt >= 3 and attempt % 2 == 1)
        if attempt % 2:
            assert record['feedback_code'] is None and DRAFT_FIELD not in record
            assert actual['messages'][1]['content'] == _request_prompt(stage1_payload(card.as_dict()), 1, None, 3)
        else:
            assert record['feedback_code'] == 'invalid_json' and 'FAILED_DRAFT_SENTINEL' in actual['messages'][1]['content']
            previous = records[attempt - 2]
            assert hashlib.sha256(record[DRAFT_FIELD].encode()).hexdigest() == previous['response_sha256']


@pytest.mark.parametrize('previous_status', ['started', 'interrupted_or_failed'])
def test_odd_stage1_after_interruption_uses_legacy_feedback_without_a_restart_marker(tmp_path, monkeypatch, stage1_draft_case, previous_status):
    monkeypatch.setattr('data_v3.ACTIVE_RETRY_HINT_VERSION', 3)
    card, row, raw, metadata, _ = stage1_draft_case
    records = json.loads(json.dumps(metadata['attempt_log']))
    records[-1]['status'] = previous_status
    if previous_status == 'interrupted_or_failed':
        records[-1]['error_code'] = 'interrupted'
    folder = tmp_path / 'stages'
    folder.mkdir()
    path = folder / f'{card.event_id}.json'
    path.write_text(json.dumps({'card_sha256': _json_hash(card.as_dict()), 'teacher_id': 'qwen3:8b', 'teacher_revision': 'a' * 64,
                                'prompt_version': PROMPT_VERSION, 'attempts': {'stage1': records}}))
    responses, requests = iter(generation_responses(card, row)), []
    def request(endpoint, payload):
        requests.append(payload)
        return {'message': {'content': json.dumps(next(responses), ensure_ascii=False)}}
    monkeypatch.setattr('generate.ollama_request', request)
    generated = generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, tmp_path / 'budget.json', 24)
    final = generated['generation_stages']['stage1']
    assert final['attempt'] == 3 and RESTART_FIELD not in final and DRAFT_FIELD not in final
    assert final['feedback_code'] == records[-1].get('error_code')
    assert requests[0]['messages'][1]['content'] == _request_prompt(stage1_payload(card.as_dict()), 1, final['feedback_code'], 3)


def test_retry_metadata_verifies_each_version_and_rejects_unknown_or_changed_versions():
    card = next(card for card in cards("train", revision=REVISION) if card.category == "회사 일정")
    payload = stage1_payload(card.as_dict())
    row = sample_v3(card)
    result = {"document": row["documents"][card.answer_doc_index], "evidence": row["evidence"]}
    metadata = fixture_metadata(payload, 1, result)
    _verify_request_metadata(metadata, payload, 1, result)
    for version in (1, 2, 3):
        _verify_request_metadata({**metadata, "retry_hint_version": version}, payload, 1, result)
    records = []
    for index, version in enumerate((1, 1, 2)):
        feedback = "wrong_event_relation" if index else None
        record = {"feedback_code": feedback, "schema_sha256": metadata["schema_sha256"],
                  "prompt_sha256": hashlib.sha256(_request_prompt(payload, 1, feedback, version).encode()).hexdigest(),
                  "status": "completed" if index == 2 else "failed"}
        if index:
            record["retry_hint_version"] = version
        if index < 2:
            record["error_code"] = "wrong_event_relation"
        records.append(record)
    metadata.update(retry_hint_version=2, feedback_code="wrong_event_relation", attempt_log=records,
                    retry_error_codes=["wrong_event_relation"] * 2, prompt_sha256=records[-1]["prompt_sha256"])
    _verify_request_metadata(metadata, payload, 1, result)
    for location in (None, 0, 1, 2):
        changed = json.loads(json.dumps(metadata))
        target = changed if location is None else changed["attempt_log"][location]
        target["retry_hint_version"] = 99
        with pytest.raises(ValueError, match="Unknown retry hint version"):
            _verify_request_metadata(changed, payload, 1, result)
    for location, message in ((None, "Stage prompt hash"), (1, "Attempt prompt hash")):
        changed = json.loads(json.dumps(metadata))
        target = changed if location is None else changed["attempt_log"][location]
        target["retry_hint_version"] = 1 if location is None else 2
        with pytest.raises(ValueError, match=message):
            _verify_request_metadata(changed, payload, 1, result)


@pytest.mark.parametrize("old_version", [None, 2])
def test_legacy_failed_attempts_resume_with_fresh_v3_and_completed_v1_cache_stays_reusable(tmp_path, monkeypatch, old_version):
    monkeypatch.setattr('data_v3.ACTIVE_RETRY_HINT_VERSION', 3)
    card = event_card("train", 33, REVISION)
    row = sample_v3(card)
    first = stage1_payload(card.as_dict())
    folder = tmp_path / "stages"
    folder.mkdir()
    path = folder / f"{card.event_id}.json"
    records = [{"feedback_code": feedback, "schema_sha256": _json_hash(stage_schema(first, 1)),
                "prompt_sha256": hashlib.sha256(_request_prompt(first, 1, feedback, old_version or 1).encode()).hexdigest(),
                "status": "failed", "error_code": "wrong_event_relation"}
               for feedback in (None, "wrong_event_relation")]
    if old_version is not None:
        for record in records:
            record["retry_hint_version"] = old_version
    cache = {"card_sha256": _json_hash(card.as_dict()), "teacher_id": "qwen3:8b", "teacher_revision": "a" * 64,
             "prompt_version": PROMPT_VERSION, "attempts": {"stage1": records}}
    monkeypatch.setattr("generate.ollama_request", lambda *args: pytest.fail("Invalid cached versions must fail before a request"))
    invalid = json.loads(json.dumps(cache))
    invalid["attempts"]["stage1"][0]["retry_hint_version"] = 99
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Unknown retry hint version"):
        generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24)
    assert not (tmp_path / "budget.json").exists()
    path.write_text(json.dumps(cache))
    responses = iter(generation_responses(card, row))
    requests = []
    def fake_request(endpoint, payload):
        requests.append(payload)
        return {"message": {"content": json.dumps(next(responses), ensure_ascii=False)}}
    monkeypatch.setattr("generate.ollama_request", fake_request)
    generated = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24)
    assert len(requests) == 5
    attempts = generated["generation_stages"]["stage1"]["attempt_log"]
    assert attempts[:2] == records
    assert attempts[-1]["retry_hint_version"] == 3
    assert attempts[-1][RESTART_FIELD] is True and attempts[-1]['feedback_code'] is None
    assert requests[0]["messages"][1]["content"] == _request_prompt(first, 1, None, 3)
    for request in requests[1:]:
        assert card.fact_value not in json.dumps(request, ensure_ascii=False)
    cache = json.loads(path.read_text())
    blind = stage2_payload(card.as_dict())
    for name in ["stage1", *[f"stage2-{component}" for component in cache["stage2"]["components"]]]:
        metadata = cache[name]
        payload = first if name == "stage1" else stage2_component_payload(blind, name.removeprefix("stage2-"))
        stage = 1 if name == "stage1" else 2
        for item in [metadata, *metadata["attempt_log"]]:
            if item.pop(RESTART_FIELD, None):
                item['feedback_code'] = 'wrong_event_relation'
            item.pop("retry_hint_version", None)
            item["prompt_sha256"] = hashlib.sha256(_request_prompt(payload, stage, item["feedback_code"], 1).encode()).hexdigest()
    cache["stage2"]["components"] = {component: {key: value for key, value in cache[f"stage2-{component}"].items() if key != "result"}
                                         for component in cache["stage2"]["components"]}
    path.write_text(json.dumps(cache))
    before = path.read_bytes()
    monkeypatch.setattr("generate.ollama_request", lambda *args: pytest.fail("Completed v1 cache must not regenerate"))
    resumed = generate_event(card.as_dict(), "qwen3:8b", "a" * 64, folder, tmp_path / "budget.json", 24)
    validate_event(resumed)
    assert path.read_bytes() == before


def test_generation_interrupt_persists_budget_and_attempt_log(tmp_path, monkeypatch):
    card = event_card("train", 0, REVISION)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr("generate.ollama_request", interrupted)
    with pytest.raises(KeyboardInterrupt):
        generate_event(card.as_dict(), "qwen3:8b", "a" * 64, tmp_path / "stages", tmp_path / "budget.json", 24)
    assert json.loads((tmp_path / "budget.json").read_text())["status"] == "interrupted_or_failed"
    cache = json.loads((tmp_path / "stages" / f"{card.event_id}.json").read_text())
    assert cache["attempts"]["stage1"][0]["error_code"] == "interrupted"


def test_v3_compiles_real_same_fact_negatives_and_stratified_pairs(tmp_path):
    raw = [sample_v3(card) for card in cards(revision=REVISION)]
    compiled = compile_rows(raw)
    assert Counter(triple["task"] for triple in compiled["triplets"]) == {"search": 400, "clustering": 800}
    assert Counter(triple["negative_type"] for triple in compiled["triplets"] if triple["task"] == "clustering") == {"A": 400, "B": 240, "C": 160}
    for split in ("dev", "test"):
        pairs = [pair for pair in compiled["pairs"] if pair["split"] == split]
        assert len(pairs) == 100
        assert Counter(pair["negative_type"] for pair in pairs if not pair["related"]) == {"A": 20, "B": 20, "C": 10}
    docs = {doc["doc_id"]: doc for doc in compiled["documents"]}
    anchors = defaultdict(list)
    for triple in compiled["triplets"]:
        if triple["task"] == "clustering":
            anchors[docs[triple["anchor_id"]]["event_id"]].append(triple["anchor_id"])
        if triple["task"] == "clustering" and triple["negative_type"] == "C":
            anchor, negative = docs[triple["anchor_id"]], docs[triple["negative_id"]]
            assert anchor["fact_value"] == negative["fact_value"]
            assert anchor["family_id"] != negative["family_id"]
            assert all(anchor["fact_value"] in document["title"] + "\n" + document["text"] for document in (anchor, negative))
    assert len(anchors) == 200
    assert all(set(values) == {f"{event_id}-d{slot}" for slot in range(4)} and len(values) == 4 for event_id, values in anchors.items())
    for pair in compiled["pairs"]:
        if pair["negative_type"] == "C":
            left, right = docs[pair["left_id"]], docs[pair["right_id"]]
            assert all(left["fact_value"] in document["title"] + "\n" + document["text"] for document in (left, right))
    for name, rows in compiled.items():
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    write_review(tmp_path / "review.csv", compiled)
    manifest = {"dataset_revision": REVISION, "files": {f"{name}.jsonl": sha256(tmp_path / f"{name}.jsonl") for name in compiled}, "review_file_sha256": sha256(tmp_path / "review.csv")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    from validate import validate_dataset
    assert validate_dataset(tmp_path)["triplets"] == 1200
    # Same metadata and event IDs are insufficient if the actual C texts lack the fact.
    changed = json.loads(json.dumps(compiled["triplets"]))
    triple = next(row for row in changed if row.get("negative_type") == "C")
    negative = docs[triple["negative_id"]]
    triple["negative_id"] = f"{negative['event_id']}-d{(negative['answer_doc_index'] + 1) % 4}"
    write_jsonl(tmp_path / "triplets.jsonl", changed)
    manifest["files"]["triplets.jsonl"] = sha256(tmp_path / "triplets.jsonl")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="C clustering anchor and negative"):
        validate_dataset(tmp_path)
    write_jsonl(tmp_path / "triplets.jsonl", compiled["triplets"])
    manifest["files"]["triplets.jsonl"] = sha256(tmp_path / "triplets.jsonl")
    changed = json.loads(json.dumps(compiled["pairs"]))
    pair = next(row for row in changed if row.get("negative_type") == "C")
    right = docs[pair["right_id"]]
    pair["right_id"] = f"{right['event_id']}-d{(right['answer_doc_index'] + 1) % 4}"
    write_jsonl(tmp_path / "pairs.jsonl", changed)
    manifest["files"]["pairs.jsonl"] = sha256(tmp_path / "pairs.jsonl")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="C sampled pair"):
        validate_dataset(tmp_path)


def test_name_masking_preserves_search_and_ordinary_given_name_words():
    document = {"title": "진료 기록", "text": "김지은은 예약했다. 지은이가 내용을 확인했다. 집을 지은 뒤 다시 본다.", "person": "김지은", "card_revision": REVISION}
    assert "김지은" not in clustering_text(document)
    assert "지은이가" not in clustering_text(document)
    assert "집을 지은 뒤" in clustering_text(document)
    assert "김지은" in matching_doc_text(document)
    assert "김지은" in clustering_text({**document, "card_revision": "shortcut-hardneg-v2-balanced"})
    assert deidentify_person("집을 지은 뒤", "김지은") == "집을 지은 뒤"
    assert "김지은" not in deidentify_person("김지은(동행자), 김지은씨가 확인했다.", "김지은")


def test_r6_wire_lists_are_three_single_lines_and_join_without_rewriting():
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    payload = stage2_payload(card.as_dict())
    items = []
    for target in payload['documents']:
        slot = target['slot']
        component = stage2_component_payload(payload, f'd{slot}')
        document_schema = stage_schema(component, 2)['properties']['document']
        assert document_schema['properties']['slot']['const'] == slot
        assert list(document_schema['properties']) == ['text', 'title', 'slot']
        document = row['documents'][slot].copy()
        if target['style'] in {'체크리스트', '개조식 요약'}:
            date_text = format_event_date(card.date, card.document_date_styles[slot])
            lines = [f'- {date_text} {card.category}의 {card.context_keywords[0]} 상황을 {card.context_keywords[2]} 계획과 함께 확인했다.',
                     '- 두 번째 관점에서 앞으로 살펴볼 행동을 따로 기록했다.',
                     '- 세 번째 관점에서 확인을 마친 뒤 다시 연락하기로 했다.']
            document['text'] = lines
            text_schema = document_schema['properties']['text']
            assert text_schema == {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 100}, 'minItems': 3, 'maxItems': 3}
        else:
            assert document_schema['properties']['text']['type'] == 'string'
            assert document_schema['properties']['text']['maxLength'] == 250
        items.append({'slot': slot, **document})
    parsed = parse_stage2(json.dumps({'documents': items, 'query': row['query']}, ensure_ascii=False), card.as_dict())
    for original, normalized in zip(items, parsed['documents']):
        expected = '\n'.join(original['text']) if isinstance(original['text'], list) else original['text']
        assert normalized['text'] == expected
    list_item = next(item for item in items if isinstance(item['text'], list))
    list_item['text'][-1] = '- [ ]'
    with pytest.raises(ValueError, match='Empty list'):
        parse_stage2(json.dumps({'documents': items, 'query': row['query']}, ensure_ascii=False), card.as_dict())


def test_r8_unicode_spaces_and_quotes_survive_while_postconditions_stay_strict():
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    document = {**row['documents'][card.answer_doc_index], 'title': '온천 "공유" 기록'}
    document['text'] += ' "원문 그대로"를 확인했다. 🗓️'
    parsed = parse_stage1(json.dumps({'document': document, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    assert parsed['document'] == document
    for title in ('', '가' * 101):
        with pytest.raises(ValueError, match='Missing document|Title length'):
            parse_stage1(json.dumps({'document': {**document, 'title': title}, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    with pytest.raises(ValueError, match='Person name in title'):
        parse_stage1(json.dumps({'document': {**document, 'title': card.person + ' 기록'}, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    bad = {**document, 'text': document['text'].replace(format_event_date(card.date, card.document_date_styles[card.answer_doc_index]), '')}
    with pytest.raises(ValueError, match='absolute event date'):
        parse_stage1(json.dumps({'document': bad, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    bad = {**document, 'text': document['text'].replace(card.fact_value, '')}
    with pytest.raises(ValueError, match='does not contain its fact'):
        parse_stage1(json.dumps({'document': bad, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    bad = {**document, 'text': document['text'] + ' 漢字'}
    with pytest.raises(ValueError, match='Chinese characters'):
        parse_stage1(json.dumps({'document': bad, 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())


def test_r6_rejects_known_clinic_practice_venue_relation_and_gives_safe_retry_hint():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '소아과 예약')
    row = sample_v3(card)
    original = row['evidence']
    wrong = f'발표 연습 장소로 {card.fact_value}를 예약했다.'
    row['documents'][card.answer_doc_index]['text'] = row['documents'][card.answer_doc_index]['text'].replace(original, wrong)
    row['evidence'] = wrong
    with pytest.raises(ValueError, match='Clinic fact relation'):
        validate_event(row)
    assert _retry_code(ValueError('Person name in title')) == 'name_in_title'
    assert _retry_code(ValueError('Clinic fact relation is not a care appointment')) == 'wrong_event_relation'
    assert 'title' in RETRY_HINTS['name_in_title']


@pytest.mark.parametrize('term', ['반려견', '반려묘', '강아지', '고양이', '동물병원', '수의사', '반려동물', '동물 병원', '반려 동물'])
@pytest.mark.parametrize('field', ['answer.title', 'answer.text', 'blind.title', 'blind.text', 'query'])
def test_clinic_rejects_an_animal_patient_in_each_generated_field(term, field):
    card = next(card for card in cards('train', revision=REVISION) if card.category == '소아과 예약')
    row = sample_v3(card)
    with pytest.raises(ValueError, match='Clinic fact relation') as error:
        if field == 'query':
            parse_stage2_component(json.dumps({'query': term + ' ' + row['query']}, ensure_ascii=False), card.as_dict(), 'query')
        else:
            role, name = field.split('.')
            slot = card.answer_doc_index if role == 'answer' else next(slot for slot in range(4) if slot != card.answer_doc_index)
            document = {**row['documents'][slot], name: row['documents'][slot][name] + ' ' + term}
            if role == 'answer':
                parse_stage1(json.dumps({'document': wire_document(card, slot, document), 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
            else:
                parse_stage2_component(json.dumps({'document': {'slot': slot, **wire_document(card, slot, document)}}, ensure_ascii=False), card.as_dict(), f'd{slot}')
    assert _retry_code(error.value) == 'wrong_event_relation'


@pytest.mark.parametrize('category', ['연차 계획', '집 수리'])
def test_pet_care_in_other_categories_keeps_its_original_text(category):
    card = next(card for card in cards('train', revision=REVISION) if card.category == category and '반려동물' in card.secondary_context)
    row = sample_v3(card)
    animals = '반려견 반려묘 강아지 고양이 동물 병원 수의사 반려동물'
    slot = card.answer_doc_index
    document = {**row['documents'][slot], 'title': animals, 'text': row['documents'][slot]['text'] + ' ' + animals}
    result = parse_stage1(json.dumps({'document': wire_document(card, slot, document), 'evidence': row['evidence']}, ensure_ascii=False), card.as_dict())
    assert result == {'document': document, 'evidence': row['evidence']}
    query = animals + ' ' + row['query']
    assert parse_stage2_component(json.dumps({'query': query}, ensure_ascii=False), card.as_dict(), 'query') == {'query': query}


def test_r6_implicit_identity_guard_does_not_mistake_puppet_for_an_older_brother():
    from data_v3 import RELATIONSHIP_MENTION_PATTERN
    assert not RELATIONSHIP_MENTION_PATTERN.search('인형 연기와 비 예보를 확인한다.')
    assert RELATIONSHIP_MENTION_PATTERN.search('형이 함께 간다고 했다.')
    assert RELATIONSHIP_MENTION_PATTERN.search('부모님과 함께 갈 예정이다.')
    card = next(card for card in cards('train', revision=REVISION) if card.identity_style == 'implicit' and card.category == '어린이집 준비' and '인형 연기' in card.situation_cues)
    row = sample_v3(card)
    validate_event(row)
    row['documents'][0]['text'] += ' 형이 함께 갈 예정이다.'
    with pytest.raises(ValueError, match='Identity shortcut'):
        validate_event(row)


def test_r6_blind_clinic_notes_must_keep_the_clinic_action_instead_of_only_the_background():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '소아과 예약')
    row = sample_v3(card)
    slot = next(slot for slot in range(4) if slot != card.answer_doc_index)
    row['documents'][slot]['text'] = row['documents'][slot]['text'].replace('진료', '발표')
    with pytest.raises(ValueError, match='central event action'):
        validate_event(row)


def test_r6_daycare_gold_is_the_notified_item_rather_than_any_logistics_object():
    card = next(card for card in cards('train', revision=REVISION) if card.category == '어린이집 준비')
    assert card.fact_label == '어린이집에서 안내받은 준비물'
    with pytest.raises(ValueError, match='daycare-notified'):
        validate_query_intent('생일 축하에 우천 가능성이 있으면 준비물은 무엇이지?', card.category)
    validate_query_intent('어린이집 선생님이 알려준 준비물은 무엇이지?', card.category)
    row = sample_v3(card)
    notified = f'선생님이 알려준 준비물은 {card.fact_value}이다.'
    row['documents'][card.answer_doc_index]['text'] = row['documents'][card.answer_doc_index]['text'].replace(row['evidence'], notified)
    row['evidence'] = notified
    validate_event(row, provenance=False)
    ungrounded = f'개인 외출 준비물은 {card.fact_value}이다.'
    row['documents'][card.answer_doc_index]['text'] = row['documents'][card.answer_doc_index]['text'].replace(notified, ungrounded)
    row['evidence'] = ungrounded
    with pytest.raises(ValueError, match='Daycare fact relation'):
        validate_event(row)


def test_r7_each_blind_component_has_its_own_whitelist_and_no_other_generated_output():
    card = event_card('train', 0, REVISION).as_dict()
    blind = stage2_payload(card)
    blind['generated_documents'] = ['SECRET_RESPONSE']
    blind['evidence'] = 'SECRET_RESPONSE'
    for component in [f"d{item['slot']}" for item in blind['documents']] + ['query']:
        payload = stage2_component_payload(blind, component)
        encoded = json.dumps(payload, ensure_ascii=False)
        assert 'SECRET' not in encoded
        assert card['fact_value'] not in encoded
        assert 'documents' not in payload
        assert set(stage_schema(payload, 2)['properties']) == ({'query'} if component == 'query' else {'document'})
        if component == 'query':
            assert 'document' not in payload
        else:
            assert payload['document']['slot'] == int(component[1:])
            assert 'query_intent' not in payload and 'query_date_style' not in payload
    query_payload = stage2_component_payload(blind, 'query')
    wrong = {'query': '온천 연락 숙소는 어디인가요? 숙소는 어디였죠?'}
    with pytest.raises(ValueError, match='multiple questions'):
        parse_stage2_component(json.dumps(wrong, ensure_ascii=False), card, 'query')


def test_r7_partial_component_resume_reuses_completed_calls_and_verifies_cache(tmp_path, monkeypatch):
    card = event_card('train', 0, REVISION)
    row = sample_v3(card)
    normal = generation_responses(card, row)
    requests = []
    def interrupted(endpoint, payload):
        requests.append(payload)
        if len(requests) == 4:
            raise ConnectionError('local API interrupted before the third blind response')
        return {'message': {'content': json.dumps(normal[len(requests) - 1], ensure_ascii=False)}}
    monkeypatch.setattr('generate.ollama_request', interrupted)
    folder = tmp_path / 'stages'
    budget = tmp_path / 'budget.json'
    with pytest.raises(ConnectionError):
        generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
    assert len(requests) == 4
    remaining = iter(normal[3:])
    def continued(endpoint, payload):
        requests.append(payload)
        return {'message': {'content': json.dumps(next(remaining), ensure_ascii=False)}}
    monkeypatch.setattr('generate.ollama_request', continued)
    generated = generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
    assert len(requests) == 6
    last_doc = next(reversed([key for key in generated['generation_stages']['stage2']['components'] if key != 'query']))
    attempts = generated['generation_stages']['stage2']['components'][last_doc]['attempt_log']
    assert [item['status'] for item in attempts] == ['interrupted_or_failed', 'completed']
    assert generated['generation_stages']['stage2']['components']['query']['retry_error_codes'] == []
    assert generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24) == generated
    assert len(requests) == 6
    cache_path = folder / f'{card.event_id}.json'
    cached = json.loads(cache_path.read_text())
    cached[f'stage2-{last_doc}']['input']['evidence'] = 'SECRET_HISTORY'
    cache_path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match='whitelist'):
        generate_event(card.as_dict(), 'qwen3:8b', 'a' * 64, folder, budget, 24)
    assert len(requests) == 6


def test_v3_weekend_calendar_keeps_collision_buckets_and_other_topic_dates():
    from datetime import date
    values = list(cards(revision=REVISION))
    assert len(values) == 525
    nonweekend_dates = [(card.event_id, card.date) for card in values if card.category != '주말 약속']
    assert _json_hash(nonweekend_dates) == '19c0f58a81f424c0635b084e9526f827b8560b91f429ee0f0725808cad574800'
    families, categories = defaultdict(list), defaultdict(list)
    for card in values:
        day = date.fromisoformat(card.date)
        assert 0 <= (day - date(2025, 1, 1)).days < 650
        families[card.family_id].append(card)
        categories[(card.split, card.category)].append(card)
        if card.category == '주말 약속':
            assert day.weekday() in (5, 6)
    for members in categories.values():
        buckets = defaultdict(list)
        for card in members:
            buckets[card.collision_group or card.event_id].append(card)
        assert len({group[0].date for group in buckets.values()}) == len(buckets)
        for group in buckets.values():
            assert len({card.date for card in group}) == 1
            if group[0].collision_group:
                assert len(group) == 5 and len({card.family_id for card in group}) == 5
        for family in {card.family_id for card in members}:
            repeated = [card for card in members if card.family_id == family]
            assert len({card.date for card in repeated}) == len(repeated)
    for members in families.values():
        for card in members:
            if card.category == '주말 약속':
                assert all(card.date != other.date for other in members if other.event_id != card.event_id)
