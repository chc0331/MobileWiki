#!/usr/bin/env python3
"""Build an inspectable Chapter 3 classification and extractive-summary baseline."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2] / "_workspace" / "chapter3"
SPLIT_NAMES = ("train", "validation", "final_test")
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def check_splits(records: list[dict], splits: dict) -> dict[str, dict]:
    by_id = {record["doc_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("중복된 문서 ID가 있습니다")

    seen_groups: set[str] = set()
    seen_docs: set[str] = set()
    for name in SPLIT_NAMES:
        part = splits[name]
        group_ids = part["group_ids"]
        doc_ids = part["doc_ids"]
        if len(group_ids) != 3 or len(doc_ids) != 4:
            raise ValueError(f"{name}: 원문 묶음 3개와 문서 4개가 필요합니다")
        if len(set(group_ids)) != len(group_ids) or len(set(doc_ids)) != len(doc_ids):
            raise ValueError(f"{name}: 중복된 원문 묶음 또는 문서 ID가 있습니다")
        if seen_groups.intersection(group_ids) or seen_docs.intersection(doc_ids):
            raise ValueError("분할 사이에 같은 원문 묶음 또는 문서가 있습니다")
        seen_groups.update(group_ids)
        seen_docs.update(doc_ids)
        for doc_id in doc_ids:
            record = by_id[doc_id]
            if record["status"] != "ready" or record["group_id"] not in group_ids:
                raise ValueError(f"{doc_id}: 준비 상태 또는 원문 묶음이 분할과 다릅니다")
        if {by_id[doc_id]["group_id"] for doc_id in doc_ids} != set(group_ids):
            raise ValueError(f"{name}: 문서와 원문 묶음 목록이 다릅니다")

    ready_ids = {record["doc_id"] for record in records if record["status"] == "ready"}
    if seen_docs != ready_ids:
        raise ValueError("준비된 문서가 분할에서 빠졌거나 준비되지 않은 문서가 들어 있습니다")
    return by_id


def classification_report(by_id: dict[str, dict], splits: dict) -> tuple[dict, dict]:
    train_ids = splits["train"]["doc_ids"]
    train_records = [by_id[doc_id] for doc_id in train_ids]

    # Learn the vocabulary and IDF from training text only. Reuse them for both held-out parts.
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform([record["clean_text"] for record in train_records])
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_vectors, [record["label"] for record in train_records])

    classes = [str(label) for label in classifier.classes_]
    terms = vectorizer.get_feature_names_out()
    feature_report = {
        "method": "TfidfVectorizer()",
        "fitted_on_doc_ids": train_ids,
        "vocabulary_size": len(terms),
        "features_in_index_order": [
            {"index": index, "term": term, "idf": float(vectorizer.idf_[index])}
            for index, term in enumerate(terms)
        ],
    }
    model_report = {
        "method": "LogisticRegression(solver='lbfgs', max_iter=1000)",
        "class_order": classes,
        "intercept_by_class": {
            label: float(classifier.intercept_[index])
            for index, label in enumerate(classes)
        },
        "weights_by_class_in_feature_order": {
            label: [float(weight) for weight in classifier.coef_[index]]
            for index, label in enumerate(classes)
        },
        "splits": {},
    }

    for name in SPLIT_NAMES:
        doc_ids = splits[name]["doc_ids"]
        vectors = (train_vectors if name == "train" else
                   vectorizer.transform([by_id[doc_id]["clean_text"] for doc_id in doc_ids]))
        predictions = classifier.predict(vectors)
        decisions = classifier.decision_function(vectors)
        probabilities = classifier.predict_proba(vectors)
        documents = []
        for index, doc_id in enumerate(doc_ids):
            record = by_id[doc_id]
            nonzero = int(vectors[index].getnnz())
            documents.append({
                "doc_id": doc_id,
                "group_id": record["group_id"],
                "language": record["language"],
                "true_label": record["label"],
                "predicted_label": str(predictions[index]),
                "is_error": bool(predictions[index] != record["label"]),
                "nonzero_feature_count": nonzero,
                "all_zero_vector": nonzero == 0,
                "decision_scores": {
                    label: float(decisions[index][class_index])
                    for class_index, label in enumerate(classes)
                },
                "probabilities": {
                    label: float(probabilities[index][class_index])
                    for class_index, label in enumerate(classes)
                },
            })
        model_report["splits"][name] = {
            "group_count": len(splits[name]["group_ids"]),
            "document_count": len(documents),
            "error_count": sum(document["is_error"] for document in documents),
            "documents": documents,
        }
    return feature_report, model_report


def extractive_summary(record: dict) -> dict:
    sentences = record["sentences"]
    token_lists = [TOKEN_PATTERN.findall(sentence.lower()) for sentence in sentences]
    frequencies = Counter(token for tokens in token_lists for token in tokens)
    scores = [sum(frequencies[token] for token in set(tokens)) for tokens in token_lists]

    # Rank by score; earlier source sentences win ties. Restore source order for reading.
    selected = sorted(sorted(range(len(sentences)), key=lambda index: (-scores[index], index))[:2])
    return {
        "doc_id": record["doc_id"],
        "group_id": record["group_id"],
        "sentence_scores": [
            {"source_index": index, "score": scores[index], "text": sentence}
            for index, sentence in enumerate(sentences)
        ],
        "selected_source_indices": selected,
        "selected_sentences": [sentences[index] for index in selected],
        "summary_text": " ".join(sentences[index] for index in selected),
        "human_summary": record["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE / "input",
                        help="Chapter 2 records.json과 splits.json이 있는 디렉터리")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE,
                        help="baseline_report.json을 쓸 디렉터리")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    for name in ("records.json", "splits.json"):
        if not (input_dir / name).is_file():
            raise ValueError(f"{input_dir / name}이 없습니다. Chapter 2 실행 예제를 먼저 실행하세요")
    records = json.loads((input_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((input_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = check_splits(records, splits)
    features, classification = classification_report(by_id, splits)

    original_records = [record for record in records
                        if record["status"] == "ready"
                        and record["doc_id"] == str(record["group_id"])[2:]]
    if len(original_records) != 9:
        raise ValueError("추출 요약에 사용할 준비된 원문은 9개여야 합니다")
    report = {
        "source_record_count": len(records),
        "excluded": [
            {"doc_id": record["doc_id"], "status": record["status"]}
            for record in records if record["status"] != "ready"
        ],
        "group_overlap_count": 0,
        "data_limit": "각 라벨의 학습 원문 묶음은 하나뿐입니다. 이 결과를 일반적인 분류·요약 성능으로 해석하지 마세요.",
        "feature_extraction": features,
        "classification": classification,
        "extractive_summaries": [extractive_summary(record) for record in original_records],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"입력 {len(records)}건, 학습에서 제외 {len(report['excluded'])}건, 원문 묶음 겹침 0건")
    print(f"학습 어휘 {features['vocabulary_size']}개")
    for name in SPLIT_NAMES:
        part = classification["splits"][name]
        print(f"{name}: 원문 묶음 {part['group_count']}개, 문서 {part['document_count']}개, 오분류 {part['error_count']}개")
    print(f"추출 요약: 원문 {len(original_records)}개")
    print(f"출력: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"자료 검사 오류: {error}") from error
