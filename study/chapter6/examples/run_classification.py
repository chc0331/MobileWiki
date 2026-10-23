#!/usr/bin/env python3
"""Chapter 6: compare two document classifiers and evaluate selective predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LABELS = ["IT", "경제", "스포츠"]
OUTSIDE = "범위 밖"
HOLD = "보류"
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_LENGTH = 128
MODELS = ("tfidf", "encoder")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data(records):
    groups = defaultdict(list)
    seen_ids, seen_texts = set(), set()
    for row in records:
        if row["doc_id"] in seen_ids or row["clean_text"] in seen_texts:
            raise ValueError("중복 문서 ID 또는 완전히 같은 본문이 있습니다")
        if not row["clean_text"].strip() or row["language"] not in ("ko", "en"):
            raise ValueError("빈 본문 또는 지원하지 않는 언어입니다")
        if row["label"] not in LABELS + [OUTSIDE] or row["split"] not in ("train", "validation", "final_test"):
            raise ValueError("라벨 또는 자료 분할 이름이 다릅니다")
        if row["split"] == "train" and row["label"] == OUTSIDE:
            raise ValueError("범위 밖 문서는 분류 층의 학습에 쓰지 않습니다")
        groups[row["group_id"]].append(row)
        seen_ids.add(row["doc_id"])
        seen_texts.add(row["clean_text"])
    for members in groups.values():
        if len({r["split"] for r in members}) != 1 or len({r["label"] for r in members}) != 1:
            raise ValueError("같은 원문 묶음의 분할 또는 라벨이 다릅니다")
        if len({r["language"] for r in members}) != len(members):
            raise ValueError("한 원문 묶음에 같은 언어가 중복됐습니다")
    counts = {}
    for split in ("train", "validation", "final_test"):
        part = [r for r in records if r["split"] == split]
        if {r["label"] for r in part if r["label"] != OUTSIDE} != set(LABELS):
            raise ValueError(f"{split}: 범위 안의 세 라벨이 모두 필요합니다")
        counts[split] = {"documents": len(part), "groups": len({r["group_id"] for r in part}),
                         "labels": dict(Counter(r["label"] for r in part)),
                         "languages": dict(Counter(r["language"] for r in part))}
    return counts


def metrics(rows, threshold):
    """Keep outside truths and held in-scope documents in all relevant denominators."""
    matrix = [[0] * 4 for _ in range(4)]
    for row in rows:
        probabilities = row["probabilities"]
        best = max(range(3), key=lambda i: probabilities[i])
        column = best if probabilities[best] >= threshold else 3
        matrix[(LABELS + [OUTSIDE]).index(row["label"])][column] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support, "predicted": predicted,
                            "precision_undefined": predicted == 0, "recall_undefined": support == 0}
    total = len(rows)
    held = sum(row[3] for row in matrix)
    automatic = total - held
    correct = sum(matrix[i][i] for i in range(3))
    known = sum(sum(matrix[i]) for i in range(3))
    outside = sum(matrix[3])
    return {"row_labels": LABELS + [OUTSIDE], "column_labels": LABELS + [HOLD],
            "matrix": matrix, "documents": total, "groups": len({r["group_id"] for r in rows}),
            "per_class": per_class, "known_accuracy": correct / known if known else None,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 3,
            "weighted_f1": sum(v["f1"] * v["support"] for v in per_class.values()) / known if known else None,
            "automatic_count": automatic, "automatic_errors": automatic - correct,
            "automatic_error_rate": (automatic - correct) / automatic if automatic else None,
            "hold_rate": held / total if total else None,
            "coverage": automatic / total if total else None,
            "outside_false_accept_rate": sum(matrix[3][:3]) / outside if outside else None}


def sliced_metrics(rows, threshold):
    result = {"overall": metrics(rows, threshold)}
    for field in ("language", "length_bucket"):
        result[field] = {value: metrics([r for r in rows if r[field] == value], threshold)
                         for value in sorted({r[field] for r in rows})}
    return result


def toy_report():
    matrix = [[3, 1, 0, 1], [1, 2, 1, 0], [0, 0, 2, 1], [1, 1, 0, 2]]
    rows = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            for _ in range(count):
                p = [0.8 if k == j else 0.1 for k in range(3)] if j < 3 else [0.4, 0.35, 0.25]
                rows.append({"label": (LABELS + [OUTSIDE])[i], "group_id": f"toy-{len(rows)}", "probabilities": p})
    return {"kind": "본문의 손 계산을 확인하도록 사람이 정한 수치",
            "threshold": 0.6, "metrics": metrics(rows, 0.6),
            "all_held": metrics(rows, 1.01)}


def setup_torch():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    return torch


def load_encoder(args):
    from transformers import AutoModel, AutoTokenizer
    common = {"revision": REVISION, "cache_dir": args.cache_dir, "local_files_only": args.offline}
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, use_fast=True, **common)
    encoder = AutoModel.from_pretrained(ENCODER, use_safetensors=True, **common)
    encoder.requires_grad_(False)
    encoder.eval()
    return tokenizer, encoder


def document_chunks(text, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    budget = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
    # Prefer the last complete paragraph fitting in the token budget.
    ends = []
    for index in range(len(text) - 1):
        if text[index:index + 2] == "\n\n":
            ends.append(sum(end <= index for start, end in offsets))
    chunks = []
    start = 0
    while start < len(ids):
        limit = min(start + budget, len(ids))
        boundaries = [end for end in ends if start < end <= limit]
        end = max(boundaries) if boundaries and limit < len(ids) else limit
        chunks.append((ids[start:end], [offsets[start][0], offsets[end - 1][1]]))
        start = end
    first = ids[:budget]
    first_span = [offsets[0][0], offsets[len(first) - 1][1]]
    return chunks, first, len(ids) + MAX_LENGTH - budget, first_span


def encode_documents(records, tokenizer, encoder, torch):
    vectors, details = [], []
    for row in records:
        chunks, first, full_count, first_span = document_chunks(row["clean_text"], tokenizer)
        sequences = [tokenizer.build_inputs_with_special_tokens(ids) for ids, _ in chunks]
        # Keep the first-window control separate even if a paragraph boundary shortened chunk 1.
        sequences.append(tokenizer.build_inputs_with_special_tokens(first))
        encoded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(pooled[:-1].mean(0))
        details.append({"full_token_count": full_count, "length_bucket": "short" if full_count <= MAX_LENGTH else "long",
                        "chunk_count": len(chunks), "source_ranges": [span for _, span in chunks],
                        "chunk_vectors": pooled[:-1], "first_vector": pooled[-1], "first_source_range": first_span})
    return torch.stack(vectors), details


def predicted_rows(records, probabilities, details, input_policy="mean_chunk_vectors"):
    rows = []
    for record, probs, detail in zip(records, probabilities, details):
        values = [float(v) for v in probs]
        best = max(range(3), key=lambda i: values[i])
        ranges = detail["source_ranges"]
        if input_policy == "full_text":
            ranges = [[0, len(record["clean_text"])]]
        elif input_policy == "first_window":
            ranges = [detail["first_source_range"]]
        rows.append({k: record[k] for k in ("doc_id", "group_id", "language", "label")} |
                    {k: detail[k] for k in ("full_token_count", "length_bucket", "chunk_count", "source_ranges")} |
                    {"input_policy": input_policy, "source_ranges": ranges, "chunk_count": len(ranges),
                     "probabilities": values, "predicted_label": LABELS[best], "max_score": values[best]})
    return rows


def save_csv(path, models, threshold_by_model=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "doc_id", "group_id", "language", "label", "length_bucket", "full_token_count",
                  "predicted_label", "max_score", "decision", "is_automatic_error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, rows in models.items():
            for row in rows:
                t = threshold_by_model.get(name, 0.0) if threshold_by_model else 0.0
                decision = row["predicted_label"] if row["max_score"] >= t else HOLD
                writer.writerow({k: row[k] for k in fields if k in row} |
                                {"model": name, "decision": decision,
                                 "is_automatic_error": decision != HOLD and decision != row["label"]})


def model_hashes(output):
    return {name: digest(output / name) for name in ("baseline.joblib", "classifier_head.pt", "training.json")}


def train(args, records, counts):
    if (args.output_dir / "final_report.json").exists():
        raise ValueError("이미 최종 평가한 실행입니다. 기존 결과를 읽고 POC A에서 새 자료로 실험하세요")
    import joblib
    import numpy as np
    import sklearn
    import transformers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    torch = setup_torch()
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform([r["clean_text"] for r in training])
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(x, [r["label"] for r in training])
    tokenizer, encoder = load_encoder(args)
    train_x, train_details = encode_documents(training, tokenizer, encoder, torch)
    val_x, details = encode_documents(validation, tokenizer, encoder, torch)
    head = torch.nn.Linear(384, 3)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    targets = torch.tensor([LABELS.index(r["label"]) for r in training])
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(train_x), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    head.eval()
    with torch.no_grad():
        encoder_p = head(val_x).softmax(1).tolist()
        first_p = head(torch.stack([d["first_vector"] for d in details])).softmax(1).tolist()
        mean_p = [head(d["chunk_vectors"]).softmax(1).mean(0).tolist() for d in details]
    baseline_p = baseline.predict_proba(vectorizer.transform([r["clean_text"] for r in validation]))
    baseline_p = baseline_p[:, [list(baseline.classes_).index(label) for label in LABELS]]
    models = {"tfidf": predicted_rows(validation, baseline_p, details, "full_text"),
              "encoder": predicted_rows(validation, encoder_p, details)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": baseline}, args.output_dir / "baseline.joblib")
    torch.save(head.state_dict(), args.output_dir / "classifier_head.pt")
    metadata = {"data_sha256": digest(args.data_file), "class_order": LABELS, "counts": counts,
                "trained_on_doc_ids": [r["doc_id"] for r in training], "vocabulary_size": len(vectorizer.vocabulary_),
                "encoder_id": ENCODER, "revision": REVISION, "encoder_frozen": True,
                "encoder_parameters": sum(p.numel() for p in encoder.parameters()), "head_parameters": 1155,
                "training": {"seed": 7, "epochs": 80, "lr": 0.1, "batch_size": len(training), "device": "cpu"},
                "training_losses": losses, "max_length": MAX_LENGTH, "pooling": "mean of masked-mean chunk vectors",
                "training_lengths": [{"doc_id": r["doc_id"], "tokens": d["full_token_count"], "bucket": d["length_bucket"]}
                                     for r, d in zip(training, train_details)],
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}}
    write_json(args.output_dir / "training.json", metadata)
    report = {"class_order": LABELS, "models": models,
              "metrics_without_hold": {name: sliced_metrics(rows, 0.0) for name, rows in models.items()},
              "encoder_controls": {"first_window": predicted_rows(validation, first_p, details, "first_window"),
                                   "mean_chunk_probabilities": predicted_rows(validation, mean_p, details, "mean_chunk_probabilities")}}
    write_json(args.output_dir / "validation.json", report)
    save_csv(args.output_dir / "validation_predictions.csv", models)
    # Verify both serialized models on validation inputs, without touching final-test predictions.
    restored = joblib.load(args.output_dir / "baseline.joblib")
    restored_p = restored["classifier"].predict_proba(restored["vectorizer"].transform([r["clean_text"] for r in validation]))
    restored_head = torch.nn.Linear(384, 3)
    restored_head.load_state_dict(torch.load(args.output_dir / "classifier_head.pt", weights_only=True))
    with torch.no_grad():
        difference = (head(val_x) - restored_head(val_x)).abs().max().item()
    write_json(args.output_dir / "reload_check.json",
               {"head_max_logit_difference": difference,
                "baseline_max_probability_difference": float(np.max(np.abs(restored_p - baseline.predict_proba(vectorizer.transform([r["clean_text"] for r in validation])))))})
    print("학습·검증 완료. 최종 평가 예측은 아직 계산하지 않았습니다.", flush=True)


def select(args):
    if (args.output_dir / "final_report.json").exists():
        raise ValueError("이미 최종 평가한 실행입니다. 선택을 다시 조정하지 말고 기존 결과를 읽으세요")
    training = read_json(args.output_dir / "training.json")
    if training["data_sha256"] != digest(args.data_file):
        raise ValueError("자료가 학습 후 바뀌었습니다. 별도 출력 경로에서 다시 학습하세요")
    validation = read_json(args.output_dir / "validation.json")
    candidates = []
    for name in MODELS:
        for step in range(21):
            threshold = step / 20
            m = metrics(validation["models"][name], threshold)
            eligible = (m["automatic_error_rate"] is not None and m["automatic_error_rate"] <= 0.1
                        and m["coverage"] >= 0.5)
            candidates.append({"model": name, "threshold": threshold, "eligible": eligible, "metrics": m})
    eligible = [c for c in candidates if c["eligible"]]
    # A stable final tie uses the lower threshold within the same model.
    selected = max(eligible, key=lambda c: (c["metrics"]["coverage"], c["metrics"]["macro_f1"],
                                           c["model"] == "tfidf", -c["threshold"])) if eligible else None
    report = {"status": "selected" if selected else "requirements_not_met", "selected": selected,
              "policy": {"max_automatic_error_rate": 0.1, "min_coverage": 0.5, "threshold_step": 0.05},
              "data_sha256": training["data_sha256"], "model_hashes": model_hashes(args.output_dir),
              "validation_sha256": digest(args.output_dir / "validation.json"), "candidates": candidates}
    write_json(args.output_dir / "selection.json", report)
    print(report["status"], selected and (selected["model"], selected["threshold"]), flush=True)


def evaluate_final(args, records):
    import joblib
    selection = read_json(args.output_dir / "selection.json")
    if selection["data_sha256"] != digest(args.data_file) or selection["model_hashes"] != model_hashes(args.output_dir):
        raise ValueError("자료 또는 모델이 선택 후 바뀌었습니다")
    if selection["validation_sha256"] != digest(args.output_dir / "validation.json"):
        raise ValueError("검증 보고서가 선택 후 바뀌었습니다")
    final_path = args.output_dir / "final_report.json"
    if final_path.exists():
        raise ValueError("이 출력 경로에는 최종 평가 결과가 있습니다. 기존 결과를 읽고 POC A의 새 평가 자료를 준비하세요")
    testing = [r for r in records if r["split"] == "final_test"]
    torch = setup_torch()
    tokenizer, encoder = load_encoder(args)
    vectors, details = encode_documents(testing, tokenizer, encoder, torch)
    baseline = joblib.load(args.output_dir / "baseline.joblib")
    clf = baseline["classifier"]
    p = clf.predict_proba(baseline["vectorizer"].transform([r["clean_text"] for r in testing]))
    p = p[:, [list(clf.classes_).index(label) for label in LABELS]]
    head = torch.nn.Linear(384, 3)
    head.load_state_dict(torch.load(args.output_dir / "classifier_head.pt", weights_only=True))
    head.eval()
    with torch.no_grad():
        encoder_p = head(vectors).softmax(1).tolist()
    models = {"tfidf": predicted_rows(testing, p, details, "full_text"), "encoder": predicted_rows(testing, encoder_p, details)}
    write_json(args.output_dir / "final_predictions.json", {"class_order": LABELS, "models": models})
    selected = selection["selected"]
    report = {"selection_status": selection["status"], "selection_sha256": digest(args.output_dir / "selection.json"),
              "comparison_without_hold": {name: sliced_metrics(rows, 0) for name, rows in models.items()},
              "selected": ({"model": selected["model"], "threshold": selected["threshold"],
                            "metrics": sliced_metrics(models[selected["model"]], selected["threshold"])} if selected else None),
              "limit": "교육용 소규모 자료의 결과입니다. 최종 평가 결과로 모델·임계값을 다시 고르지 않습니다."}
    save_csv(args.output_dir / "final_predictions.csv", models,
             {selected["model"]: selected["threshold"]} if selected else {})
    write_json(final_path, report)
    print(final_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("toy", "train", "select", "evaluate-final"))
    parser.add_argument("--data-file", type=Path, default=Path(__file__).with_name("documents.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_workspace/chapter6/run")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "_workspace/chapter5/cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.stage == "toy":
        write_json(args.output_dir / "toy_report.json", toy_report())
        print(args.output_dir / "toy_report.json")
        return
    records = read_json(args.data_file)
    counts = check_data(records)
    if args.stage == "train":
        train(args, records, counts)
    elif args.stage == "select":
        select(args)
    else:
        evaluate_final(args, records)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"실습 자료·실행 순서 확인: {error}") from error
