#!/usr/bin/env python3
"""Chapter 5: inspect tokens, frozen-encoder classification, and short generation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
GENERATOR = "google/flan-t5-small"
GENERATOR_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
PAIRS = [
    ("ko_it", "배터리 충전 시간이 줄었다."),
    ("en_it", "Battery charging time decreased."),
    ("ko_sport", "선수가 경기에서 골을 넣었다."),
    ("en_sport", "The player scored a goal in the game."),
]
SUMMARY_INPUTS = [
    ("en", "On May 3, the library opened a new reading room. It has 20 seats. Admission is free. On Mondays the room is closed."),
    ("ko", "5월 3일 도서관에 새 열람실이 문을 열었다. 좌석은 20개다. 입장은 무료다. 월요일에는 열람실을 닫는다."),
]


def toy_report():
    vocabulary = {token: i for i, token in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "배터리", "가", "오래", "간다", ".", "충전"])}
    texts = ["배터리 가 오래 간다 .", "배터리 충전 ."]
    sequences = [[2] + [vocabulary.get(t, 1) for t in text.split()] + [3] for text in texts]
    length = max(map(len, sequences))
    ids = [row + [0] * (length - len(row)) for row in sequences]
    mask = [[1] * len(row) + [0] * (length - len(row)) for row in sequences]
    return {"kind": "손으로 정한 학습용 모형이며 사전학습 모델이 아닙니다",
            "vocabulary": vocabulary, "texts": texts, "input_ids": ids, "attention_mask": mask,
            "embedding_for_id_4": [0.2, 0.8],
            "attention_example": {"weights": [0.1, 0.6, 0.3], "values": [[1, 0], [0, 2], [1, 1]], "weighted_sum": [0.4, 1.5]},
            "chunk_probabilities": [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
            "mean_probabilities": [0.5, 0.4, 0.1]}


def load_tokenizer(args, model_name, revision):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True,
                                         cache_dir=args.cache_dir, local_files_only=args.offline)


def tokenize_report(args):
    tokenizer = load_tokenizer(args, ENCODER, ENCODER_REVISION)
    rows = []
    texts = [text for _, text in PAIRS]
    batch = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length,
                      return_offsets_mapping=True)
    for i, (name, text) in enumerate(PAIRS):
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        offsets = batch["offset_mapping"][i]
        kept_end = max((end for start, end in offsets if end > start), default=0)
        rows.append({"name": name, "text": text, "utf8_byte_count": len(text.encode("utf-8")),
                     "character_count": len(text), "full_token_count": len(full_ids),
                     "input_ids": batch["input_ids"][i],
                     "tokens": tokenizer.convert_ids_to_tokens(batch["input_ids"][i]),
                     "attention_mask": batch["attention_mask"][i], "offset_mapping": offsets,
                     "kept_source_end": kept_end, "omitted_source_suffix": text[kept_end:],
                     "truncated": len(full_ids) > args.max_length})
    return {"model_id": ENCODER, "revision": ENCODER_REVISION,
            "max_length_including_special_tokens": args.max_length,
            "special_tokens": tokenizer.special_tokens_map, "documents": rows}


def classify_report(args):
    import torch
    from torch import nn
    from transformers import AutoModel
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    tokenizer = load_tokenizer(args, ENCODER, ENCODER_REVISION)
    encoder = AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION,
                                       cache_dir=args.cache_dir, local_files_only=args.offline,
                                       use_safetensors=True)
    encoder.requires_grad_(False)
    encoder.eval()

    def encode_batch(inputs):
        with torch.no_grad():
            hidden = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def encode(texts):
        return encode_batch(tokenizer(texts, padding=True, truncation=True,
                                      max_length=args.max_length, return_tensors="pt"))

    spec = importlib.util.spec_from_file_location("chapter3_baseline", ROOT / "study/chapter3/examples/run_baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    records = json.loads((args.input_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((args.input_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = baseline.check_splits(records, splits)
    train_ids = splits["train"]["doc_ids"]
    labels = sorted({by_id[i]["label"] for i in train_ids})
    parts = {}
    for name in ("train", "validation"):
        ids = splits[name]["doc_ids"]
        parts[name] = (encode([by_id[i]["clean_text"] for i in ids]),
                       torch.tensor([labels.index(by_id[i]["label"]) for i in ids]), ids)
    head = nn.Linear(encoder.config.hidden_size, len(labels))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    x, y, _ = parts["train"]
    for _ in range(80):
        head.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(x), y)
        loss.backward()
        optimizer.step()
    head.eval()

    def predict(vectors):
        with torch.no_grad():
            return head(vectors).softmax(dim=1)

    results = {}
    for name, (vectors, targets, ids) in parts.items():
        probabilities = predict(vectors)
        results[name] = [{"doc_id": doc_id, "language": by_id[doc_id]["language"],
                          "true_label": labels[targets[i].item()],
                          "predicted_label": labels[probabilities[i].argmax().item()],
                          "probabilities": dict(zip(labels, probabilities[i].tolist()))} for i, doc_id in enumerate(ids)]
    pair_probabilities = predict(encode([text for _, text in PAIRS]))
    language_pairs = [
        {"name": name, "text": text,
         "intended_label": "IT" if name.endswith("it") else "스포츠",
         "predicted_label": labels[pair_probabilities[i].argmax().item()],
         "probabilities": dict(zip(labels, pair_probabilities[i].tolist()))}
        for i, (name, text) in enumerate(PAIRS)
    ]
    # A constructed long text exercises overflow handling, not document-quality evaluation.
    source_id = parts["validation"][2][0]
    long_text = (by_id[source_id]["clean_text"] + "\n") * 8
    chunk_inputs = tokenizer(long_text, truncation=True, max_length=args.max_length,
                             stride=min(8, args.max_length - 3), return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding=True, return_tensors="pt")
    chunk_inputs.pop("overflow_to_sample_mapping")
    offsets = chunk_inputs.pop("offset_mapping")
    chunk_vectors = encode_batch(chunk_inputs)
    chunk_probabilities = predict(chunk_vectors)
    merged = chunk_probabilities.mean(dim=0)
    full_count = len(tokenizer(long_text, add_special_tokens=False, verbose=False)["input_ids"])
    # Token identity can stay equal while context-dependent representations change.
    bank_texts = ["I sat by the bank.", "I went to the bank for money."]
    bank_inputs = tokenizer(bank_texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        base_vectors = encoder.get_input_embeddings()(bank_inputs["input_ids"])
        contextual = encoder(**bank_inputs).last_hidden_state
    positions = []
    for row in bank_inputs["input_ids"]:
        pieces = tokenizer.convert_ids_to_tokens(row.tolist())
        positions.append(next(i for i, token in enumerate(pieces) if token.lstrip("▁").lower() == "bank"))
    a, b = positions
    context_demo = {"texts": bank_texts,
                    "token_id_equal": bool(bank_inputs["input_ids"][0, a] == bank_inputs["input_ids"][1, b]),
                    "input_embedding_max_difference": (base_vectors[0, a] - base_vectors[1, b]).abs().max().item(),
                    "contextual_max_difference": (contextual[0, a] - contextual[1, b]).abs().max().item()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.output_dir / "classifier_head.pt")
    return {"encoder_id": ENCODER, "revision": ENCODER_REVISION, "encoder_frozen": True,
            "encoder_parameter_count": sum(p.numel() for p in encoder.parameters()),
            "head_parameter_count": sum(p.numel() for p in head.parameters()),
            "training": {"epochs": 80, "lr": 0.1, "seed": 7, "batch_size": len(train_ids)},
            "class_order": labels, "trained_on_doc_ids": train_ids,
            "max_length": args.max_length, "splits": results,
            "language_pairs": language_pairs, "context_demo": context_demo,
            "long_example": {"kind": "검증 문서를 반복한 처리 연습용 입력", "source_doc_id": source_id,
                             "content_token_count": full_count, "chunk_count": len(offsets),
                             "source_ranges": [[min(start for start, end in row.tolist() if end > start),
                                                max(end for start, end in row.tolist() if end > start)] for row in offsets],
                             "chunk_probabilities": chunk_probabilities.tolist(),
                             "mean_probabilities": dict(zip(labels, merged.tolist())),
                             "predicted_label": labels[merged.argmax().item()]},
            "data_limit": "작은 자료의 학습 흐름을 확인합니다. 일반적인 다국어 분류 성능이 아닙니다."}


def generate_report(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    torch.set_num_threads(1)
    tokenizer = load_tokenizer(args, GENERATOR, GENERATOR_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR, revision=GENERATOR_REVISION,
                                                 cache_dir=args.cache_dir, local_files_only=args.offline,
                                                 use_safetensors=True)
    model.eval()
    rows = []
    for language, source in SUMMARY_INPUTS:
        prompt = "Summarize in the same language in one sentence: " + source
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, num_beams=1,
                                    max_new_tokens=args.max_new_tokens)
        ids = output[0].tolist()
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        rows.append({"language": language, "source": source, "prompt": prompt,
                     "input_ids": inputs["input_ids"][0].tolist(),
                     "input_token_count": inputs["input_ids"].shape[1],
                     "unknown_token_count": int((inputs["input_ids"] == tokenizer.unk_token_id).sum()),
                     "generated_ids_including_decoder_start": ids,
                     "new_token_count": len(ids) - 1,
                     "has_eos": tokenizer.eos_token_id in ids[1:],
                     "summary": summary, "empty_summary": not summary.strip()})
    return {"model_id": GENERATOR, "revision": GENERATOR_REVISION,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "settings": {"do_sample": False, "num_beams": 1, "input_max_length": 256,
                         "max_new_tokens": args.max_new_tokens}, "documents": rows,
            "limit": "다운로드·실행 가능성과 한국어 요약 품질은 별개입니다. 원문 근거를 사람이 대조합니다."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("toy", "tokenize", "classify", "generate"))
    parser.add_argument("--input-dir", type=Path, default=ROOT / "_workspace/chapter3/input")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_workspace/chapter5")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "_workspace/chapter5/cache")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--offline", action="store_true", help="이미 받은 모델·토크나이저만 사용합니다")
    args = parser.parse_args()
    if not 8 <= args.max_length <= 128 or args.max_new_tokens < 1:
        parser.error("max-length는 8~128, max-new-tokens는 양수여야 합니다")
    funcs = {"toy": toy_report, "tokenize": tokenize_report,
             "classify": classify_report, "generate": generate_report}
    report = funcs[args.stage]() if args.stage == "toy" else funcs[args.stage](args)
    if args.stage != "toy":
        import platform
        import torch
        import transformers
        report["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.stage}_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    if args.stage == "tokenize":
        for row in report["documents"]:
            print(row["name"], row["full_token_count"], row["tokens"])
    elif args.stage == "generate":
        for row in report["documents"]:
            print(row["language"], "unknown", row["unknown_token_count"], "summary", repr(row["summary"]))
    elif args.stage == "classify":
        print("encoder frozen", report["encoder_frozen"], "head parameters", report["head_parameter_count"])
        print("long example chunks", report["long_example"]["chunk_count"])


if __name__ == "__main__":
    main()
