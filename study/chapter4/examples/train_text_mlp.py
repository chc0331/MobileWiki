#!/usr/bin/env python3
"""Chapter 4: train a tiny text network, inspect its history, and reload its weights."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
# Reuse Chapter 3's data checks without copying its splitting rules.
spec = importlib.util.spec_from_file_location(
    "chapter3_baseline", ROOT / "study/chapter3/examples/run_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def make_model(kind: str, input_size: int, class_count: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_size, class_count)
    return nn.Sequential(
        nn.Linear(input_size, 8), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(8, class_count),
    )


def evaluate(model, x, y, doc_ids, labels, by_id):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
        probabilities = logits.softmax(dim=1)
        predicted = logits.argmax(dim=1)
    return {
        "loss": loss,
        "accuracy": (predicted == y).float().mean().item(),
        "documents": [{
            "doc_id": doc_id,
            "language": by_id[doc_id]["language"],
            "true_label": labels[y[i].item()],
            "predicted_label": labels[predicted[i].item()],
            "all_zero_vector": bool(torch.count_nonzero(x[i]) == 0),
            "probabilities": dict(zip(labels, probabilities[i].tolist())),
        } for i, doc_id in enumerate(doc_ids)],
    }


def gradient_demo():
    w = torch.tensor(0.5, requires_grad=True)
    loss = (w * 2 - 3) ** 2
    loss.backward()
    gradient = w.grad.item()
    with torch.no_grad():
        w -= 0.1 * w.grad
    return {"old_weight": 0.5, "old_loss": loss.item(),
            "gradient": gradient, "new_weight": w.item(),
            "new_loss": ((w * 2 - 3) ** 2).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "_workspace/chapter3/input")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_workspace/chapter4/base")
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate-final", action="store_true",
                        help="설정을 정한 뒤에만 최종 평가 결과를 출력합니다")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, lr은 양수여야 합니다")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    records = json.loads((args.input_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((args.input_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = baseline.check_splits(records, splits)
    train_ids = splits["train"]["doc_ids"]
    labels = sorted({by_id[doc_id]["label"] for doc_id in train_ids})
    label_to_id = {label: i for i, label in enumerate(labels)}
    vectorizer = TfidfVectorizer()
    train_sparse = vectorizer.fit_transform([by_id[i]["clean_text"] for i in train_ids])
    parts = {}
    split_names = ("train", "validation", "final_test") if args.evaluate_final else ("train", "validation")
    for name in split_names:
        ids = splits[name]["doc_ids"]
        sparse = train_sparse if name == "train" else vectorizer.transform([by_id[i]["clean_text"] for i in ids])
        # Dense storage is acceptable ONLY for this tiny learning dataset.
        x = torch.tensor(sparse.toarray(), dtype=torch.float32)
        y = torch.tensor([label_to_id[by_id[i]["label"]] for i in ids], dtype=torch.long)
        parts[name] = (x, y, ids)
    x_train, y_train, _ = parts["train"]
    x_val, y_val, _ = parts["validation"]
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = make_model(args.model, x_train.shape[1], len(labels))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        online_sum = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            online_sum += loss.item() * len(y_batch)
        # Both curves use eval() and one fixed set of weights at this epoch's end.
        train_eval = evaluate(model, *parts["train"][:2], train_ids, labels, by_id)
        val_eval = evaluate(model, x_val, y_val, parts["validation"][2], labels, by_id)
        history.append({"epoch": epoch, "online_train_loss": online_sum / len(y_train),
                        "train_eval_loss": train_eval["loss"], "validation_loss": val_eval["loss"],
                        "train_accuracy": train_eval["accuracy"], "validation_accuracy": val_eval["accuracy"]})
        if val_eval["loss"] < best_loss:
            best_loss, best_epoch = val_eval["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model_state.pt")
    config = {"model": args.model, "input_size": x_train.shape[1], "class_order": labels,
              "tfidf": {"vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
                        "idf": vectorizer.idf_.tolist()}}
    (args.output_dir / "features_and_model.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Reconstruct every input column from the saved vocabulary, not a new fit.
    saved = json.loads((args.output_dir / "features_and_model.json").read_text(encoding="utf-8"))
    restored_vectorizer = TfidfVectorizer(vocabulary=saved["tfidf"]["vocabulary"])
    restored_vectorizer.idf_ = np.array(saved["tfidf"]["idf"])
    restored = make_model(saved["model"], saved["input_size"], len(saved["class_order"]))
    restored.load_state_dict(torch.load(args.output_dir / "model_state.pt", map_location="cpu", weights_only=True))
    restored.eval()
    restored_x = torch.tensor(restored_vectorizer.transform([by_id[i]["clean_text"] for i in parts["validation"][2]]).toarray(), dtype=torch.float32)
    with torch.no_grad():
        difference = (model(x_val) - restored(restored_x)).abs().max().item()
    if difference > 1e-7:
        raise RuntimeError(f"저장 전후 출력이 다릅니다: {difference}")
    logistic = LogisticRegression(solver="lbfgs", max_iter=1000).fit(train_sparse, [by_id[i]["label"] for i in train_ids])
    comparison = {}
    for name in split_names:
        ids = parts[name][2]
        predictions = logistic.predict(vectorizer.transform([by_id[i]["clean_text"] for i in ids]))
        comparison[name] = {"correct_count": sum(p == by_id[i]["label"] for p, i in zip(predictions, ids)),
                            "document_count": len(ids)}
    report = {"settings": {"model": args.model, "lr": args.lr, "epochs": args.epochs,
                           "batch_size": args.batch_size, "seed": args.seed, "device": "cpu"},
              "versions": {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__},
              "gradient_demo": gradient_demo(), "class_order": labels, "input_shape": list(x_train.shape),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "fitted_on_doc_ids": train_ids, "group_overlap_count": 0,
              "best_epoch_by_validation_loss": best_epoch, "history": history,
              "reload_max_logit_difference": difference,
              "splits": {name: evaluate(model, *part, labels, by_id) for name, part in parts.items()},
              "chapter3_logistic_comparison": comparison,
              "data_limit": "학습 원문 묶음이 종류마다 하나뿐입니다. 실제 서비스 성능을 추정하지 않습니다."}
    path = args.output_dir / "training_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"input {list(x_train.shape)}, labels {labels}, parameters {report['parameter_count']}")
    print(f"best epoch {best_epoch}; reload max difference {difference:.8f}")
    for name, result in report["splits"].items():
        print(f"{name}: loss {result['loss']:.4f}, accuracy {result['accuracy']:.2f}")
    print(path)


if __name__ == "__main__":
    main()
