#!/usr/bin/env python3
"""Convert the pinned fine-tuned EmbeddingGemma 2 adapter for Android."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_MODEL = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
PROBES = (
    "task: search result | query: 아이 병원 어디로 가기로 했었지?",
    "title: 소아과 예약 | text: 아이 진료는 푸른숲 소아과에서 받기로 했다.",
    "task: clustering | query: 가족 여행\n바람길 펜션에 금요일 저녁 도착할 예정이다.",
)


class MeanPooledEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


class Int8RowEmbedding(torch.nn.Module):
    """Store each vocabulary row as symmetric int8 and restore it at lookup time."""

    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        weight = embedding.weight.detach().float()
        row_scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = torch.round(weight / row_scales).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", quantized, persistent=True)
        self.register_buffer("row_scales", row_scales, persistent=True)
        self.register_buffer("embed_scale", embedding.embed_scale.detach().float().clone(), persistent=False)
        self.padding_idx = embedding.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        quantized = torch.nn.functional.embedding(input_ids, self.weight_int8, self.padding_idx)
        row_scales = torch.nn.functional.embedding(input_ids, self.row_scales, self.padding_idx)
        return quantized.float() * row_scales * self.embed_scale.to(device=input_ids.device)


def token_embedding_name(encoder: torch.nn.Module) -> str:
    embeddings = [(name, module) for name, module in encoder.named_modules() if isinstance(module, torch.nn.Embedding)]
    if len(embeddings) != 1:
        raise SystemExit(f"Expected exactly one token embedding table, found {len(embeddings)}")
    return embeddings[0][0]


def quantize_token_embedding(encoder: torch.nn.Module, name: str) -> None:
    parts = name.split(".")
    parent = encoder
    for part in parts[:-1]:
        parent = getattr(parent, part)
    embedding = getattr(parent, parts[-1])
    if not isinstance(embedding, torch.nn.Embedding):
        raise SystemExit("Token embedding module changed during export")
    setattr(parent, parts[-1], Int8RowEmbedding(embedding))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("poc/embedding/artifacts/trained/adapter"))
    parser.add_argument("--training", type=Path, default=Path("poc/embedding/artifacts/trained/training.json"))
    parser.add_argument("--hub-publish", type=Path, default=Path("poc/embedding/artifacts/hub_publish.json"))
    parser.add_argument("--output", type=Path, default=Path("poc/android/app/src/main/assets/model/embedding_encoder.onnx"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--precision", choices=("float32", "int8_embedding"), default="int8_embedding")
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()

    training = json.loads(args.training.read_text(encoding="utf-8"))
    hub_publish = json.loads(args.hub_publish.read_text(encoding="utf-8"))
    if training["base_revision"] != BASE_REVISION:
        raise SystemExit(f"Pinned base revision mismatch: {training['base_revision']}")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")

    torch.set_num_threads(4)
    sentence_model = SentenceTransformer(
        BASE_MODEL,
        revision=BASE_REVISION,
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
        default_prompt_name=None,
    )
    sentence_model.load_adapter(str(args.adapter))
    sentence_model.max_seq_length = args.sequence_length
    sentence_model.eval()
    encoder = MeanPooledEncoder(sentence_model[0].auto_model).eval()

    encoded = sentence_model.tokenize(list(PROBES))
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] > args.sequence_length:
        raise SystemExit("Probe text exceeds configured sequence length")
    pad_count = args.sequence_length - input_ids.shape[1]
    if pad_count:
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_count), value=sentence_model.tokenizer.pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_count), value=0)

    with torch.inference_mode():
        reference = sentence_model.encode(
            list(PROBES),
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        wrapped = encoder(input_ids, attention_mask).cpu().numpy()
    cosine = np.sum(reference * wrapped, axis=1)
    if not np.all(cosine > 0.9999):
        raise SystemExit(f"Mean-pooling wrapper changed the reference output: {cosine.tolist()}")

    precision_cosine = cosine
    if args.precision == "int8_embedding":
        embedding_name = token_embedding_name(encoder)
        quantize_token_embedding(encoder, embedding_name)
        with torch.inference_mode():
            quantized_vectors = encoder(input_ids, attention_mask).cpu().numpy()
        precision_cosine = np.sum(reference * quantized_vectors, axis=1)
        if not np.all(precision_cosine > 0.999):
            raise SystemExit(f"Row-wise int8 embedding lookup changed reference embeddings: {precision_cosine.tolist()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequence_dim = torch.export.Dim("sequence", min=2, max=args.sequence_length)
    torch.onnx.export(
        encoder,
        (input_ids[:1].contiguous(), attention_mask[:1].contiguous()),
        str(args.output),
        input_names=("input_ids", "attention_mask"),
        output_names=("embeddings",),
        opset_version=18,
        dynamo=True,
        dynamic_shapes={
            "input_ids": {1: sequence_dim},
            "attention_mask": {1: sequence_dim},
        },
        external_data=False,
        artifacts_dir=str(args.output.parent / "onnx_export_diagnostics"),
    )
    if args.output.stat().st_size > args.max_bytes:
        raise SystemExit(
            f"Converted model is {args.output.stat().st_size} bytes; limit is {args.max_bytes}"
        )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_vectors = []
    for index in range(len(PROBES)):
        values = session.run(
            ["embeddings"],
            {
                "input_ids": input_ids[index:index + 1].cpu().numpy().astype(np.int64),
                "attention_mask": attention_mask[index:index + 1].cpu().numpy().astype(np.int64),
            },
        )[0][0]
        ort_vectors.append(values)
    ort_vectors = np.asarray(ort_vectors)
    conversion_cosine = np.sum(reference * ort_vectors, axis=1)
    if not np.all(conversion_cosine > 0.999):
        raise SystemExit(f"ONNX Runtime changed the reference embeddings: {conversion_cosine.tolist()}")

    dynamic_cosines = []
    for text in PROBES:
        single = sentence_model.tokenize([text])
        with torch.inference_mode():
            expected = sentence_model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        actual = session.run(
            ["embeddings"],
            {
                "input_ids": single["input_ids"].cpu().numpy().astype(np.int64),
                "attention_mask": single["attention_mask"].cpu().numpy().astype(np.int64),
            },
        )[0][0]
        dynamic_cosines.append(float(np.dot(expected, actual)))
    if not np.all(np.asarray(dynamic_cosines) > 0.999):
        raise SystemExit(f"Dynamic ONNX Runtime output changed the reference embeddings: {dynamic_cosines}")

    tokenizer_path = args.adapter / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer asset not found: {tokenizer_path}")
    tokenizer_output = args.output.parent / "tokenizer.json"
    tokenizer_output.write_bytes(tokenizer_path.read_bytes())
    packaged_model_bytes = args.output.stat().st_size + tokenizer_output.stat().st_size
    if packaged_model_bytes > args.max_bytes:
        raise SystemExit(
            f"Model and tokenizer total {packaged_model_bytes} bytes; limit is {args.max_bytes}"
        )
    manifest = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_model_repo": hub_publish["model_repo"],
        "source_model_revision": hub_publish.get("adapter_revision", hub_publish["model_revision"]),
        "adapter": str(args.adapter),
        "adapter_checkpoint": training.get("development_selection", {}).get("selected", {}).get("checkpoint"),
        "sequence_length": args.sequence_length,
        "pooling": "attention-mask mean then L2 normalize",
        "runtime": "ONNX Runtime Android CPU",
        "precision": "int8_rowwise_token_embedding_float32_transformer" if args.precision == "int8_embedding" else "float32",
        "embedding_quantization": "symmetric per-row int8; float32 row scale; max-absolute-value / 127",
        "probe_cosines_before_conversion": cosine.tolist(),
        "probe_cosines_after_precision_change": precision_cosine.tolist(),
        "probe_cosines_after_onnxruntime": conversion_cosine.tolist(),
        "dynamic_shape_probe_cosines": dynamic_cosines,
        "onnx_sha256": sha256(args.output),
        "onnx_size_bytes": args.output.stat().st_size,
        "tokenizer_sha256": sha256(tokenizer_output),
        "tokenizer_size_bytes": tokenizer_output.stat().st_size,
        "packaged_model_bytes": packaged_model_bytes,
        "onnx_opset": 18,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
