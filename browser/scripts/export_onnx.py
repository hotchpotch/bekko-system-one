"""Export a local, full-weight ModernBERT shared-prefix checkpoint to portable FP32 ONNX.

Run from the repository root:
uv run --with onnx --with onnxscript python browser/scripts/export_onnx.py CHECKPOINT
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from bekko_system_one.modules import DecisionHeads, SharedPrefix
from bekko_system_one.prefix import PrefixEncoder
from bekko_system_one.query_budget import QueryParts

TASKS = ["choice", "noul", "score"]
KEYS = ["prefix_ids", "prefix_mask", "doc_ids", "doc_mask", "owners"]


class PortableModel(nn.Module):
    def __init__(self, encoder, heads):
        super().__init__()
        if heads.choice_interaction is not None:
            raise ValueError("The browser exporter does not support Choice interaction heads")
        self.encoder = PrefixEncoder(encoder.encoder.backbone, backend="sdpa")
        self.heads = heads

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
        hidden = self.encoder(prefix_ids, prefix_mask, doc_ids, doc_mask, owners)
        pooled = SharedPrefix.pool(hidden, doc_mask)
        return torch.cat([self.heads.heads[t](pooled) for t in TASKS], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("browser/public/model"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = args.checkpoint
    settings = json.loads((source / "0_SharedPrefix/shared_prefix.json").read_text())
    if settings["lora"] or settings["task_tokens"] or settings["query_truncation"] != "balanced":
        raise ValueError(
            "This exporter supports full-weight, balanced-query checkpoints without task tokens"
        )
    enc = SharedPrefix.load(
        str(source / "0_SharedPrefix"),
        model_kwargs={
            "attention_backend": "sdpa",
            "frozen_linear_bf16": False,
            "fused_rotary": False,
        },
    ).eval()
    heads = DecisionHeads.load(str(source / "1_DecisionHeads")).eval()
    model = PortableModel(enc, heads).eval()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        dict(
            task="noul",
            instruction="Is the customer asking for a refund?",
            state="Please return my money. The item arrived broken.",
            candidates=[dict(id="true", text="Yes"), dict(id="false", text="No")],
        ),
        dict(
            task="choice",
            instruction="Choose the best category.",
            state="明日の東京の天気を教えてください。",
            candidates=[
                dict(id="weather", text="Weather forecast"),
                dict(id="sports", text="Sports news"),
                dict(id="food", text="Recipes and cooking"),
            ],
        ),
        dict(
            task="score",
            instruction="Rate the sentiment from 1 (negative) to 5 (positive).",
            state="I love it! It works perfectly.",
            layout="state_instruction",
            candidates=[dict(id=str(i), text=str(i), value=i) for i in range(1, 6)],
        ),
        dict(
            task="choice",
            instruction="Select the matching answer.",
            state="",
            system="Be precise.",
            candidates=[dict(id="a", text=""), dict(id="b", text="東京 🐢 café\nsecond line")],
        ),
    ]
    cases.append(
        dict(
            task="choice",
            instruction="Choose the relevant passage.",
            state="weather forecast " * 90,
            candidates=[
                dict(id="long", text="rain tomorrow " * 75),
                dict(id="short", text="sports"),
            ],
        )
    )
    fixtures = []
    with torch.inference_mode():
        for request in cases:
            parts = QueryParts(
                request["instruction"],
                request["state"],
                request.get("system", ""),
                request.get("layout", "instruction_state"),
            )
            qids, dids = enc.tokenize_branches(
                [parts.render()], [c["text"] for c in request["candidates"]], query_parts=[parts]
            )
            features = enc.collate_tokens(qids, dids, [0] * len(dids))
            logits = model(*(features[k] for k in KEYS))
            # Independently compare to the original packed trainer implementation.
            original = enc(dict(features))["sentence_embedding"]
            expected = torch.cat([heads.heads[t](original) for t in TASKS], dim=1)
            torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
            fixtures.append(
                dict(request=request, prefix_ids=qids, doc_ids=dids, logits=logits.tolist())
            )
        torch.onnx.export(
            model,
            tuple(features[k] for k in KEYS),
            str(out / "model.onnx"),
            input_names=KEYS,
            output_names=["logits"],
            opset_version=18,
            dynamo=False,
            external_data=False,
            dynamic_axes={
                "prefix_ids": {0: "queries", 1: "prefix_length"},
                "prefix_mask": {0: "queries", 1: "prefix_length"},
                "doc_ids": {0: "candidates", 1: "document_length"},
                "doc_mask": {0: "candidates", 1: "document_length"},
                "owners": {0: "candidates"},
                "logits": {0: "candidates"},
            },
        )
    onnx.checker.check_model(str(out / "model.onnx"))
    tokenization_cases = []
    for layout in ["instruction_state", "state_instruction"]:
        request = dict(
            task="choice",
            instruction="instruction " * 3000,
            state="context " * 5000,
            system="Follow the instruction.",
            layout=layout,
            candidates=[dict(id="long", text="document " * 3000), dict(id="short", text="short")],
        )
        parts = QueryParts(request["instruction"], request["state"], request["system"], layout)
        qids, dids = enc.tokenize_branches(
            [parts.render()], [c["text"] for c in request["candidates"]], query_parts=[parts]
        )
        tokenization_cases.append(dict(request=request, prefix_ids=qids, doc_ids=dids))
    (out / "tokenization-parity.json").write_text(json.dumps(tokenization_cases))
    # Copy only tokenizer data, not training paths or checkpoint metadata.
    (out / "tokenizer.json").write_text(
        (source / "0_SharedPrefix/tokenizer/tokenizer.json").read_text()
    )
    tokenizer_config = {
        key: getattr(enc.tokenizer, key)
        for key in ["cls_token", "sep_token", "pad_token", "unk_token", "mask_token"]
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    manifest = dict(
        format_version=1,
        tasks=TASKS,
        query_length=enc.query_length,
        document_length=enc.document_length,
        cls_token_id=enc.tokenizer.cls_token_id,
        sep_token_id=enc.tokenizer.sep_token_id,
        pad_token_id=enc.tokenizer.pad_token_id,
        model_sha256=hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
        heads_sha256=hashlib.sha256(
            (source / "1_DecisionHeads/model.safetensors").read_bytes()
        ).hexdigest(),
        weights_sha256=hashlib.sha256(
            (source / "0_SharedPrefix/model.safetensors").read_bytes()
        ).hexdigest(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "parity.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Exported {out / 'model.onnx'}: {(out / 'model.onnx').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
