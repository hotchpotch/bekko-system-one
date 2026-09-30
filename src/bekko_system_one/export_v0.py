"""Export an independently loadable Sentence Transformers inference checkpoint."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .inference_v0 import BekkoInference, BekkoSentenceTransformer
from .modules import DecisionHeads, SharedPrefix


def runtime_source():
    """Copy the standalone source verbatim; no export-only dependency rewriting."""
    source = Path(__file__).with_name("inference_v0.py").read_text()
    module = ast.parse(source)
    if ast.get_docstring(module) is None:
        raise ValueError("Standalone runtime must include its interface guide")
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError("Standalone runtime must not contain relative imports")
    return source


def portable_module(model, *, prefix_layout="instruction_state"):
    if (
        len(model) != 2
        or not isinstance(model[0], SharedPrefix)
        or not isinstance(model[1], DecisionHeads)
    ):
        raise ValueError("Expected a SharedPrefix + DecisionHeads training checkpoint")
    encoder, heads = model[0], model[1]
    backbone = copy.deepcopy(encoder.encoder.backbone).float().cpu()
    if encoder.settings.get("lora"):
        backbone = backbone.merge_and_unload(safe_merge=True)
    settings = {key: encoder.settings[key] for key in ("query_truncation", "task_tokens")}
    runtime = BekkoInference(
        backbone,
        copy.deepcopy(encoder.tokenizer),
        tasks=heads.tasks,
        choice_interaction=heads.choice_interaction_config,
        prefix_layout=prefix_layout,
        **settings,
    )
    runtime.heads.load_state_dict(heads.heads.state_dict(), strict=True)
    if heads.choice_interaction is not None:
        assert runtime.choice_interaction is not None
        runtime.choice_interaction.load_state_dict(
            heads.choice_interaction.state_dict(), strict=True
        )
    return runtime.eval().requires_grad_(False)


def export_model(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Export requires a local training checkpoint directory")
    if output.exists() or checkpoint in output.parents or output in checkpoint.parents:
        raise ValueError("Choose a new output directory separate from the checkpoint")
    source = SentenceTransformer(
        str(checkpoint),
        device="cpu",
        local_files_only=True,
        trust_remote_code=True,
        model_kwargs={
            "attention_backend": "sdpa",
            "frozen_linear_bf16": False,
            "fused_rotary": False,
        },
    ).eval()
    rendering = checkpoint / "release_rendering.json"
    layout = (
        json.loads(rendering.read_text()).get("prefix_layout", "instruction_state")
        if rendering.exists()
        else "instruction_state"
    )
    runtime = portable_module(source, prefix_layout=layout)
    model = SentenceTransformer(modules=[runtime], device="cpu")
    model.save_pretrained(str(output), create_model_card=False)
    (output / "inference_v0.py").write_text(runtime_source())
    modules = json.loads((output / "modules.json").read_text())
    assert len(modules) == 1
    modules[0]["type"] = "inference_v0.BekkoInference"
    (output / "modules.json").write_text(json.dumps(modules, indent=2) + "\n")
    (output / "requirements.txt").write_text(
        "torch>=2.10,<2.11\ntransformers==5.17.0\nsentence-transformers==6.1.0\nsafetensors>=0.7\ntqdm>=4.67\n"
    )
    (output / "README.md").write_text(
        "# Bekko System One v0 inference\n\n"
        "Install `requirements.txt`; no Bekko training package is required.\n\n"
        '`BekkoSentenceTransformer` defaults to `attn_implementation="auto"`: '
        "compatible CUDA + flash-attn selects FA2, otherwise SDPA. Explicit "
        "`flash_attention_2` fails at loading if unavailable; `sdpa` needs no FA2 wheel. "
        "Inspect `model[0].attn_implementation`. For 17M, speeds are generally similar; "
        "for 68M and larger, prefer FA2 for throughput and benchmark your workload. "
        "BF16 probabilities can differ between backends.\n\n"
        "```python\nfrom inference_v0 import BekkoSentenceTransformer\n\n"
        'model = BekkoSentenceTransformer("PATH_OR_HUB_MODEL_ID", device="cpu", trust_remote_code=True)\n'
        "# request contains native state_json and decisions, without targets.\n"
        "result = model.predict(request)\n"
        "# Lists use bounded, length-bucketed batches with progress enabled.\n"
        "results = model.predict([request], batch_size=128, token_budget=64000)\n"
        "# Optional: compile tensor execution; the first calls include compilation.\n"
        "model.compile_inference()\nresult = model.predict(request)\n```\n\n"
        'For CUDA, select `device="cuda"`; attention defaults to automatic FA2/SDPA selection. '
        "PEFT, datasets and W&B are not required; FlashAttention is optional for SDPA. "
        "Tokenization/rendering remain eager. Compile cache reuse depends on shapes, device and runtime. "
        "Score is the expectation over explicit numeric criterion values; Noul returns P(yes). "
        "Adaptive inference shares the backbone position budget between query and candidates, "
        "reserving half for each and lending unused capacity; candidates receive the odd token. "
        "Candidates default to a 3800-token cap including special tokens. "
        "Override context_length, query_length or document_length per prediction call. "
        "Overlong queries use the saved query truncation policy; candidates truncate on the right. "
        "Use `predict()` for typed decisions, not `encode()` on raw strings.\n"
    )
    # Ensure the code advertised by modules.json really resolves through ST remote-code loading.
    restored = BekkoSentenceTransformer(
        str(output), device="cpu", local_files_only=True, trust_remote_code=True
    )
    if type(restored[0]).__module__.startswith("bekko_system_one"):
        raise RuntimeError("Portable export unexpectedly loaded the installed training package")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_model(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
