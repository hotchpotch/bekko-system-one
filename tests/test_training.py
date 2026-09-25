import json
import subprocess
import sys

from sentence_transformers import SentenceTransformer
from transformers import ModernBertModel

from bekko_system_one import Group, predict


def test_cli_training_and_st_reload(tmp_path, tiny_encoder):
    base = tmp_path / "base"
    ModernBertModel(tiny_encoder.encoder.backbone.config).save_pretrained(base)
    tiny_encoder.tokenizer.save_pretrained(base)
    source = tmp_path / "train.jsonl"
    rows = [
        dict(query="query", candidates=["good", "bad"], task=t, target=[0.8, 0.2])
        for t in ["choice", "noul", "score"]
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    output = tmp_path / "run"
    config = dict(
        model=dict(
            model_name_or_path=str(base),
            tasks=["choice", "noul", "score"],
            query_length=32,
            document_length=32,
            attention_backend="sdpa",
            frozen_linear_bf16=False,
            fused_rotary=False,
            lora=dict(rank=2, alpha=4),
        ),
        data=dict(
            sampling="source_passes",
            sources={"demo": dict(path=str(source), passes=1)},
            validation_sources={"demo": dict(path=str(source))},
        ),
        training=dict(
            output_dir=str(output),
            device="cpu",
            batch_size=2,
            initial_tokens=64,
            learning_rate=0.0002,
            head_learning_rate=0.0002,
            eval_steps=1,
        ),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    import os

    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    result = subprocess.run(
        [sys.executable, "-m", "bekko_system_one.training", "--config", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    history = [json.loads(r) for r in (output / "history.jsonl").read_text().splitlines()]
    assert sorted(r["groups"] for r in history) == [1, 2]
    model = SentenceTransformer(
        str(output / "model"), trust_remote_code=True, local_files_only=True, device="cpu"
    )
    assert len(predict(model, [Group.from_dict(r) for r in rows])) == 3
    assert (output / "validation.json").is_file()
