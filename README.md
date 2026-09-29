# Bekko System One

Train and run small models for typed decisions and document reranking with
Sentence Transformers. A shared-prefix encoder processes the query once and
scores its candidates with task-specific heads. Training supports full encoder
fine-tuning or LoRA, soft targets, packed tokens, and adaptive microbatches.

| Task | Result |
| --- | --- |
| Choice | A selected candidate and a probability distribution over options |
| Noul (yes/no) | The probability of the authored positive answer |
| Score | An expected numeric value over explicitly defined score levels |
| Reranking | Scores used to order candidate documents |

This is an experimental training and inference toolkit. Smoke runs check that
training and export work; they do not establish model quality. Distributed
training and optimizer-state resume are not implemented.

## Getting started

Clone the repository and run commands from its root:

```sh
git clone https://github.com/hotchpotch/bekko-system-one.git
cd bekko-system-one
uv sync --locked
uv run --locked bekko-system-one --help
```

The training environment targets **Linux x86_64 and Python 3.12**. The lockfile
pins PyTorch 2.10.0 (CUDA 13.0), Transformers 5.17.0, and Sentence Transformers
6.1.0. CPU tests and SDPA inference do not need the external FlashAttention
package. For GPU training with the provided configurations, install its matching
wheel:

```sh
uv sync --locked --extra fa2
```

A compatible NVIDIA driver and CUDA GPU are required for those configurations.
The `fa2` wheel is specific to this Python/PyTorch/CUDA stack.

## Train with your own data

Start with [the reranker configuration](configs/reranker.yaml) or
[the typed-decision configuration](configs/typed-decisions.yaml). Copy the
configuration, set its data paths, and select a new output directory. The files
under `data/` are placeholders; training datasets are not bundled.

Each JSONL row describes one complete candidate group:

```json
{"query":"Find marine animals.","candidates":["Dolphins live in the ocean.","Bread is baked."],"task":"reranker","target":[1.0,0.0]}
```

`target` is a probability distribution in candidate order. Keep targets out of
query and candidate text. Arrow datasets and structured typed-decision releases
are also supported; see the [training reference](docs/training.md).

After editing the configuration:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config configs/reranker.yaml --max-steps 2 --output-dir output/first-run
```

This checks the training path with your data. Remove `--max-steps 2` for the
configured training budget. Runs require a new output directory and save a
Sentence Transformers checkpoint under `model/`, alongside configuration and
metrics. The two templates train separate models with their own weights.

The [v0 release recipes](docs/training-v0.md) provide 17M, 68M, and 400M settings.
They require access to the configured Hub datasets and enable online W&B tracking;
see that guide before running them. Your own JSONL data does not require access
to those datasets or a W&B account.

## Run inference

Load a trained reranker with the package installed:

```python
from sentence_transformers import SentenceTransformer
from bekko_system_one import rank

model = SentenceTransformer(
    "output/first-run/model",
    device="cpu",
    trust_remote_code=True,
    local_files_only=True,
    model_kwargs={
        "attention_backend": "sdpa",
        "frozen_linear_bf16": False,
        "fused_rotary": False,
    },
)
scores = rank(model, "Find marine animals.", ["Dolphins live in the ocean.", "Bread is baked."])
print(scores.argsort(descending=True))
```

`rank()` returns raw logits in document order. For typed decision checkpoints,
`predict()` returns per-decision probabilities and `predict_typed()` interprets
candidate IDs and numeric score levels. See the
[package inference reference](docs/inference.md) for rendering, batching, and
optional CUDA optimizations.

For deployment, [export a standalone checkpoint](docs/inference-v0.md) with the
`BekkoSentenceTransformer.predict()` interface and optional `torch.compile`.
Its SDPA runtime does not require this training package, PEFT, datasets, W&B, or
the external FlashAttention package. Use [the example input](examples/v0-input.json)
as a starting point.

The [browser app](browser/README.md) runs ONNX models locally using CPU or WebGPU.
It has its own Node.js setup, checkpoint export instructions, and supported-model
constraints.

## Documentation

- [Training reference](docs/training.md): data formats, sampling, truncation, and heads.
- [v0 training recipes](docs/training-v0.md): model sizes, budgets, tracking, and artifacts.
- [Package inference](docs/inference.md): checkpoint loading, typed results, and performance.
- [Standalone inference](docs/inference-v0.md): export, Python API, and CLI.
- [Browser inference](browser/README.md): React UI, Node API, and ONNX export.
- [Development](docs/development.md): development setup and validation.

## Development

```sh
uv sync --locked
uv run --locked tox
```

The default checks run CPU Python tests, Ruff, and ty. CPU tests use tiny local models
and need no downloaded weights. CUDA tests require the optional stack; see
[Development](docs/development.md) for explicit CPU and GPU commands.

## License

The code in this repository is licensed under the [MIT License](LICENSE).
Model weights, datasets, and third-party dependencies retain their own licenses
and access conditions; the code license does not grant rights to those assets.
