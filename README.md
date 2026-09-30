# Bekko System One

> Small models for choosing, judging, scoring, and reranking.

[Models](#model-catalog) · [Quickstart](#quickstart) ·
[Browser app](browser/README.md) · [Training guide](docs/quickstart-training.md) ·
[Bekko dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)

Bekko System One turns an instruction, context, and a set of candidates into a
structured decision. Use it to route a support request, check whether evidence
supports a statement, rate an answer on your own scale, or rank retrieved
passages. It provides training and inference with Sentence Transformers, plus
standalone Python and ONNX exports for deployment.

## Highlights

- **Explicit decisions:** Choice returns a candidate ID, Noul returns a yes/no
  probability, and Score returns a value on a scale you define.
- **Shared-prefix attention:** candidates reuse the instruction and context
  computation while remaining independent in the encoder.
- **Three model sizes:** the v0 family spans 17M, 68M, and 400M models, with Python
  inference on CPU or CUDA and ONNX inference in Node.js or the browser.
- **Train on your data:** use the public Bekko release dataset or supply your own
  examples, with full fine-tuning, LoRA, and soft targets.
- **Reproducible runs:** training records resolved model/data revisions, sampling
  settings, configuration, and evaluation artifacts.

## Tasks at a glance

| Task | Example | Result |
| --- | --- | --- |
| Choice | Which department should handle this request? | Selected candidate ID and option probabilities |
| Noul (yes/no) | Does the customer ask for a refund? | Probability of the authored positive answer |
| Score | How urgent is this request on a 0–4 scale? | Expected numeric value and level probabilities |
| Reranking | Which passages are relevant to this query? | Candidate scores or an ordered list of document IDs |

See [Concepts](docs/concepts.md) for task semantics and the attention structure.
This toolkit is experimental; see [Evaluation and limitations](docs/evaluation.md)
for how to assess model quality and performance.

## Model catalog

| Model | Hugging Face repository | Inference formats |
| --- | --- | --- |
| 17M | [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | Standalone Python, ONNX |
| 68M | [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | Standalone Python, ONNX |
| 400M | [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | Standalone Python, ONNX |

The examples use 17M, the smallest release. Review each model card for its
license, evaluation results, and limitations. Model size alone does not establish
accuracy for your task. See [Compatibility](docs/compatibility.md) for runtime
requirements and supported export features.

## Quickstart

### Browser app

The [browser app](browser/README.md) provides editable Choice, Noul, and Score
examples with CPU and WebGPU execution. Inference runs on your device; input
text is not sent to an inference server. The guide covers local setup, model
downloads, and static hosting.

### Python setup

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

### Run a prediction

Run this example from the repository root. It downloads the 17M model and
processes a customer request through all three decision types on CPU, without
FlashAttention. The [input file](examples/v0-input.json) defines the message,
routing options, yes/no meanings, and urgency levels.

```sh
uv run --locked python - <<'PY'
import json
from pathlib import Path
from bekko_system_one import BekkoSentenceTransformer

model = BekkoSentenceTransformer(
    "hotchpotch/bekko-system-one-v0-17m",
    device="cpu",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
request = json.loads(Path("examples/v0-input.json").read_text())
result = model.predict(request, show_progress_bar=False)
print(json.dumps(result, indent=2))
PY
```

The input message asks for a duplicate payment to be refunded. The result is
keyed by decision ID, with each decision answering a separate question:

| Output field | Meaning |
| --- | --- |
| `result["department"]["selected_id"]` | Which supplied department the model selected |
| `result["refund"]["probability_yes"]` | Probability assigned to the defined refund condition |
| `result["urgency"]["score"]` | Expected urgency on the supplied 0–4 scale |

Each decision also returns probabilities keyed by candidate ID. Actual values
depend on the checkpoint; probabilities do not establish calibrated confidence.

Public release models need no Hub login. The first load needs network access;
later loads can reuse the local cache. `trust_remote_code=True` executes model
repository code; use a trusted release and pin `revision` to its commit for
reproducible deployments.

### Batch and tune inference

Pass a list to process multiple requests and receive results in the same order:

```python
results = model.predict(
    [request, request],
    batch_size=128,
    token_budget=64000,
    show_progress_bar=False,
)
```

`batch_size` bounds the processing window; `token_budget` controls the estimated
work per microbatch, not a strict memory limit. Set `device="cuda"` when loading
for GPU execution. See [standalone inference](docs/inference-v0.md) for adaptive
input limits, optional FA2, and compilation.

## Training

### Start with Bekko dataset

[Bekko dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)
is public at release. Start with the
[training quickstart](docs/quickstart-training.md) to run the 17M smoke recipe
with public training/validation data and local logging. It includes checkpoint
export and prediction, without requiring the separate S1MB test dataset or a
W&B account.

The [full v0 recipes](docs/training-v0.md) cover 17M, 68M, and 400M models,
S1MB evaluation, and online tracking. A smoke run verifies the pipeline;
it does not establish model quality.

### Use your own data

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

For inference with a training checkpoint, use `rank()` for raw document logits,
`predict()` for candidate probabilities, or `predict_typed()` for interpreted
results. The [package inference guide](docs/inference.md) includes loading and
prediction examples. To deploy independently of the training package,
[export a standalone model](docs/inference-v0.md) or a supported
[ONNX checkpoint](browser/README.md#export-a-checkpoint).

## Evaluation

Smoke runs check training, checkpoint reload, and export. They do not establish
release-model quality. Use the model cards for revision-specific results and
[Evaluation and limitations](docs/evaluation.md) for metric interpretation and
reproducible comparisons. Distributed training and optimizer-state resume are
not implemented.

## Documentation

- [Concepts](docs/concepts.md): task semantics and shared-prefix attention.
- [Compatibility](docs/compatibility.md): environments, features, and export limits.
- [Training quickstart](docs/quickstart-training.md): public Bekko data to prediction.
- [Evaluation and limitations](docs/evaluation.md): interpreting and reporting results.
- [Troubleshooting](docs/troubleshooting.md): setup, data, memory, and export errors.
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

## Acknowledgments

Bekko builds on [Sentence Transformers](https://github.com/huggingface/sentence-transformers)
and [Transformers](https://github.com/huggingface/transformers). The v0 training
recipes initialize from the Ettin reranker family; model sources and revisions
are recorded in the [release configurations](docs/training-v0.md#recipes).
Browser inference uses [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Author and discussion

Created by [Yuichi Tateno (@hotchpotch)](https://github.com/hotchpotch).
Questions and discussion are welcome in
[GitHub Discussions](https://github.com/hotchpotch/bekko-system-one/discussions).
See [Contributing](CONTRIBUTING.md) for the current participation policy.
