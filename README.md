# Bekko System One

> Small System One decision models for choosing, judging, scoring, and reranking.

[Release article](https://huggingface.co/blog/hotchpotch/bekko-system-one-v0-release/) ·
[Models](#model-catalog) · [Quickstart](#quickstart) · [Benchmarks](#benchmarks) ·
[Browser app](browser/README.md) · [Training guide](docs/quickstart-training.md) ·
[Training dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)

Bekko System One is an **experimental project for training and running very
small System One decision models**, in the same category as
[TypeSafe AI's Jev](https://docs.typesafe.ai/concepts/system-one). The v0 family
spans 17M to 400M parameters. These models turn instructions, context, and
candidates into typed decisions and probabilities that software can use directly.

**Version 0 has limited generalization, but can still perform very well on some
tasks.** Its strongest overall benchmark results should be read alongside its
weaker results on instruction- and context-adaptation tasks. Bekko's training
set also includes dataset families represented in the evaluation; strong results
on those tasks do not establish broad generalization. The
[benchmark comparison](#benchmarks) below shows both views.

Use it to route a support request, check whether evidence
supports a statement, rate an answer on your own scale, or rank retrieved
passages. It provides training and inference with Sentence Transformers, plus
standalone Python and ONNX exports for deployment.

## ✨ Highlights

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

### Start with the training dataset

The [training dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)
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

## Benchmarks

**Results as of September 30, 2026 (2026-09-30).** For the latest results and
per-benchmark comparisons, see the
[📊 S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard).
The date identifies this snapshot, not the execution date of every evaluation.

[System One Mosaic Benchmark (S1MB)](https://github.com/hotchpotch/S1MB) combines
public NLP tasks, Open-Jev/Laya-derived tasks, and synthetic adaptation tasks.
Its standard evaluation covers **137 benchmarks across 106 subsets**, with
14,009 cases and 26,269 decisions: 59 Noul, 57 Choice, and 21 Score benchmarks.

### Overall results

The tables show **models with at most 500M total parameters, plus Jev 1.13**
as a reference. Jev's parameter count is not disclosed in the snapshot; it is
not included in the size limit. Rows are sorted by Task Avg within each table.
All models below cover all 137 benchmarks in the overall evaluation.

| Model | Total parameters | Task Avg | Noul | Choice | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Jev 1.13](https://docs.typesafe.ai/models) | Not disclosed | 59.59 | 64.63 | 67.22 | 46.92 |
| [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 395M | 50.60 | 51.24 | 61.32 | 39.25 |
| [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 68M | 40.46 | 42.91 | 51.62 | 26.85 |
| [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 17M | 27.57 | 31.43 | 35.57 | 15.70 |
| [von](https://huggingface.co/wfzyx/von) | 395M | 16.21 | 20.15 | 23.99 | 4.48 |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | 15.00 | 20.06 | 18.93 | 5.99 |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | 13.36 | 20.19 | 16.31 | 3.58 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | 9.28 | 14.01 | 13.03 | 0.79 |

**These scores are not raw accuracy.** Each benchmark is scored relative to a
simple baseline, such as always choosing the same option or returning a constant
value. Scores are clipped to 0–100: 0 means at or below the baseline, and 100 is
the reference ceiling. A score of 50 means halfway from baseline to that ceiling,
not 50% correct. Scores are averaged within each task; **Task Avg weights Noul,
Choice, and Score equally**. See the
[scoring definitions](https://github.com/hotchpotch/S1MB/blob/main/evaluator/SCORING.md).

Bekko 400M has the highest overall Task Avg among the models at or below 500M
shown here. However, the training and evaluation manifests share 77 subset names,
indicating exposure to related task families. This is not a count of duplicated
test examples or proof that all rows were used in training. Interpret the overall
result together with the adaptation results below.

### Generalization: instruction and context adaptation

S1MB includes six synthetic benchmarks: **Diverse** and **Contextual** variants
for each of Noul, Choice, and Score, with 100 cases each (600 total). They test
responses to different instructions, contexts, and decision criteria. These six
are already included in the overall 137. The table below reports their scores
separately, using the same model-size selection as above.

| Model | Total parameters | Task Avg | General Noul | General Choice | General Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Jev 1.13](https://docs.typesafe.ai/models) | Not disclosed | 96.27 | 99.00 | 98.68 | 91.14 |
| [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 395M | 54.48 | 52.00 | 80.30 | 31.14 |
| [von](https://huggingface.co/wfzyx/von) | 395M | 43.19 | 41.00 | 65.89 | 22.67 |
| [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 68M | 32.48 | 31.00 | 57.52 | 8.90 |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | 31.82 | 22.00 | 58.80 | 14.67 |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | 26.50 | 9.00 | 54.83 | 15.67 |
| [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 17M | 18.96 | 15.00 | 39.11 | 2.76 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | 14.47 | 6.00 | 36.67 | 0.75 |

The gap is substantial: Bekko 400M's Task Avg trails Jev 1.13 on this adaptation
view (54.48 versus 96.27). General Choice is its strongest adaptation score at
80.30, while General Score reaches only 31.14. Version 0 can be useful for
specific tasks, but its strong overall results do not establish broad
generalization.

The synthetic questions and intended answers were authored and self-reviewed
by GPT-6-Astra, without independent human validation. They measure adaptation
within this test design, not generalization to all unseen tasks or guaranteed
separation from every model's training data.

See the [full 23-model snapshot and methodology](docs/benchmarks.md) for both
tables, training-exposure details, and provenance limits. These are supplied
benchmark results, not measurements from the quickstart smoke run.

## 📚 Documentation

- [Concepts](docs/concepts.md): task semantics and shared-prefix attention.
- [Compatibility](docs/compatibility.md): environments, features, and export limits.
- [Training quickstart](docs/quickstart-training.md): public Bekko data to prediction.
- [Benchmark snapshot](docs/benchmarks.md): September 30, 2026 results and methodology.
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

## 🙏 Acknowledgments

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
