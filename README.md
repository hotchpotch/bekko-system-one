# Bekko System One

Train shared-prefix rerankers and typed decision models with Sentence Transformers.
Each query is encoded once per batch. Candidates attend to its per-layer keys and
values while remaining independent of other candidates. One encoder and LoRA
adapter serve independent scalar scoring heads.

## Install and train

Python 3.12 / Linux x86_64, PyTorch 2.10.0 / CUDA 13.0, Transformers 5.17.0,
and Sentence Transformers 6.1.0 are pinned. The optional `fa2` extra installs the
FlashAttention 2.8.3 wheel for this stack.

```sh
uv sync --locked --extra fa2
# Set source paths and a new output directory in the selected configuration.
CUDA_VISIBLE_DEVICES=0 uv run --extra fa2 bekko-system-one --config configs/reranker.yaml
CUDA_VISIBLE_DEVICES=0 uv run --extra fa2 bekko-system-one --config configs/typed-decisions.yaml
```

These configurations train separate models: a dedicated reranker, or Choice /
Noul / Score with a shared adapter. Their learned LoRA weights are independent.
Use `--max-steps 2` for an integration check. The output directory must be new.
It contains configuration, JSONL history, optional validation metrics, and a
Sentence Transformers checkpoint under `model/`. Distributed training and
optimizer-state resume are not currently implemented.

The model is an ordinary `SentenceTransformer` containing `SharedPrefix` and
`DecisionHeads`. The group-aware training loop calls these modules directly,
using separate LoRA/head learning rates, FP32 loss, clipping, warmup/linear decay,
and adaptive microbatch accumulation. It does not use the default text-pair
collator of `SentenceTransformerTrainer`.

## Data

Each source is a JSONL file or an Arrow `Dataset` directory from
`datasets.Dataset.save_to_disk`. Select explicit training and validation splits;
the trainer does not infer or repartition them.

```json
{"query":"marine animals","candidates":["Dolphins live in the ocean.","Bread is baked."],"task":"reranker","target":[1.0,0.0]}
```

`query` is the complete rendered instruction/state. `candidates` contains rendered
options or documents. `task` is `reranker`, `choice`, `noul`, or `score`. `target`
is a normalized distribution in candidate order; soft targets are preserved.
Targets and metadata never enter the tokenizer. Keep labels and answers out of
the input text, and use the same rendering during training and inference.

For typed decisions, use a query such as
`Instruction: {instruction}\nState: {state_json}` and options such as
`Candidate: true: The statement is true.` Score options represent discrete levels:
the head predicts their distribution, from which an expected score can be computed.

`source_passes` visits each row for every configured pass, including tail batches;
each source completes one shuffled pass before beginning its next pass.
`weighted` samples sources with probability proportional to
`source_size ** sampling_alpha`, shuffling and recycling their rows. Its
`epoch_fraction: 0.5` means a budget equal to half the total group count, not a
fixed 50% subset. Weighted mode uses complete logical batches, excludes sources
smaller than a batch, and drops each shuffled pool's tail before recycling it.
Source choice and row permutations use separate random streams. Configure one
source per dataset to preserve dataset-level weighting. The templates contain
recipes, not bundled training datasets or performance claims.

## Load and predict

Install this package in the inference environment. Checkpoints include backbone
configuration, all encoder/LoRA weights, tokenizer, and heads. Local loading does
not require the original base-model directory or network access. Sentence
Transformers 6 requires `trust_remote_code=True` for custom module classes,
including classes imported from an installed package.

```python
from sentence_transformers import SentenceTransformer
from bekko_system_one import Group, predict, prepare_batch, rank

model = SentenceTransformer("output/reranker/model", device="cuda", trust_remote_code=True)
scores = rank(model, "marine animals", ["Dolphins live in the ocean.", "Bread is baked."])
order = scores.argsort(descending=True)
# Standard ST pair encoding returns raw scores for a single-head model.
scores = model.encode([("marine animals", "Dolphins live in the ocean.")])

typed = SentenceTransformer("output/typed-decisions/model", device="cuda", trust_remote_code=True)
groups = [Group(
    query='Instruction: Classify the sentiment.\nState: {"text":"I loved it."}',
    candidates=["Candidate: positive: Positive sentiment.", "Candidate: negative: Negative sentiment."],
    task="choice",
)]
probabilities = predict(typed, groups)
logits = typed(prepare_batch(typed, groups))["scores"]
```

`rank()` returns raw logits in document order, caching the prefix once across all
candidate chunks. `predict()` normalizes within each decision, preserving option
order. `prepare_batch()` routes mixed task groups. Generic `encode()` is intended
for single-head query/candidate pairs.

For CPU inspection, load with `device="cpu"`, `local_files_only=True`, and
`model_kwargs={"attention_backend": "sdpa", "frozen_linear_bf16": False, "fused_rotary": False}`.

## Performance

Valid tokens stay packed through projections, attention, and feed-forward layers.
FlashAttention 2 uses variable-length, suffix-aligned attention. Frozen linear
weights can stay BF16; adapters, residuals, norms, heads, and loss retain FP32.
Fused RoPE supports forward and first-order backward. Shared tokenization and
adaptive microbatches retain whole candidate groups. Each microbatch loss sum is
divided by the actual logical batch size. An OOM discards partial gradients and
replays the whole batch with restored random state and a smaller token budget.

`bekko_system_one.inference.SharedPrefixInference` provides a separate inference
snapshot with fused gather/RoPE and optional CUDA Graph capture. Construct it with
an evaluated model and CPU feature dictionary; `score(features)` returns logits.
Masks, ownership, and head routing must remain fixed; `cache_prefix=True` also
fixes prefix tokens. Save the original model rather than an inference snapshot.

Different lengths, batch layouts, and BF16 kernels can produce small numerical
differences. Benchmark the intended workload: packaging alone is not an
end-to-end throughput guarantee.

## Development

```sh
uv run tox
CUDA_VISIBLE_DEVICES=0 uv run --extra fa2 pytest -m cuda
```

CPU tests need no FlashAttention or downloaded weights. Tests cover standalone ST
save/load, soft targets, gradient accumulation, prefix cache reuse, source pass
counts, OOM replay, and CUDA forward/backward.
