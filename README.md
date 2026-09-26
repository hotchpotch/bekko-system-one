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

`predict()` and `rank()` default to `inference="optimized"`: shared prefixes,
packed tokens and cached inference snapshots, without custom Triton kernels or
CUDA Graph capture. On CUDA with FlashAttention 2, encoder linear weights are
precast to BF16; decision heads remain FP32. Training's fused RoPE is disabled
in these snapshots as well. The source model and its training mode are preserved.

`predict()` tokenizes once and batches whole questions with `token_budget=16000`,
preserving question/candidate order. An oversized question runs alone. `rank()`
computes the prefix once across all candidate chunks in a call.

Enable Triton fusion explicitly through `InferenceEngine`:

```python
from bekko_system_one import InferenceEngine

engine = InferenceEngine(typed)
probabilities = engine.predict(groups)  # no custom Triton JIT
engine.prepare_fast_inference()  # no example groups required
probabilities = engine.predict(groups)  # kernels compile lazily as needed

# Optional: move warmup work ahead of serving real requests.
engine.prepare_fast_inference(example_groups=representative_groups)

legacy = InferenceEngine(typed, inference="legacy")
probabilities = legacy.predict(groups)
```

`prepare_fast_inference()` enables fused RoPE and K/V gather on CUDA/FA2 and
returns the engine. Without examples it creates/reuses the snapshot, executing
no kernels; actual inputs drive compilation automatically. Optional examples
warm up prediction twice. Empty examples are valid. Unseen lengths, candidate
counts or task routing need no new user preparation; additional JIT work may
occur on a cache miss. CPU and non-FA2 models keep the portable, unfused forward.
This is a package-specific preparation API, not PyTorch `compile()`.

In fast mode, `predict()` captures a CUDA Graph on the second occurrence of an
exact layout. The cache keeps at most two layouts sharing one snapshot; other
layouts use fused eager inference. Masks, ownership, head routing and stream
must match for replay. Example warmup only retains the layouts that fit this
cache, not every possible future input. `rank()` uses fused eager inference with
per-call prefix reuse. Snapshot creation, JIT and graph capture add startup cost
and memory; repeated calls reuse their results.

The functional API also accepts `inference="fast"` for lazy fusion, or
`inference="legacy"` for the previous path. Engines and functional calls share
one runtime cache per source model; changing mode replaces that cached runtime.

`inference="legacy"` preserves the previous inference implementation; for
`predict()` it uses a single batch and ignores `token_budget` for batching.
Ordinary SentenceTransformer `encode()` and direct `model(features)` calls retain
their existing behavior. Training evaluation also retains its existing forward.

Normal optimizer updates, `load_state_dict()`, parameter replacement and device
or dtype changes invalidate the cached snapshot on the next optimized call.
After changing adapter selection or other runtime configuration, or mutating
weights through `.data`, call `clear_inference_cache(model)` (exported at package
level). It also releases the cached snapshot and graphs. Do not modify/train the
source model concurrently with inference. Save the original model; runtime caches
are external to the module and are not part of its checkpoint.

For explicit fixed-layout sessions,
`bekko_system_one.inference.SharedPrefixInference` remains available. Construct
it with an evaluated model and CPU features; `score(features)` returns logits.
`cache_prefix=True` also fixes prefix tokens and reuses their K/V across calls.

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

### Task markers after the shared prefix

Optionally add a learned special token at the start of each candidate branch:

```yaml
model:
  task_tokens:
    choice: "[CHOICE]"
    noul: "[NOUL]"
    score: "[SCORE]"
```

Use `Group(..., task="choice")` (or `noul` / `score`) with the usual prediction
and training APIs. The query remains shared across tasks. A candidate is encoded
as `[TASK] candidate [SEP]`; the marker counts toward `document_length`, and
mean pooling includes it. Tasks absent from the mapping retain their original
input format. Pair-only `encode()` inputs use the `reranker` task; use `Group`
inputs for typed decisions.

New marker embeddings start from the existing SEP embedding. Full finetuning
updates them with the encoder; LoRA trains only their selected embedding rows
alongside the adapters and heads. Token IDs, vocabulary, and weights persist in
the saved model. For raw document scores through a typed head, use
`rank(model, query, documents, task="score")` or the same `task` argument on
`InferenceEngine.rank()`. This also applies the matching task marker.
