# Package inference reference

This guide covers training checkpoints loaded with the installed package. For
exported checkpoints and the typed `BekkoSentenceTransformer` interface, see
[standalone v0 inference](inference-v0.md). Run commands from the repository root.

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

## Typed predictions and numerical evaluation

`predict()` continues to return probability tensors. `predict_typed()` interprets
judgment groups using explicit candidate metadata and returns dataclasses:

- `ChoicePrediction`: `selected_id`, `probabilities` keyed by candidate ID.
- `NoulPrediction`: `probability_yes`, `probabilities`. The positive candidate is
  identified by `true` or `yes`, regardless of its position.
- `ScorePrediction`: `score = sum(probability * value)`, `normalized_score`,
  `probabilities` and `values` keyed by ID. Normalization is
  `(score - min(values)) / (max(values) - min(values))`.

```python
from bekko_system_one import predict_typed, render_input_group

# Native input contains state_json and decisions, without labels or provenance.
groups = [render_input_group(case_input, i) for i in range(len(case_input["decisions"]))]
answers = predict_typed(model, groups)
# InferenceEngine(model).predict_typed(groups) exposes the same interface.
```

For Score values 0, 1, 2, 3, 4 and probabilities 0, 0, .12, .60, .28, the score is
3.16 and its normalized value is .79. Explicit values, not presentation positions,
determine the result. Different scales and shuffled candidate order are supported.
Score and Noul are independent judgments; no equality constraint ties their outputs.
Existing checkpoints work without retraining. Typed prediction requires the relevant
metadata and rejects document-ranking groups; use `predict()` or `rank()` for those.

`Group.metadata` is an optional `DecisionMetadata` containing `candidate_ids`,
`candidate_values`, `kind` (`judgment` or `ranking`), and optional `case_id`, `group_id`,
`decision_id`. The release renderer supplies these fields. JSONL groups may supply
an equivalent `metadata` object. The metadata survives token preparation and packing
but is never concatenated into model inputs. Existing groups without metadata remain
valid for training, probability inference and distribution evaluation.

Evaluation retains `cross_entropy`, `accuracy`, `brier` and
`target_mass_at_prediction`, and adds:

- `score_mae`: absolute difference of predicted and target expectations, in the
  original units. Both use the same candidate values.
- `score_normalized_mae`: that difference divided by the scale width, for comparisons
  across different scales. Ranking and groups without numeric scales are excluded.
- `binary_brier`: squared error of Noul's yes probability against its soft target.
  The existing two-candidate distribution `brier` remains twice this value.
- `kl_divergence`: KL(target || prediction), separating target entropy from CE.

Each result includes `metric_counts`. A metric with no eligible decisions is `null`,
not zero. Macro summaries average only eligible datasets for each metric and expose
both `metric_counts` and `metric_datasets`; dataset size does not change macro weight.
Raw-unit `score_mae` can combine different scales, so prefer `score_normalized_mae`
for such summaries. Soft-target errors measure agreement with that teacher; they do
not establish calibration against real-world frequencies.

Training history includes `usage_by_task` per source and step. `training_usage.json`
aggregates decisions, candidate counts, post-truncation query/candidate tokens and
exact unique case counts by source and task; document ranking is counted separately.
Case IDs are source-local. `identified_decisions` and `case_identity_complete` report
coverage; unknown unique counts are `null`. Partial counts cover only known IDs.
Unique case tracking uses memory proportional to the number of observed case IDs.
Token counts count accepted decisions once, exclude padding and OOM retries, and are
not hardware-operation counts. `attention_work_tokens` repeats the query length per
candidate; `query_tokens` counts it once per decision, without shared-prefix deduplication.

Training caps and `epoch_fraction` are in **decisions**, while validation sampling is
in **cases** and retains every decision in each selected case. Adding another task
can change source weights, update counts, and unique-case coverage. Use these logs
to compare actual work before changing sampling ratios or loss weights.
