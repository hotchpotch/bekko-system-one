# Bekko System One

Train shared-prefix rerankers and typed decision models with Sentence Transformers.
Each query is encoded once per batch. Candidates attend to its per-layer keys and
values while remaining independent of other candidates inside the encoder. One encoder
and optional LoRA adapter serve scalar scoring heads. Choice can additionally use a
small comparison head over the candidates of each decision.

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

## Bekko System One v0 release recipes

[Train v0 from this repository](docs/training-v0.md) using the private Bekko and
S1MB Hub datasets and public Ettin reranker initialization. Configurations for
17M, 68M and 400M include full 300k-cap training, a weighted 10% budget
(alpha=0.5), and a small smoke run. All use cosine scheduling and online W&B
tracking. Latest Hub revisions are resolved and recorded when a run starts.

```sh
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config configs/bekko-system-one-v0-17m-smoke.yaml
```

Authenticate with Hugging Face and W&B first; see the recipe guide for setup,
budget semantics, evaluation scope, larger models and output artifacts.

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

## Training from typed-decision releases

The training CLI also reads a release directory containing `training-manifest.json`
and saved Hugging Face datasets at `<dataset>/<split>`. Each manifest entry contains
`dataset` and `split`; `train` entries are used for training, and `evaluation`
entries are filtered by the requested validation or test role. Test-only datasets
are never used as validation. Related source variants remain separate sampling
sources; choose the release and its mixture deliberately.

```yaml
seed: 42
model:
  checkpoint: /path/to/saved-bekko-model
  query_length: 2048
  document_length: 1024
data:
  root: /path/to/training-release
  sampling: weighted
  sampling_alpha: 0.5
  epoch_fraction: 0.01
evaluation:
  validation_samples: 5
  validation_sampling: balanced
  test_root: /path/to/test-release
  token_budget: 16000
  before_training: true
training:
  output_dir: /path/to/new-run
  device: cuda
  batch_size: 512
  learning_rate: 0.0001
  head_learning_rate: 0.0005
  eval_steps: 100
```

Run `CUDA_VISIBLE_DEVICES=<gpu> bekko-system-one --config config.yaml`.
`model.checkpoint` loads encoder and task-head weights with a fresh optimizer and
schedule. It is continued fine-tuning, not optimizer-state resumption. Length and
runtime overrides are saved in the resulting model. Alternatively, use the existing
`model_name_or_path` model construction settings to start from a backbone.

Release rows contain `state_json`, `decisions`, and aligned `decision_prompts`.
Training indexes individual decisions, preserves soft labels by option ID, and
renders only instructions, state and candidate descriptions. Document ranking
uses `input_format: reranking`, the Score head, and document-only candidate text.
Pointwise ratings keep their document in the state, so choose a query length that
accommodates it. Input limits apply equally to initial and final evaluation.

Structured rows may omit the constant `schema_version` column and empty
`legacy_aux_json` / `provenance_json` columns. If a version is present, it must
be `system_one.v1`. Input and target validation remains the same.

### Loading a Hugging Face dataset

For typed decision rows hosted on the Hub, `data.root` and `evaluation.test_root`
also accept a mapping. These sources use the standard `datasets.load_dataset()`
API and its download/cache mechanism; an intermediate `save_to_disk()` export
is unnecessary.

```yaml
data:
  root:
    repo_id: organization/typed-decisions
    revision: main             # A commit SHA is recommended for reproducibility.
    configs: [subset_a, subset_b]  # Omit to include all configurations.
    token: true                # Use credentials from `hf auth login` or HF_TOKEN.
  sampling: uniform
evaluation:
  test_root:
    repo_id: organization/typed-decisions
    revision: main
    configs: [subset_a, subset_b]
    token: true
  validation_samples: 5
```

Hub sources must contain the same typed rows as local releases. Configurations
become dataset names for sampling and metrics. Split names select their roles:
`train`/`train_*`, `validation`/`validation_*`, and `test`/`test_*`.
Calibration, OOD and unlabeled splits are not used for those roles. Missing
validation remains empty; it never falls back to test. No custom manifest is
required on the Hub. `configs` restricts which configurations are considered;
the existing exclusion and per-dataset sampling settings still apply.

The trainer resolves each repository/revision to a commit before loading data
and records the resolved revision, configurations, split sizes and fingerprints
in `data_manifest.json`. Train and evaluation references to the same revision
share the pinned commit. Authentication or download failures are surfaced
without a local fallback. `token` accepts only a boolean, so credentials are
never embedded in saved configuration files; omit it for the Hub client's
default authentication behavior. Optional `cache_dir` selects a datasets cache.
Local filesystem paths continue to use `load_from_disk()` and the local manifest.

### Sharing state across instructions

Release training supports two input orders through `data.prefix_layout`:

| Layout | Shared query branch | Candidate branch |
| --- | --- | --- |
| `instruction_state` (default) | System prompt, instruction, state | Candidate or document |
| `state_instruction` | System prompt, state, instruction | Candidate or document |

Both layouts keep the system prompt first and keep the instruction and state in
one shared query branch. Only their order changes; candidate text, soft targets,
sampling, and attention structure are unchanged. Different instructions still
produce different query prefixes even when the state is identical.
`query_length` covers the full query branch and `document_length` covers the
candidate branch. Changing order can change which text survives truncation.

The previous experimental `state` layout moved instructions into candidate
branches. It is no longer accepted; it must not be treated as an alias for
`state_instruction`, since these layouts have different semantics.

The selected layout is recorded in `data_manifest.json` and the saved model's
`release_rendering.json`. For inference on release rows, read that file and call
`bekko_system_one.release.render_group(row, position, prefix_layout=layout)`, then
pass the resulting groups to `predict()`. Generic `Group` and `rank()` inputs are
already rendered text and do not automatically apply this setting. When continuing
training, explicitly set the matching `data.prefix_layout` (or deliberately change
it for an experiment); loading model weights alone does not select a data renderer.

`validation_samples` is a case cap per dataset across all its validation splits;
all decisions within each sampled case are retained. Sampling uses a fixed seed,
and indices, case IDs, fingerprints and manifest hashes are recorded in
`data_manifest.json`. The separate test release is evaluated in full after saving
and reloading the final model. `before_training` also records initial validation
and test metrics for a fixed-budget before/after comparison; test metrics do not
control stopping or checkpoint selection.

`validation_sampling: balanced` balances argmax labels for Noul and Choice with
repeated candidate ID/description sets. With five cases, two adequately supplied
labels receive 3 and 2 cases; the seed determines the extra case. Ties form a
separate stratum. Scarce labels are exhausted before unused slots are reassigned,
and hard / high-confidence soft / low-confidence soft proportions are retained
within single-axis label quotas as closely as rounding permits. Targets stay
unchanged. Multi-axis cases use greedy label coverage followed by quota deficits
and rarity, preserving all decisions in each selected case. Their quotas can
conflict. Score and variable-answer Choice cases without eligible balancing axes
use uniform sampling.

The case cap stays strict even when there are more classes than slots. Sampling
audits in `data_manifest.json` record available, desired and selected label counts,
missing or unavailable labels, confidence bands and unmet quotas. All validation
splits for a dataset are considered together; test data is never substituted.
Balanced selection scans validation labels once at startup and is reused at every
evaluation. It changes the evaluation distribution, so compare runs using the same
policy and sample IDs. The default `validation_sampling: uniform` retains the
earlier sampling behavior; `validation_samples: null` evaluates all validation cases.

Release configs can omit entire datasets by their exact manifest names:

```yaml
data:
  root: /path/to/training-release
  exclude_datasets: [unwanted_training_subset]
evaluation:
  validation_exclude_datasets: [unwanted_validation_subset]
  test_root: /path/to/test-release
  test_exclude_datasets: [unwanted_test_subset]
```

These three lists are independent: a training exclusion does not implicitly remove
validation or test data. Names must exist in the corresponding release manifest;
unknown names, duplicate names and non-list values are rejected. A known dataset
without the requested split has no effect. Exclusions apply before loading,
sampling and computing the training budget. Requested exclusions and the skipped
splits are recorded in `data_manifest.json`. Empty training or configured test
selections are errors; a release without validation is allowed. This is dataset
selection, not per-row language detection or filtering.

Weighted sampling selects a source with probability proportional to its decision
count raised to `sampling_alpha`. `epoch_fraction` is a decision budget, rounded
down to complete logical batches, not unique dataset coverage. Sources smaller
than one logical batch are excluded and reported in `result.json`.
Metrics include per-dataset/task cross-entropy, accuracy (ties in the target count
as correct), Brier score and target mass at the predicted candidate. Summary files
weight datasets equally within each task and tasks equally for mean cross-entropy.

### Per-dataset sampling without replacement

Use `data.sampling: uniform` to visit a configured fraction of each source,
without alpha weighting or repeated decisions:

```yaml
data:
  root: /path/to/release
  sampling: uniform
  epoch_fraction: 0.1
  dataset_samples:
    large_dataset: "20%"
    another_dataset: 200000
```

Limits use **decision counts**, not case counts. A percentage or integer specifies
the source budget at `epoch_fraction: 1`. For each source, training selects
`floor(base_count * epoch_fraction)` decisions, where a percentage sets
`base_count = floor(source_count * percentage)` and an integer sets
`base_count = min(source_count, integer)`. Thus 20% followed by a 10% run selects
approximately 2% of the original source (subject to integer rounding). Unlisted
sources default to 100%; zero skips a source. Unknown or excluded source names
are rejected. Validation and test are unaffected.

Remove `sampling_alpha` when using this mode; supplying it is an error.
For uniform sampling, `epoch_fraction` must be in `(0, 1]`. The selection is reproducible for a fixed seed and source
ordering; smaller budgets select nested subsets when the seed and per-source
limits are unchanged. Logical batches contain one source, their order is
shuffled, and partial final batches are retained. Each selected decision is
visited once. Source-pass repetition settings do not apply to this mode.

`sampling_plan.json` records the original counts, per-source base budgets and
selected counts. `result.json` records the actual number of trained decisions.
A `max_steps` smoke limit can stop before the full selection is visited; the plan
still describes the full requested budget.

### Mixing instruction and state order during training

Supply weights instead of a fixed `prefix_layout` to choose an order for each
training decision:

```yaml
data:
  root: /path/to/release
  prefix_layout_weights:
    instruction_state: 1
    state_instruction: 1
```

There is no `auto` flag. Equal weights give a 50/50 probability; `3:1` gives
75/25. Observed counts need not match the ratio exactly. Missing layouts have
weight zero. Weights must be finite nonnegative numbers with a positive total.
Specifying both `prefix_layout` and `prefix_layout_weights` is an error. If both
are absent, the default remains `instruction_state`. Mixed layouts require
structured release data (`data.root`).

The selection uses a separate seeded random stream, so evaluation and source
sampling do not consume its random state. Each occurrence of a decision gets one
order shared by all its candidates; a later occurrence may get another order.
Orders are selected before microbatching and remain unchanged during OOM retries.
The system prompt stays first and candidate text is unchanged.

Mixed training evaluates **both fixed orders**, even if a training weight is zero,
on the exact same validation/test samples. Output names include the order, for
example `test-instruction_state-summary.json` and
`test-state_instruction-summary.json`, or
`validation-100-state_instruction-summary.json`. No ambiguous unsuffixed evaluation
file is written. Evaluation therefore does more work than fixed-order training.

History records each batch's `prefix_layout_counts` and `prefix_layout_sha256`;
`result.json` records total layout counts. `data_manifest.json` and the saved
model's `release_rendering.json` record the weights and evaluation layouts.
At inference time, explicitly choose a fixed layout in `render_group`; training
weights do not make inference random or automatically detect the input layout.

For controlled comparisons, fixed-layout training can also evaluate both orders:
set `evaluation.prefix_layouts: [instruction_state, state_instruction]`. This
uses the same order-suffixed evaluation filenames as mixed training. Set
`evaluation.validation_seed` to fix validation case selection independently of
the training seed; otherwise it defaults to the training seed.

### Balanced instruction/context truncation

For structured release inputs, set the model's `query_truncation` to `balanced`:

```yaml
model:
  query_length: 4096
  document_length: 2048
  query_truncation: balanced
```

The query limit includes CLS/SEP, the system prompt, field labels, and the
separator. After reserving those tokens, half of the remaining budget is
available to each of instruction and context. Unused capacity transfers to the
other field. For a 4,000-token content budget, lengths of 8,000/1,000 retain
3,000/1,000; lengths of 1,000/7,000 retain 1,000/3,000. If both exceed their
shares, each retains 2,000. An odd token goes to instruction. Field tails are
removed. A system prompt that leaves fewer than two content tokens raises an
error instead of silently removing the fields.

Components are tokenized separately before layout selection, so both layouts
retain identical content token sequences. This can differ from tokenizing a
single concatenated string even when no truncation is necessary. The setting
is saved in the model and used for training and evaluation; checkpoints without
it retain the existing `right` truncation behavior. Candidate tokenization still
uses the independent `document_length` limit.

Release rendering supplies explicit field boundaries automatically. For custom
inference or JSONL inputs, supply `QueryParts` rather than attempting to recover
boundaries from text labels:

```python
from bekko_system_one import Group, QueryParts, predict

parts = QueryParts(
    instruction="Choose the supported answer.",
    context="Evidence to assess.",
    system="Evaluate the evidence carefully.",
    layout="state_instruction",
)
group = Group(parts.render(), ["Candidate A", "Candidate B"], "choice", query_parts=parts)
probabilities = predict(model, [group])
```

`Group.from_dict` accepts a `query_parts` object with these same fields. Its
`query` must match `QueryParts.render()`. Balanced models reject unstructured
queries, including plain `(query, candidate)` pairs and `rank` calls; use the
structured group API. This avoids silently applying a different policy at
inference time. Switching truncation policies also changes evaluation inputs,
so evaluate comparison checkpoints with the same policy.

### Trial budgets from one configuration

Use the same base YAML for small trials:

```bash
bekko-system-one --config train.yaml --smoke
bekko-system-one --config train.yaml --train-percent 1
```

`--smoke` means 0.1%; `--train-percent 1` means 1%. These options replace
`data.epoch_fraction`, rather than multiplying its existing value. With uniform
sampling, the fraction is applied to each dataset **after** its configured cap,
rounded down to whole decisions. Small datasets can therefore contribute zero
decisions to a smoke run. Weighted sampling applies the fraction to the total capped-pool count
to set its presentation budget. Percentage overrides are not supported for `source_passes`.

The output directory automatically gets `-smoke` or `-1pct` appended to its
basename. Use `--output-dir runs/another-trial` for a repeat; existing directories
are never overwritten. `--max-steps` remains an optional additional step limit.
Evaluation settings and the initial checkpoint stay as specified in the YAML:
a 1% trial starts independently from that checkpoint, not from the smoke output.
The effective settings are saved in each run's `training_config.json`; uniform
runs also save their exact per-dataset counts in `sampling_plan.json`. No extra
YAML files are needed.

### System One v1 case format

`bekko_system_one.dataset_schema` defines the `system_one.v1` case format. It keeps
model input separate from targets while allowing structured task context:

```python
from bekko_system_one.dataset_schema import from_legacy, inference_input, dataset_features

structured_row = from_legacy(legacy_row)
model_input = inference_input(structured_row)
```

The Arrow row stores `input.state_json` with `input.decisions`. A decision is
either a `judgment`, with a Noul, Choice, or Score type and typed `criteria`, or
a `ranking`, with a list of `documents`. The separate `targets` list stores
decision IDs, target IDs, probabilities, and annotation kind. Targets and
provenance are outside `input`. State, instructions, criterion descriptions,
and document contents are JSON strings so structured values can be preserved.
Use `dataset_features()` when creating a Hugging Face `Dataset` from converted rows.

Ranking documents and judgment criteria have distinct roles. An IR reranking
case can put the query in the shared state and its candidate documents in one
ranking decision; its target distribution is relative to that candidate set. A
pointwise relevance case can put one query-document pair in the state and use a
Score decision with ordered relevance criteria. Its target is a judgment on
that score scale.

`inference_input(structured_row)` returns the parsed state and decisions without
targets, provenance, or conversion metadata. Keep targets on the training and
evaluation side of the boundary. The release loader accepts both `system_one.v1`
rows and legacy flat rows. `DecisionSource` indexes structured decisions against the
separate target list, and `render_group` adapts a selected case at render time.
Converted rows retain the established rendering, candidate order and text, and
target alignment. Uniform and balanced sampling both select whole cases, even
when a case has several decisions.

To initialize full-parameter training from a learned LoRA checkpoint, load it
with `frozen_linear_bf16=False` (and `attention_backend="sdpa"` on CPU), then
use `bekko_system_one.checkpoint.merge_lora_for_full_training(model)`. This
returns a separate FP32 model with adapter weights merged into the backbone,
all parameters trainable, and non-LoRA checkpoint settings. Save it with
`save_pretrained`. Floating-point rounding can introduce small score differences;
check representative predictions before starting a controlled comparison.

### Capped alpha sampling

With `sampling: weighted`, `dataset_samples` limits each source's fixed eligible
pool before applying alpha. Source weights are `base_count ** sampling_alpha`;
the total update budget is `floor(sum(base_counts) * epoch_fraction / batch_size)`.
Rows are shuffled within the fixed pool and may repeat after it is exhausted;
there are no duplicates within a batch. Pools smaller than one logical batch
are excluded from source selection (their counts still contribute to the budget).
Caps accept the same integer or percentage syntax as uniform sampling. A zero
cap removes a source from both the budget and selection. Validation/test are
unchanged. `sampling_plan.json` records caps, weights, exclusions and budget.

Native Noul criteria are also included in the query's state as
`{"noul":{"yes":"<true criterion>","no":"<false criterion>"},"state":<original state>}`.
This gives each candidate access to both meanings, including a task-specific negative
meaning such as “not supported” rather than “contradicted”. Definitions precede the
original state so right truncation retains them first. The original state is nested
without overwriting any keys. Candidate IDs, descriptions, order and soft targets
remain unchanged. Each decision uses its own criteria. Noul requires an explicit
`true`/`false` or `yes`/`no` pair; missing criteria are not synthesized.

Use the same rendering for label-free native inference:

```python
from bekko_system_one import predict, render_input_group

groups = [
    render_input_group(case_input, i, prefix_layout="instruction_state")
    for i in range(len(case_input["decisions"]))
]
probabilities = predict(model, groups)
```

`case_input` is the structured row's `input` object (`state_json` and `decisions`).
No targets or provenance are passed to inference. Choice and Score criteria remain
candidate branches; ranking documents remain document branches.

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


## Optional Choice interaction

Set `model.choice_interaction: {width: 128, heads: 4}` to add one residual
attention block over mean-pooled candidate vectors. This works with a newly built
model or when continuing an independent checkpoint. The final correction starts
at zero, preserving the original scores before training. Existing contextual
checkpoints restore the block automatically; omit this option when resuming them.

Only candidates in the same Choice decision interact. Sharing a prefix across
multiple decisions does not merge their candidate sets. No candidate-position
embeddings are added. Noul, Score, and reranker heads keep their independent
scoring paths. Joint training can still change their shared encoder.

Use complete `Group` objects with `predict`/`predict_typed` or `prepare_batch`.
A bare pairwise `encode` call cannot supply a Choice candidate set and is rejected
by contextual heads. `rank(..., task="choice")` collects candidate embeddings
across chunks before applying the comparison head. Legacy, optimized, and fast
CUDA Graph inference preserve the same decision boundaries.

The prefix and candidate encodings remain independent of the candidate set;
only the comparison head depends on it. The head adds quadratic attention in the
number of candidates, rather than their combined token length. This is an
experimental capacity increase, not a guarantee of better accuracy or calibration.

The browser ONNX exporter currently rejects contextual Choice checkpoints; its
portable graph supports independent heads only.

## Standalone inference export

Export a self-contained Sentence Transformers model with `trust_remote_code=True`,
minimal runtime dependencies and optional `torch.compile`. See
[standalone v0 inference](docs/inference-v0.md) for export, loading and CLI usage.
