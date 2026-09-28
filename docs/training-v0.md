# Training Bekko System One v0

These release recipes run entirely from this repository. They initialize from the
public `cross-encoder/ettin-reranker-{17,68,400}m-v1` models and read two private
Hugging Face datasets through `datasets.load_dataset()`:

- [Bekko v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0): train and validation.
- [S1MB](https://huggingface.co/datasets/hotchpotch/s1mb-dataset): test only.

No previous training checkpoint, sibling checkout, local dataset export, or
conversion script is required. Dataset access and the initial model download
require network access and a Hugging Face account authorized for both datasets.
Training writes a self-contained Sentence Transformers model under the run's
`model/` directory. It does not upload a model release automatically.

## Setup

Run commands from the repository root. Use a CUDA GPU and the pinned Linux /
Python 3.12 environment:

```sh
uv sync --locked --extra fa2
uv run --extra fa2 hf auth login
uv run --extra fa2 wandb login
```

A saved Hugging Face login or `HF_TOKEN` supplies dataset authentication. A saved
W&B login or `WANDB_API_KEY` supplies tracking authentication. Do not put API keys
in YAML. W&B uses the authenticated user's default entity; set `wandb.entity` in
YAML to select a team.

The recipes omit `revision`: each run resolves the current Hub `main` for both
datasets and the backbone, then uses those immutable commits throughout that run.
`resolved_config.json`, `data_manifest.json` and W&B retain the resolved revisions.
To reproduce a run, pass its `resolved_config.json` as `--config` and supply a new
`--output-dir`. JSON is accepted by the YAML config reader. This restarts training
from the same base model and data; optimizer-state resume is not implemented.

## Recipes

| Size | Full, 300k cap | 10%, alpha 0.5 | Smoke |
|---|---|---|---|
| 17M | [full](../configs/bekko-system-one-v0-17m.yaml) | [10%](../configs/bekko-system-one-v0-17m-10pct.yaml) | [smoke](../configs/bekko-system-one-v0-17m-smoke.yaml) |
| 68M | [full](../configs/bekko-system-one-v0-68m.yaml) | [10%](../configs/bekko-system-one-v0-68m-10pct.yaml) | [smoke](../configs/bekko-system-one-v0-68m-smoke.yaml) |
| 400M | [full](../configs/bekko-system-one-v0-400m.yaml) | [10%](../configs/bekko-system-one-v0-400m-10pct.yaml) | [smoke](../configs/bekko-system-one-v0-400m-smoke.yaml) |

```sh
# Start with the small integration run.
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config configs/bekko-system-one-v0-17m-smoke.yaml

# Weighted 10% training budget.
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config configs/bekko-system-one-v0-17m-10pct.yaml

# Full capped pool.
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config configs/bekko-system-one-v0-17m.yaml
```

Substitute `68m` or `400m` to select a larger recipe. Expose exactly one GPU per
process. Independent runs can use different GPUs; these are single-GPU recipes,
not distributed training. Output directories are distinct by size and recipe,
and existing directories are rejected. Use `--output-dir output/my-new-run` for
a repeat. A second run will receive a distinct W&B run ID even with the same name.

All nine recipes use full encoder training (`lora: null`), Choice / Noul / Score
heads, cosine decay, 10% warmup, and a 1:1 mix of instruction/state prefix orders.
The query/document limits are 4096/2048 with balanced query truncation. The 400M
recipe uses gradient checkpointing. Adaptive token budgets bound microbatches;
the logical full/10% batch size remains 512 decisions.

`data.dataset_cap: 300000` bounds each source pool in **judgments**, before its
sampling weight or global training fraction is calculated. The published Bekko
v0 train is already physically capped with complete cases/groups. The trainer's
cap is a second decision-level limit; it does not create a new train/test split.
Explicit `dataset_samples` limits, when present, apply before the global cap.

- Full uses uniform sampling without replacement and includes short tail batches.
- 10% uses source probabilities proportional to `capped_count ** 0.5`. Its budget
  is `floor(sum(capped_counts) * 0.1 / batch_size)` optimizer steps. This is sampled
  volume, not unique 10% coverage: smaller sources can repeat. Weighted mode
  excludes sources smaller than a logical batch and discards pool tails before
  recycling; `sampling_plan.json` lists those sources.
- Smoke selects eight train configurations with up to 64 judgments each and a
  logical batch size of 32. It includes all three typed heads and relevance tasks.
  Validation uses two sampled cases per available source; test uses four complete
  S1MB configurations. The smoke score is not the full S1MB benchmark result.

Full and 10% evaluate all active S1MB configurations, in both prefix orders.
Validation uses up to 50 balanced cases per source, during training and at the
end. S1MB test is evaluated initially only for 10%/smoke, and after saving and
reloading the final model for every recipe. It is not used to choose a checkpoint.
Quarantined S1MB sources are not named Hub configurations and are not loaded.

The dedicated smoke YAML limits downloads and evaluation. The generic CLI
`--smoke` flag only replaces the fraction with 0.1%; it does not make the same
subset/evaluation selection. Use the dedicated YAML for a quick integration run.
Likewise, `--train-percent 10` preserves the sampling mode; choose the `-10pct`
YAML when alpha=0.5 weighting is intended.

## W&B and local reports

Recipes enable online W&B tracking in project `bekko-system-one`, group
`bekko-system-one-v0`. Size, budget and cap are tags; run names default to the
output directory name. For offline work, set `wandb.mode: offline` in a config
copy; use `disabled` to avoid W&B entirely. Tracking failures are surfaced rather
than silently switching to an untracked run.

Recorded information includes:

- Resolved model/data commits, effective config, cap and sampling plan, parameter
  counts, and planned steps.
- Loss, both learning rates, gradient norm, decisions processed, throughput,
  adaptive token budget, peak GPU memory and OOM retries.
- Validation/test macro metrics for each prefix order, with numeric Score MAE /
  normalized MAE and Noul Brier where applicable. Dataset-level metrics appear as
  tables; initial/final metrics are also retained in the run summary.
- Final training usage and a small artifact containing resolved config, sampling
  plan, usage, result and evaluation-summary JSON files.

Raw examples, state, targets and model weights are not uploaded to W&B. The model
checkpoint stays local. Tracking uses `train/step` as its explicit chart axis and
closes the run with a failure status if training raises an exception.

Local files remain available independently of W&B: `training_config.json`,
`resolved_config.json`, `sampling_plan.json`, `data_manifest.json`, `history.jsonl`,
`training_usage.json`, evaluation JSONs, `result.json`, and `wandb_run.json`.
The last file contains the run URL. Final model loading uses local checkpoint
files only, including the tokenizer, backbone configuration, weights and heads.

## Verification

On 2026-09-28, all three dedicated smoke recipes completed from this repository
using the Hub sources and saved login credentials, with no local dataset or
initial-checkpoint dependency. Each trained 512 judgments in 16 updates, evaluated
both prefix orders after checkpoint reload, and finished online W&B syncing with
zero OOM retries. W&B history and final metrics were compared with the local
reports. These checks establish integration, not full-training model quality.
The CPU suite passed 161 tests (8 CUDA tests deselected), with lint and type checks
passing. Full and 10% training were not executed as part of this verification.
