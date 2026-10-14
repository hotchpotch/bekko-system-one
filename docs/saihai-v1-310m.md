# Training Saihai v1 310M

[Saihai v1 310M](https://huggingface.co/hotchpotch/saihai-v1-310m) is a Japanese
Typed Decision Model trained from
[cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m).
It uses Bekko's shared-prefix encoder and Choice, Noul and Score heads.
This guide documents its training data, recipe and reproduction procedure.
The portable configuration is [configs/saihai-v1-310m.yaml](../configs/saihai-v1-310m.yaml).

## Training data

Use [hotchpotch/saihai-v1-310m-dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-310m-dataset),
configuration `all-mixed`. The recipe pins the original data revision
`c00c6f122a42f950a5e58582d69c34d1263250c3` and preserves the selected row order.
Later documentation and license updates do not change this training pool.

| Component | Training judgments | Selection |
| --- | ---: | --- |
| Bekko System One dataset v0 | 840,947 | Approximately 10% of its capped train pool, alpha=0.5 |
| Japanese dataset-derived and template decisions | 116,771 | All selected Japanese train examples |
| Japanese gen_gen | 14,400 | 720 subcategories, 20 cases each; no validation |
| **Total** | **972,118** | One globally mixed pool |

The pool has **820,822 training cases** and **14,416 validation cases** containing
**21,884 validation judgments**. A case may contain multiple decisions.
English GPT-6 Sol gen_gen is excluded; synthetic sources already present in
Bekko v0 are retained.

Bekko v0's source pool contains 8,421,771 judgments across 153 sources after
its per-source 300,000-judgment cap. Square-root allocation (alpha=0.5) sets
source quotas, respecting source capacity. Selection uses seed 42, without
replacement, and preserves complete related groups. The 10% target is 842,177
judgments; whole-group selection yields 840,947, leaving 1,230 unused.
The Dataset includes selection records and row-level source lineage.

**This selection is already materialized.** Train with uniform sampling and
`epoch_fraction: 1.0`; applying alpha or another 10% fraction changes the recipe.
Preserve soft targets, candidate order, numeric Score values and grouping.
Only the `input` column enters model inference; targets and provenance do not.

## Training recipe

| Setting | Value |
| --- | --- |
| Base revision | `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3` |
| Finetuning | Full encoder and task heads; no LoRA |
| Seed | 42 |
| Batch size | 512 decisions |
| Training duration | One pass; 1,899 optimizer updates including the tail |
| Encoder / head learning rate | `1e-5` / `2e-4` |
| Schedule | Cosine; 10% warmup |
| Weight decay | `0.01` |
| Query / candidate token caps | 4,096 / 2,048 |
| Query truncation | Balanced |
| Attention / precision | FlashAttention 2 / BF16 training |
| Gradient checkpointing | Enabled |
| Prefix layouts | Equal weights for `instruction_state` and `state_instruction` |
| Validation | Fixed balanced 120-case subset, seed 42, both layouts |
| Validation interval | Every 250 updates |
| W&B | Offline |

The learning rates above are the recorded BS512 recipe. Validation monitors the
run; the published model is the final checkpoint after the complete pass.
Shared-prefix packing and token-budget microbatches preserve complete decisions.
The configured batch size counts decisions, not candidate branches or rows.

## Run training

Use Python 3.12 and the locked training environment, including the matching
optional FlashAttention wheel. Select an available GPU; the example uses GPU 0.
If the Dataset is access-controlled, configure Hugging Face authentication with
an account that can access it before starting. The recipe uses `token: true`.

```sh
uv sync --locked --extra fa2
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 python -m bekko_system_one.training \
  --config configs/saihai-v1-310m.yaml
```

The relative output directory is `output/saihai-v1-310m`. Use a new output
location for another run to preserve previous checkpoints and measurements.
No separate dataset generation or resampling is needed.

## Reproduce the original implementation

Training against the current Bekko implementation uses the documented recipe.
For the original implementation, the model repository also includes:

- `training/code/bekko_system_one/`: the frozen trainer package.
- `training/config.yaml`: the pinned portable training configuration.
- `training/pyproject.toml` and `training/uv.lock`: dependency manifests.
- `training/release-manifest.json`: revisions, settings and package hashes.
- `checkpoint/`: the original native checkpoint.

The original run used a development working tree; its Git HEAD alone does not
identify every source file. Download a fixed model snapshot and verify the
recorded hashes, install the compatible locked training dependencies, then use
its frozen package instead of the installed trainer:

```sh
PYTHONPATH=MODEL_SNAPSHOT/training/code CUDA_VISIBLE_DEVICES=0 \
  uv run --locked --extra fa2 python -m bekko_system_one.training \
  --config MODEL_SNAPSHOT/training/config.yaml
```

Replace `MODEL_SNAPSHOT` with the downloaded model directory. The snapshot is
self-contained and does not require another experiment workspace. Matching
code, data and settings does not guarantee bit-identical weights across hardware
or nondeterministic kernels. Keep model/data revision pins in reproduction records.

## Export and use the model

Export the resulting native checkpoint into an independently loadable runtime:

```sh
uv run --locked --extra fa2 python -m bekko_system_one.export_v0 \
  --checkpoint output/saihai-v1-310m/model \
  --output output/saihai-v1-310m-inference
```

Choose a new export directory. Install its `requirements.txt` on the inference
machine; neither the Bekko nor saihai training package is required.
The Hub model supports the same AutoModel interface:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "hotchpotch/saihai-v1-310m",
    trust_remote_code=True,
    device="cuda",
    attn_implementation="sdpa",
)
response = model.systemone({
    "state": "昨日購入した商品を返品したいです。返金をお願いします。",
    "questions": {
        "refund": {"type": "noul", "instructions": "顧客は返金を求めているか？"},
    },
}, show_progress_bar=False)
print(response["answers"]["refund"]["noul"])
```

Use the local export path in place of the Hub ID for a newly trained model.
See [standalone inference](inference-v0.md) for native `predict()` inputs,
explicit numeric rubrics, batching, attention backends and compilation.

## Evaluation and interpretation

The original final checkpoint completed S1MB japanese-v1: 27 benchmarks,
8,067 judgments, Dataset revision `59b66dd94c430c1ec6db1c948fa2cc0410400a27`.
The predetermined primary layout was `instruction_state`.

| Metric | instruction_state | state_instruction |
| --- | ---: | ---: |
| Noul | 23.85 | 25.87 |
| Choice | 38.94 | 39.57 |
| Score | 27.09 | 26.60 |
| gen-Noul | 22.67 | 26.67 |
| gen-Choice | 45.83 | 46.63 |
| gen-Score | 0.00 | 0.00 |
| Task Avg | 29.96 | 30.68 |

These are baseline-adjusted benchmark scores on a 0–100 scale, not raw accuracy.
Task Avg weights the three task means equally. Training includes source families
represented in evaluation; results do not establish unseen-family generalization.
The gen-Score result is at or below its reference baseline.

Export and Hub checks verified short-input inference parity separately; they
are not a new full benchmark evaluation. No S1MB evaluation inputs or targets
are bundled with the training/model release. Follow the
[S1MB evaluation guide](https://github.com/hotchpotch/S1MB/blob/main/docs/evaluation.md)
when evaluating a new checkpoint and retain all six metrics and both layouts.

## Licenses and attribution

Model weights use Apache-2.0, matching the Ruri base; the model repository retains
its license and attribution. Training data follows each original source's terms,
not the model license. Japanese gen_gen's selected 14,400-case catalog was created
by [@hotchpotch](https://huggingface.co/hotchpotch) and uses CC BY-SA 4.0.
Consult the Dataset's README, `SOURCES.md`, source records and notices for the
other datasets, source-material attribution and any unresolved qualifications.
