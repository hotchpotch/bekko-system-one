# Train with Bekko dataset

Train a small 17M integration checkpoint using the public
[Bekko v0 dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0),
export it, and run a prediction. Run every command from the repository root.

## Requirements

Use the Linux x86_64 / Python 3.12 environment described in the
[README](../README.md), a compatible NVIDIA GPU and driver, and network access
for the backbone and dataset downloads:

```sh
uv sync --locked --extra fa2
```

This training recipe uses CUDA and FlashAttention 2. CPU inference is supported.
No measured minimum VRAM or runtime is specified for this walkthrough. Download
time, training time, and peak memory depend on hardware, input lengths, and
caches. A small decision budget does not imply a small download.

## Prepare a public-data configuration

The release recipes include S1MB test evaluation, authenticated Hub access,
and online W&B tracking. Create a copy for a first run using only Bekko training
and validation data:

```sh
uv run --locked python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/bekko-system-one-v0-17m-smoke.yaml").read_text())
config["data"]["root"]["token"] = False
config["evaluation"].pop("test_root", None)
config["evaluation"]["before_training"] = False
config["wandb"]["mode"] = "disabled"
config["training"]["output_dir"] = "output/quickstart-17m"
destination = Path("output/quickstart-config.yaml")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
print(destination)
PY
```

Once Bekko is public, this configuration needs no Hub login or W&B account.
It selects up to 64 decisions from each of eight training configurations and
retains the recipe's small validation sample. It does not repartition data or
use test rows as validation. Both fixed prefix orders are evaluated.

## Train and inspect the checkpoint

```sh
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 bekko-system-one \
  --config output/quickstart-config.yaml
```

Select an available GPU with `CUDA_VISIBLE_DEVICES`. The output directory must
be new; for another run, pass `--output-dir output/quickstart-17m-repeat`.

The checkpoint is saved under `output/quickstart-17m/model/`. Inspect
`result.json` for completion and training counts, validation summaries for
integration results, and `resolved_config.json` plus `data_manifest.json` for
model/data revisions and sampling details. The [recipe guide](training-v0.md)
describes the complete artifact set.

This short run checks the pipeline. Its predictions are not representative
of a fully trained release model.

## Export and predict

```sh
uv run --locked python -m bekko_system_one.export_v0 \
  --checkpoint output/quickstart-17m/model \
  --output output/quickstart-17m-inference

uv run --locked python output/quickstart-17m-inference/inference_v0.py \
  --model output/quickstart-17m-inference \
  --input examples/v0-input.json --device cpu --attn-implementation sdpa
```

The export directory must also be new. The CLI prints JSON keyed by decision ID:
Choice includes `selected_id`, Noul includes `probability_yes`, and Score includes
`score` on the supplied scale. All include candidate probabilities.
See [standalone inference](inference-v0.md) for batching and deployment.

## Move to a full run

Choose a full or 10% [release recipe](training-v0.md), then select your budget,
evaluation sources, and tracking mode. Bekko and S1MB have separate repositories
and access conditions. To reproduce a run, use its resolved configuration and
a new output directory. Optimizer-state resume is not implemented.
