# Reproduce Saihai v1 310M

Saihai v1 310M is a Japanese typed-decision finetune of
`cl-nagoya/ruri-v3-reranker-310m` using this toolkit. The model and its exact
training pool are initially access-controlled research releases. Authentication
and repository access are required. Preserve the source-specific data terms;
repository access is not a grant of public redistribution rights.

## Train the materialized mixed pool

Use Python3.12 and the locked environment, including the matching FA2 wheel:

```sh
uv sync --locked --extra fa2
hf auth login
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 python -m bekko_system_one.training \
  --config configs/saihai-v1-310m.yaml
```

Select an available GPU for your machine. The pinned Dataset contains the
already selected alpha=.5 Bekko v0 10% subset (840,947 judgments) plus all
131,171 Japanese judgments, totaling972,118. `all-mixed` preserves the original
row order. Use uniform sampling, epoch_fraction1, seed42 and BS512:1,899
updates including the tail. Applying alpha or a 10% fraction again changes
this experiment. Original validation is retained and monitored in both layouts.
The dataset's selection.json and source lineage document how the fixed subset
was constructed; unknown source rights remain explicitly qualified.

The recipe uses the pinned Ruri base, full encoder finetuning, encoder/head
LR1e-5/2e-4, cosine, warmup10%, query4096/document2048, shared prefixes,
FA2 BF16, gradient checkpointing and equal prefix-layout weights.
All checkpoints and resolved settings go to the config's relative output path.

For exact implementation provenance, the model repository includes a frozen
`training/code/bekko_system_one/` snapshot and dependency manifests. The
original training used a development working tree, so the recorded HEAD alone
is insufficient to identify its source. Download the pinned model repository,
install this repository's locked training dependencies, and select that source:

```sh
PYTHONPATH=MODEL_SNAPSHOT/training/code CUDA_VISIBLE_DEVICES=0 \
  uv run --locked --extra fa2 python -m bekko_system_one.training \
  --config MODEL_SNAPSHOT/training/config.yaml
```

Replace MODEL_SNAPSHOT with the downloaded model folder. The frozen code has
no dependency on another experiment workspace. Its package hashes are in the
release manifest. Matching data/code/config does not promise bit-identical
weights across different hardware or nondeterministic kernels.

## Export and infer

```sh
uv run --locked python -m bekko_system_one.export_v0 \
  --checkpoint output/saihai-v1-310m/model \
  --output output/saihai-v1-310m-inference
```

The existing exporter produces an independently loadable inference module,
weights and tokenizer. Install the exported requirements.txt for inference.
The native checkpoint is also retained under checkpoint/ in the model release.

```python
from transformers import AutoModel

repo = "hotchpotch/saihai-v1-310m"
revision = "FULL_COMMIT_HASH"
model = AutoModel.from_pretrained(repo, revision=revision, token=True, trust_remote_code=True,
            device="cpu", attn_implementation="sdpa")
# request is a native input object with state_json and ordered decisions.
result = model.predict(request, query_length=4096, document_length=2048)
```

See [standalone inference](inference-v0.md) for Choice/Noul/Score request
formats, adaptive input budgets and backend selection. Pass no targets or
provenance into inference. Pin the same model revision for code and weights.

The model card reports the original checkpoint's full S1MB japanese-v1
measurements. Export/Hub verification records short-input parity separately;
it is not a new full benchmark reevaluation. The evaluation Dataset is
separately access-controlled and is not bundled with this model or its train.
