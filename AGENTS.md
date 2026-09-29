# Repository guidance

Bekko System One is a standalone training and inference toolkit for shared-prefix
rerankers and Choice / Noul / Score decisions. Keep code, documentation, and
examples usable from a fresh checkout.

## Layout

- `src/bekko_system_one/`: reusable Python implementation.
  - `training.py`, `cli.py`, `sampling.py`: training entry point and budgets.
  - `dataset_schema.py`, `release.py`, `hub_release.py`: typed data and rendering.
  - `modules.py`, `prefix.py`, `packed_prefix.py`: encoder and task heads.
  - `model.py`, `inference.py`: package inference and runtime caches.
  - `export_v0.py`, `inference_v0.py`: standalone export and runtime.
- `tests/`: CPU tests with tiny local models and explicitly marked CUDA tests.
- `configs/`: portable example configurations and documented release recipes.
- `examples/`: small, synthetic public inputs.
- `docs/`: training and inference references.
- `browser/`: React UI, ONNX/Node runtime, exporter, and browser checks.
- `data/`, `output/`: ignored local data, checkpoints, and generated results.

## Implementation boundaries

- Accept data paths, model revisions, and training settings through configuration.
  Keep config parsing and validation in the reusable package.
- Do not add machine-specific paths, private data locations, sibling-checkout
  dependencies, credentials, or internal experiment history to code or docs.
- Keep experiments out of reusable defaults. Public examples must state their
  data, hardware, authentication, and optional dependency requirements.
- Preserve existing work in the checkout. Do not overwrite unrelated changes.
- Keep README focused on onboarding; put detailed behavior in `docs/` and update
  relevant documentation when changing public APIs or configuration semantics.

## Model and data invariants

- Preserve candidate order, explicit candidate IDs, numeric score values, and
  soft targets. Targets and provenance must not enter tokenized input.
- Respect explicit dataset splits. Never use test data as validation or silently
  repartition a release. Record revisions and sampling settings for reproducibility.
- Keep candidate groups complete through batching, accumulation, and OOM replay.
  Candidates remain independent in the encoder; optional Choice interaction is
  scoped to one decision after pooling.
- Keep training and inference rendering, prefix order, truncation, and task-marker
  behavior aligned. Check saved/reloaded and exported models when changing these.
- Standalone exports must run without importing this package. Keep browser export
  limitations explicit and verify parity for supported architectures.
- Separate measured results from hypotheses; document benchmark conditions and
  startup costs rather than claiming unconditional speedups.

## Validation

Use Python through `uv run`; preserve the pinned stack and `uv.lock`.

```sh
uv sync --locked
uv run --locked tox
# Explicit CPU-only test selection:
uv run --locked pytest -m "not cuda"
```

Run focused checks while developing and relevant broader checks before finishing.
For GPU changes, select an available GPU and install the matching optional wheel:

```sh
uv sync --locked --extra fa2
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 pytest -m cuda
```

For browser changes, use `npm ci` and `npm run build` in `browser/`.
The full `npm test` suite requires local ONNX parity fixtures; follow
`browser/README.md` for fixture-free checks and export validation. Do not start
training, publish packages/models, or deploy a Space as a side effect of docs or
unit-test work. Report what was checked and any checks that could not run.
