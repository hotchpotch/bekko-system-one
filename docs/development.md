# Development

Use Linux x86_64, Python 3.12, and uv. Run from the repository root:

```sh
uv sync --locked
uv run --locked tox
```

Tox runs CPU tests, Ruff, and ty with Hugging Face downloads disabled.
Tests construct tiny local models and do not require dataset credentials.
For focused checks:

```sh
uv run --locked pytest -m "not cuda"
uv run --locked ruff check .
uv run --locked ty check
```

## GPU checks

Install the matching optional FlashAttention wheel and select an available GPU:

```sh
uv sync --locked --extra fa2
CUDA_VISIBLE_DEVICES=0 uv run --locked --extra fa2 pytest -m cuda
```

Keep the pinned PyTorch/CUDA/FlashAttention versions compatible when updating
dependencies, and update the lockfile together with dependency changes.

## Browser checks

See [the browser guide](../browser/README.md) for Node.js setup and export commands.
Run `npm ci`, the fixture-free tests documented there, and `npm run build` from
`browser/`. Full `npm test` additionally requires local ONNX parity fixtures.
A successful build alone does not validate inference parity or WebGPU execution.
