# Inference speed demo

A standalone terminal demo of Bekko System One on S1MB Choice, Noul and Score
tasks. Requires Linux/macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Dependencies are installed automatically. The first run downloads the public
S1MB dataset and pinned model from Hugging Face; no token is normally required.
GPU execution requires a compatible NVIDIA driver for PyTorch CUDA 13.0.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --script demo/run_bekko_system_one_v0_speed.py
```

You can also select another supported Bekko model with `--model`: `17m`
(default), `68m`, or `400m`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --script demo/run_bekko_system_one_v0_speed.py --model 68m
```

Compare published benchmark scores on the
[S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard).
The demo's local scores reflect its run configuration and may differ from
published results; its throughput measurements are separate from benchmark scores.

Use a wide terminal (around 180 columns). Initialization loads, tokenizes and
prepares the model before prompting for Enter. Press Esc to stop after the current
batch or exit the final screen; the final frame remains in terminal history.

Options:

- `--model 17m|68m|400m` (default: `17m`), `--bs 32`, `--seed 42`.
- `--passes 3`: replay the dataset three times; use `1` for a single pass.
- `--warmup-batches 16`: prepare the first 16 batches; also accepts `all` or `0`.
- `--input-max N`: override the checkpoint-native context limit.
- `--verbose-log /tmp/bekko-speed.jsonl`: write diagnostics to a new file.
- `--device cpu --no-compile`: run without a GPU.

Compiled execution uses SDPA by default and falls back to eager inference when
a new compilation would be needed after warmup. Use `--allow-recompile` to allow
runtime compilation. Live rates update once per second over a five-second window;
final rates cover the whole run. Tokens measure pretokenized input volume, not
generated text. Initialization and warmup are excluded from inference timing;
wall time includes the UI. Replay passes increase work counts, not unique questions
or benchmark scoring weight.
