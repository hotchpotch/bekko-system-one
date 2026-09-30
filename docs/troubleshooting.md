# Troubleshooting

Run repository commands from the checkout root unless a guide says otherwise.
Check [compatibility](compatibility.md) before changing dependencies.

| Symptom | What to check |
| --- | --- |
| Dependency or FA2 wheel installation fails | Use the pinned Linux x86_64 / Python 3.12 environment; FA2 must match Python, PyTorch, and CUDA. |
| A public dataset requests a token | Release YAML files set `token: true`. Set `data.root.token: false` for anonymous Bekko access; evaluation sources have independent settings. |
| Hub access fails | Check repository ID, revision, and access conditions for that particular model or dataset. |
| W&B asks for credentials | Set `wandb.mode: disabled` for local-only runs or `offline` for offline tracking. |
| Output directory exists | Choose a new `--output-dir`. Continued fine-tuning starts a fresh optimizer. |
| Explicit FA2 loading fails | Check GPU and binary compatibility; standalone inference can use `attn_implementation="sdpa"`. |
| CPU checkpoint loading fails on CUDA options | Follow the package example: use SDPA and disable `frozen_linear_bf16` and `fused_rotary`. |
| CUDA runs out of memory | Reduce training token budgets or inference `token_budget`; inspect candidate counts and sequence lengths for an oversized decision. |
| Balanced truncation rejects plain text | Supply structured input or `QueryParts` with explicit field boundaries. |
| Predictions differ across runtimes | Compare revisions, rendering, prefix order, budgets, precision, and quantization. |
| ONNX export rejects a checkpoint | Check the supported backbone, three scalar heads, balanced truncation, and restrictions on LoRA, task markers, and Choice interaction. |
| Browser model loading fails | Check visibility, network access, CORS, and memory. Static builds have no authenticated development proxy. |
| WebGPU execution fails | Select CPU and retry; adapter availability does not guarantee sufficient memory or model support. |

Lower input limits change the text seen by the model. Record changes when
evaluating quality. Adaptive microbatching preserves complete training decisions
and cannot guarantee that arbitrarily large candidate groups fit in memory.

For further questions, use
[GitHub Discussions](https://github.com/hotchpotch/bekko-system-one/discussions).
Include the command, relevant configuration, versions, and error message, with
credentials and private input data removed.
