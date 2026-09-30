# Compatibility and requirements

The training and development lockfile targets Linux x86_64 and Python 3.12.
See [installation](../README.md#python-setup) for pinned versions. CPU inference
support does not imply that the full training environment installs on every OS.

## Execution paths

| Path | Environment | Artifact and requirements |
| --- | --- | --- |
| Training | Pinned Python environment; provided recipes use one CUDA GPU | ModernBERT-compatible backbone or Bekko training checkpoint; provided recipes require FA2 |
| Package inference | Pinned Python environment; CPU or CUDA | Training checkpoint; disable CUDA-specific options on CPU |
| Standalone v0 inference | Exported requirements; CPU or compatible CUDA environment | v0 export; no training package, PEFT, datasets, or W&B; FA2 optional |
| Node.js inference | Node.js 22+; CPU | ONNX export and browser directory dependencies |
| Browser inference | WASM; WebGPU optional | Supported ONNX export; WebGPU requires a secure context and usable adapter |

Standalone requirements are separate from the training lockfile. Validate them
in the target environment. This guide does not establish macOS, Windows, or MPS
support. Browser memory requirements vary with model and input size.

## Model features

| Feature | Training / package inference | Standalone v0 export | Browser ONNX export |
| --- | --- | --- | --- |
| ModernBERT shared-prefix encoder | Supported | Supported | Supported within exporter constraints |
| LoRA checkpoint | Supported | Adapters merged into full weights | Not accepted directly |
| Task marker tokens | Supported | Preserved | Not supported |
| Choice interaction head | Supported | Preserved | Not supported |
| Balanced query truncation | Requires structured boundaries | Supported | Required |
| Right query truncation | Supported | Supported | Not supported |
| Decision heads | Configurable task heads | Checkpoint heads preserved | Three scalar Choice / Noul / Score heads |

The browser exporter requires a full-weight training checkpoint containing
`0_SharedPrefix/` and `1_DecisionHeads/`. A standalone export is a different
artifact; do not substitute it for the browser exporter's input.

## Input budgets and parity

Training and package inference use the checkpoint's query/document limits.
Fresh standalone exports use [adaptive context allocation](inference-v0.md#adaptive-input-budget).
Browser inference uses its exported manifest. These paths can retain different
text when an input exceeds a budget; do not assume parity for truncated inputs.

Compare rendering, prefix order, token limits, IDs, and numeric values when
checking outputs. Precision, attention backends, and quantization can also
change probabilities. Verify supported ONNX exports with the
[browser parity checks](../browser/README.md#export-a-checkpoint).

Distributed training and optimizer-state resume are not implemented. Continued
training from a checkpoint starts a fresh optimizer and schedule.
