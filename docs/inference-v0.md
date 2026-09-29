# Standalone v0 inference

Export a trained checkpoint from the training environment:

```sh
uv run python -m bekko_system_one.export_v0 \
  --checkpoint output/bekko-system-one-v0-17m-smoke/model \
  --output output/bekko-system-one-v0-17m-inference
```

Choose a new output directory. The export contains weights, tokenizer, model
configuration and one standalone `inference_v0.py`. LoRA weights, when present,
are merged during export. Optional Choice interaction weights are preserved.
The exported directory can also be uploaded as a Hugging Face model repository.

On the inference machine, install the exported `requirements.txt`. Install a
CUDA-compatible PyTorch build if using a GPU. The runtime needs PyTorch,
Transformers, Sentence Transformers, safetensors and tqdm, including their normal
transitive dependencies. It does not need this repository, PEFT, datasets, W&B
or the external FlashAttention package. Attention uses PyTorch SDPA.

The examples below import `inference_v0.py` from the exported directory; add that
folder to your Python import path or run from it. With the Bekko package installed,
use `from bekko_system_one import BekkoSentenceTransformer` instead. This subclass
accepts the usual SentenceTransformer loading arguments and exposes `predict()`,
`predict_groups()`, `compile_inference()` and `disable_compile()` directly on the
model. Plain `SentenceTransformer(...)` still loads the module format, but does
not provide these typed convenience methods.

```python
import json
from inference_v0 import BekkoSentenceTransformer

model = BekkoSentenceTransformer(
    "output/bekko-system-one-v0-17m-inference",  # or a Hub model ID
    trust_remote_code=True,
    device="cuda",  # "cpu" also works
)
with open("examples/v0-input.json") as stream:
    request = json.load(stream)
result = model.predict(request)
print(result)

# Optional: compile tensor execution using PyTorch Inductor.
model.compile_inference()
result = model.predict(request)
# model.disable_compile() restores eager execution.
```

Compilation is lazy: the first prediction includes compilation time. Input shape
changes can cause further compilation. Tokenization and typed output processing
remain eager. Measure warmed calls separately from startup; compilation does not
guarantee a speedup for every request size. CUDA uses BF16 autocast, while CPU uses
FP32; small numerical differences are expected.

The exported file also provides a CLI, without installing the training package:

```sh
python output/bekko-system-one-v0-17m-inference/inference_v0.py \
  --model output/bekko-system-one-v0-17m-inference \
  --input examples/v0-input.json --device cuda --compile
```

See [the input example](../examples/v0-input.json). Pass a native dataset row's
`input` object, containing `state_json` and `decisions`, without targets. Use
`predict()` for typed decisions; raw-text `SentenceTransformer.encode()` is not
this model's interface.

- Choice returns the selected candidate ID and its candidate distribution.
- Noul adds the authored yes/no meanings to the state and returns `probability_yes`.
- Score requires explicit numeric criterion values and returns their expected
  value, its normalized value and the underlying distribution. A 0–4 scale stays
  on that scale in the numeric score.
- Relative document ranking returns document IDs in descending probability order.

For batched inference, pass a list of native input objects:

```python
results = model.predict(
    requests,
    batch_size=128,
    token_budget=64000,
    context_length=7999,  # Must not exceed the checkpoint positional capacity.
    document_length=3800,
    show_progress_bar=True,
)
```

A single dict returns a result dict; a list returns a list in the original order.
Decision IDs may repeat across requests. Empty requests return empty result dicts.
The progress bar is enabled by default, writes to stderr and counts completed
requests. Set `show_progress_bar=False` to disable it.

`batch_size` (default 128) bounds cases rendered per window and decisions tokenized
per window. Tokenization deduplicates shared text. Decisions are sorted by branch
length within each window to reduce padding, then restored to input order.
Complete decisions are packed within `token_budget` (default 64,000), estimated
as candidate count times the sum of maximum query and document lengths in the
microbatch. This is a work estimate, not a strict memory limit. An oversized
decision runs alone; its candidates are never split. Probabilities are transferred
to the CPU once per microbatch. `predict_groups()` exposes the same batch and
length controls for already-rendered groups, with progress counted in decisions.

### Adaptive input budget

Fresh exports use `adaptive-v1`. `context_length` defaults to the backbone's
position capacity (7,999 for v0 17M). Candidate sequences are capped at
`document_length=3800` by default. Each decision initially reserves half the
context for its query and half for its longest candidate; candidates receive
any odd token. Unused capacity transfers to the other branch, subject to its cap.
All candidates in a decision use the same query. Special tokens count toward
all limits. For example:

- Query 7,200 and longest candidate 799 fit unchanged in 7,999 positions.
- Query 7,200 and candidate 3,000 retain query 4,999 and candidate 3,000.
- Overlong query and candidate use query 4,199 and candidate 3,800 by default.

`query_length` optionally caps the query (default: context capacity minus two).
`document_length` and `context_length` can also be overridden per `predict()` or
`predict_groups()` call without changing saved settings. Raising the candidate
cap permits candidates to borrow unused query capacity; if both branches are
long, they retain their half shares. `context_length` cannot exceed the backbone's
positional capacity. Lower per-call context limits clamp the default branch caps.

Query overflow follows the checkpoint's `balanced` instruction/state allocation
or `right` policy. Candidates are right-truncated, retaining their special tokens.
This is deliberate truncation, not overflow rejection. Allocation is independent
of batch neighbors. The packer starts another microbatch if combining different
branch lengths would exceed the context after padding. `token_budget` is a
separate throughput/memory work budget, not the per-decision context limit.
Training checkpoint budgets remain unchanged; re-export to get these inference
defaults. The prefix layout defaults to the checkpoint's fixed layout, or
`instruction_state` for mixed-layout training. Override with
`prefix_layout="state_instruction"` when required.

The CLI accepts either a JSON object or an array, with `--batch-size`,
`--token-budget`, `--context-length`, `--query-length`, `--document-length` and
`--no-show-progress-bar`. JSON results go to stdout and progress goes to stderr.
Previously exported model directories must be re-exported to include this runtime.

Smoke checkpoints validate execution only; use a fully trained checkpoint for
model quality evaluation and release.
