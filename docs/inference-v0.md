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
Transformers, Sentence Transformers and safetensors, including their normal
transitive dependencies. It does not need this repository, PEFT, datasets, W&B
or the external FlashAttention package. Attention uses PyTorch SDPA.

```python
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "output/bekko-system-one-v0-17m-inference",  # or a Hub model ID
    trust_remote_code=True,
    device="cuda",  # "cpu" also works
)
with open("examples/v0-input.json") as stream:
    request = json.load(stream)
result = model[0].predict(request)
print(result)

# Optional: compile tensor execution using PyTorch Inductor.
model[0].compile_inference()
result = model[0].predict(request)
# model[0].disable_compile() restores eager execution.
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

`predict(request, token_budget=16000)` batches complete decisions within an
approximate attention-work budget. A decision is never split across batches;
if one exceeds the budget it runs alone. The exported prefix layout defaults to
the checkpoint's fixed layout, or `instruction_state` for mixed-layout training.
Override with `prefix_layout="state_instruction"` when required.

Smoke checkpoints validate execution only; use a fully trained checkpoint for
model quality evaluation and release.
