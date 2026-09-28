# Bekko browser inference

A Node.js runtime and React static browser interface for a full-weight Bekko 17M checkpoint.
It supports Choice, Noul (binary yes/no), and ordinal Score. Node uses ONNX Runtime
CPU; the browser uses single-threaded WASM in a Web Worker. Input text never goes
to an inference server. Model, tokenizer, JavaScript, and WASM are served locally. The default model uses
INT8 token embeddings with FP32 Transformer blocks and decision heads (29.02 MB).

## Export the checkpoint

From the Python repository root, with the existing `uv` environment:

```sh
uv run --with onnx==1.23.0 --with onnxscript==0.7.2 python browser/scripts/export_onnx.py /path/to/checkpoint
```

Then create the embedding-INT8 default (and comparison variants):

```sh
uv run --with onnx==1.23.0 --with onnxruntime==1.30.0 python browser/scripts/quantize_onnx.py browser/public/model --output browser/public/model/quantized
```

The input directory must contain `0_SharedPrefix/` and `1_DecisionHeads/`.
The exporter supports the current full-weight ModernBERT model with balanced
query truncation, no LoRA, no task markers, and all three decision heads. It
preserves independent prefix attention, candidate-to-prefix attention, rotary
positions, sliding windows, mean pooling, and the trained scalar heads.

Files are generated in `public/model/` (ignored by Git). `model.onnx` is FP32,
approximately 67.5 MB for 17M, with dynamic token and candidate dimensions.
`manifest.json` records model/weight hashes and token budgets. The exporter checks
ONNX validity and compares its portable implementation with the original packed
Python encoder. It also writes reference fixtures for JavaScript parity tests.
No training data is needed during export or inference.

## Run

Requires Node.js 22 or later. From this directory:

```sh
npm ci
npm test
npm run infer -- examples/noul.json
npm run dev
```

Open the URL printed by Vite. The default binds only to localhost. An alternate
model directory can be supplied after the request path:

```sh
npm run infer -- examples/choice.json /path/to/exported/model
```

Build and preview a self-contained static site:

```sh
npm run build
npm run preview
```

Keep **all of `dist/`**, including `assets/` and `model/`. Serve it over HTTP(S),
not `file://`. There are no CDN dependencies or inference API calls. Initial
uncompressed assets total approximately 47 MB. The static build includes only
the selected embedding-INT8 model; FP32 and other comparison models stay local. A running page reuses its loaded
model; persistence across reloads depends on normal HTTP caching.

## Request and output

```js
import { loadModel } from './src/node.js';
import { renderDecision } from './src/decision.js';

const model = await loadModel();
try {
  const result = await model.predict(renderDecision({
    task: 'choice',
    instruction: 'Which topic best describes the news article?',
    state: { article: 'The tennis player won the championship final.' },
    criteria: [
      { id: 'sports', description: 'Sports news.' },
      { id: 'business', description: 'Business and economic news.' },
    ],
  }));
  console.log(result);
} finally {
  await model.release();
}
```

- `choice`: returns `selected_id` and per-ID `probabilities`.
- `noul`: use exactly two IDs, `true`/`false` or `yes`/`no`, in either order.
  Returns `probability_yes` and `probabilities`. Candidate `text` is the actual
  model input; IDs determine interpretation only.
- `score`: each candidate needs a distinct finite numeric `value`. Returns the
  probability-weighted `score`, min/max-normalized `normalized_score`,
  `probabilities`, and `values`. This is an ordinal distribution, not a separate
  regression output. Numeric values are explicit metadata, never inferred from
  candidate text.

For the low-level `predict` API, optional fields are `system` (string) and `layout` (`instruction_state`, the
default, or `state_instruction`). The runtime reproduces Python's component-wise
balanced query tokenization. For the current checkpoint, query length is 4096
and candidate length is 2048, including special tokens; excess tails are
truncated. The runtime accepts 2–64 candidates and scores them sequentially to
bound attention memory. Prefix encoding is recomputed per candidate in this
simple implementation. Full-length inputs can still be expensive on browser
CPU; short inputs are recommended for interactive demos.

## Verification

After export, `npm test` compares exact token IDs and all three head logits
against Python for English, Japanese, Unicode, empty text, different candidate
counts/lengths, both prompt orders, and input exceeding the local attention
window. Separate tokenization fixtures check query/document truncation limits.
Tests require the FP32 and embedding-INT8 exports and fail if they are absent.
The FP32 parity test keeps its strict tolerance. The default-model test separately
checks the quantization policy and bounds probability drift to 0.03 on the five
smoke fixtures; this is a regression bound, not a general accuracy guarantee.

To check the built site in an actual browser, open the preview with
`playwright-cli`, then run:

```sh
playwright-cli run-code --filename=scripts/check-browser.js
playwright-cli run-code --filename=scripts/check-model-loading.js
```

## Future Hugging Face Static Space

Build locally and copy the **contents** of `dist/` to the root of a Static Space,
along with `SPACE_README.md` renamed to `README.md`. This includes the exported
model. No Python service, Node service, GPU, cross-origin isolation headers, or
build step is required on the host. Nothing is uploaded by the build command.
The built site has been tested locally; deployment to Spaces is a separate step.

References: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html),
[Hugging Face tokenizers.js](https://github.com/huggingface/tokenizers.js),
[Static HTML Spaces](https://huggingface.co/docs/hub/spaces-sdks-static).

## INT8 comparison variants

From the Python repository root:

```sh
uv run --with onnx==1.23.0 --with onnxruntime==1.30.0 python browser/scripts/quantize_onnx.py browser/public/model --output browser/public/model/quantized
```

This creates five variant directories. `embedding-int8` is the Node/browser
default; `public/model/model.onnx` remains the FP32 comparison source. Each directory works as the model-directory argument to the Node CLI.

| Directory | Token embedding table | Transformer linear weights / computation |
| --- | --- | --- |
| `embedding-int8` | Row-wise symmetric INT8 | FP32 / FP32 |
| `blocks-int8` | FP32 | Per-channel INT8 / dynamic UINT8×INT8 |
| `embedding-blocks-int8` | Row-wise symmetric INT8 | Per-channel INT8 / dynamic UINT8×INT8 |
| `blocks-weight-int8` | FP32 | Per-channel INT8 storage / FP32 |
| `embedding-blocks-weight-int8` | Row-wise symmetric INT8 | Per-channel INT8 storage / FP32 |

All three task heads remain FP32, with exactly preserved weight/bias values.
Tokenizer files and vocabulary are unchanged. LayerNorm, rotary positions,
attention score operations, softmax, and residuals remain floating point.
The block-only scope covers the seven blocks' QKV, attention output, and MLP
linear weights, shared between prefix and candidate branches. There are 53
linear applications in the exported graph: the final prefix layer only needs
QKV, so its unused output projection and MLP are removed during export.

Weight-only variants explicitly restore weights using Cast and Mul, avoiding
Q/DQ matrix fusion. These compress downloads, **not necessarily runtime memory
or computation**: a runtime can restore the full FP32 matrices during session
initialization. Dynamic variants also quantize activations at runtime and can
change predictions substantially; measure them on your own tasks before use.

From this directory, compare a JSON array of `{request, target, dataset,
case_id, decision_id}` examples (targets aligned to candidates):

```sh
node scripts/compare_quantization.js /path/to/cases.json public/model /path/to/comparison.json
node scripts/compare_quantization.js /path/to/cases.json public/model /path/to/weight-only.json fp32,blocks-weight-int8,embedding-blocks-weight-int8
```

The comparison measures candidate-at-a-time inference, matching the browser
runtime. Supply requests rendered exactly as in training when measuring trained
model quality. It records hashes, file sizes, logit/probability changes, top-1
agreement, Noul yes-probability changes, normalized Score changes, target cross
entropy, and rough latency. Target argmax accuracy is only a supplementary
metric when labels are soft distributions.

To compare Node with browser WASM on the export's synthetic parity fixtures:

```sh
node scripts/prepare-quantized-smoke.js
npm run build
# Use the dev server (which exposes comparison assets), run an example, then:
playwright-cli run-code --filename=scripts/check-quantized-browser.js
```

The browser script reports `withinTolerance` at an absolute probability
difference of 1e-4; `false` is a measured cross-runtime discrepancy, not a pass.
Quantized operator behavior can differ across backends. The production build includes only the selected model. Use `npm run dev` when
running the optional all-variant browser comparison.


## English web examples and input controls

The UI includes 35 editable demonstrations covering email, paraphrases,
entailment, rules, news topics, entity types, and document relevance.
`src/examples.json` contains only instructions, context, and criteria. Examples
are editable starting points for trying each decision type.

Use **Load model** to prepare the model before running a decision, or run an
example to load it automatically. File progress shows received bytes and download
completion; initialization is shown separately. Unknown download sizes remain
indeterminate. A loaded model is reused across decisions, and failed downloads
can be retried. **Reset example** restores the selected example's inputs.

Context is edited in ordinary text fields (for example Subject and Body,
Premise and Hypothesis, or Query and Document). No JSON editing is required.
Yes/No uses standard criteria, with optional custom meanings in a disclosure.
Choice accepts one option per line, either a label or `id | description`.
Score accepts one `number | description` per line, preserving explicit numeric
values. Results show the selected answer, probabilities, or the expected score
on its original scale. Score documents are independent examples; the score
levels are criteria, not competing documents.

The UI fixes `instruction_state` order and provides no system prompt field.
`renderDecision` matches the native training renderer: context is serialized as
JSON, Noul includes both criterion meanings in its context envelope, and
candidates receive `Candidate: id: description`. Native Python renderer fixtures
cover all three types. The Node low-level `predict` function remains available
for already-rendered inputs.

Request JSON (the actual decision sent to the worker), rendered model input,
and response JSON are pretty-printed in collapsed disclosures. The persistent
model status reports the actual ONNX byte length after the inference session
has initialized, e.g. `Model loaded · 29.0 MB · Ready on your device`. This number
excludes tokenizer and WASM assets.

`npm test` checks input parsing, native renderer parity, and inference parity. `scripts/check-browser.js` checks all
three task UIs, default Yes/No criteria, invalid score levels, request/response
disclosures, mobile overflow, and a single model load across repeated decisions.
`scripts/check-model-loading.js` checks preload, failed-download recovery, model
reuse, and resetting edited inputs. Shared React controls live in `src/ui.jsx`;
the inference worker stays independent of the interface.
