# Bekko System One in the browser

Run Noul (yes/no), Choice, and Score decisions locally with a React interface and
ONNX Runtime. Choose a model, load an example, and edit the inputs to try your own
use case. Inference runs on your device; input text is not sent to an inference
server.

This directory also provides a Node.js CPU API, checkpoint export tools, and a
static-site deployment script.

## Quick start

Requires Node.js 22 or later. From this directory:

```sh
npm ci
cp .env.example .env.local
# Set HF_TOKEN in .env.local to a token with read access to the model repositories.
npm run dev
```

Alternatively, supply `HF_TOKEN` through your shell environment at startup. Do not
use a `VITE_` prefix: the token is server-side only. `.env.local` is Git-ignored.
Restart Vite after changing it.

The local development server streams the four permitted model assets through a
same-origin `/__hf_models/` endpoint. It authenticates to Hugging Face without
exposing the token to browser JavaScript or forwarding it to CDN redirects. This
is a download proxy, not an inference server; input text stays in the browser.
Only expose this local server to trusted users, since they can download the models
using its access.

Open the URL printed by Vite. The development server binds to localhost.
Select a model and a decision type, then choose an example and click **Run
decision**. The model loads automatically on the first run; **Load Model** lets
you load it in advance.

To build and preview the static app:

```sh
npm run build
npm run preview
```

The authenticated model proxy runs only in the development server, not in
`npm run preview` or `dist/`. Static hosting requires publicly accessible models.
Deploy the contents of `dist/` to a static host once that requirement is met. Serve the app over HTTP(S), not
`file://`. No Python service or inference server is required.

## Models and downloads

The demo uses the `onnx_browser/` exports in the release repositories:

- [17M](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m)
- [68M](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m)
- [400M](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m)

These repositories currently require authenticated access. Use the local development
server with `HF_TOKEN` while they are private.
[`src/models.js`](src/models.js) defines the model URLs and pins a specific Hub
revision so deployments use a consistent set of files.

| Model | ONNX file size | Status |
| --- | ---: | --- |
| bekko-system-one-v0-17m | 29.0 MB | Available |
| bekko-system-one-v0-68m | 196.3 MB | Available |
| bekko-system-one-v0-400m | 1,426.2 MB | Available; requires substantial memory |

Sizes are decimal MB for the ONNX file alone. Tokenizer files and the inference
runtime are additional downloads. All three models use **INT8 token
embeddings; Transformer blocks and prediction heads remain FP32**. The tokenizer
is not quantized.

Each model directory contains:

```text
model.onnx
manifest.json
tokenizer.json
tokenizer_config.json
```

The manifest specifies the model filename, hashes, quantization policy, task-head
order, and token budgets. Model files are fetched on demand and are not bundled
into `dist/`.

The app also downloads the ONNX Runtime WASM binary from jsDelivr at the exact
pinned `onnxruntime-web` version. The runtime module is bundled into the worker;
the WASM download needs no authentication, including when the app is hosted in a
private Space. Initial use requires network access to both Hugging Face and
jsDelivr. Runtime downloads use normal browser HTTP caching. The authenticated local model
proxy returns `Cache-Control: private, no-store`; offline availability is not
guaranteed.

## CPU and WebGPU

The browser offers two execution devices:

- **CPU:** ONNX Runtime Web's WASM backend, using one thread.
- **WebGPU:** GPU execution with CPU assistance for unsupported operations.
  Requires HTTPS or localhost and a usable WebGPU adapter.

WebGPU is selected initially when an adapter is available. Otherwise, the app
selects CPU. Adapter availability does not guarantee that every model or input
will fit on a device. If WebGPU fails, select CPU and retry. Performance depends
on the device, model, input lengths, and candidate count.

Inference runs in a Web Worker. The page reuses its loaded session for subsequent
decisions. Switching model or execution device discards the current worker; the
next load or run creates a new session.

## Shared-prefix inference

The instruction and context form a common **prefix**. Each candidate is a
separate branch that attends to this prefix. The exported ONNX model preserves
that structure: it computes the prefix's keys and values at each Transformer
layer, then shares them across the candidate branches.

The runtime passes candidates together in a batch instead of running the entire
model separately for each candidate. For example, five short Score levels can
share one prefix computation in one `session.run()` call. This avoids computing
the same instruction and context five times and reduces runtime-call overhead.
The browser and Node API use the same batching logic.

This optimization uses the existing **single ONNX file**. It requires no extra
weights, increases neither model size nor download size, and applies to Noul,
Choice, and Score on both CPU and WebGPU. Candidates remain independent: they
attend to the shared prefix, not to each other. Small floating-point differences
from sequential execution are possible.

Long inputs are divided into smaller candidate batches to limit attention growth.
The batch size is based on the prefix length and the longest padded candidate,
with a budget of 1,048,576 candidate attention pairs:

```text
candidates per batch × candidate length × (prefix length + candidate length)
```

At least one candidate is processed per run. This is a batching heuristic, not a
guaranteed limit on total memory: model weights, prefix attention, and runtime
buffers also consume memory. The prefix is recomputed for each batch. Candidate
logits are collected in their original order, and probabilities are calculated
across all candidates after every batch finishes.

Prefix keys and values are shared **within a model run**, not cached between
separate decisions. Repeating a decision reuses the loaded model but recomputes
the prefix. Longer prefixes and more candidates offer more opportunity to save
work; a fixed speedup is not guaranteed.

The UI's **Inference** time measures time spent awaiting `session.run()`, summed
across batches. It excludes model downloads, initialization, tokenization, input
tensor preparation, and result rendering.

## Inputs and results

Choose a decision type on the left and an example on the right. Examples are
editable; **Reset example** restores the selected example's inputs.

| Decision | Input | Result |
| --- | --- | --- |
| Noul (Yes/No) | A question and the meanings of Yes and No | Yes probability and both answer probabilities |
| Choice | One option per line: a name or `id | description` | The most probable option and all option probabilities |
| Score | One level per line: `number | description` | A probability-weighted score, which can fall between levels |

Context is an editable set of key/value fields. Keys must be unique and nonempty;
use **Add context field** to add another entry. Both keys and values are sent to
the local model. There is no system prompt field in the UI.

Results preserve candidate order. Noul and Choice highlight the most probable
candidate. Score highlights the level nearest the weighted score, which may
differ from the most probable level; ties use the first matching level in input
order. Expand **Input JSON** or **Output JSON** to inspect the decision, rendered
model input, and result.

Requests accept 2–64 candidates. Noul requires exactly two. Token budgets come
from the model manifest, and overlong inputs are truncated. Keep relevant
information within those budgets; short inputs are best for interactive use.

## Node.js CPU API

Use a local ONNX export directory containing the four runtime files listed above.
From this directory:

```sh
npm run infer -- examples/choice.json /path/to/checkpoint/onnx
```

The CLI example is an already-rendered model request. For structured context and
criteria, use `renderDecision`:

```js
import { loadModel } from './src/node.js';
import { renderDecision } from './src/decision.js';

const model = await loadModel('/path/to/checkpoint/onnx');
try {
  const result = await model.predict(renderDecision({
    task: 'choice',
    instruction: 'Which topic describes this article?',
    state: { article: 'The player won the tennis championship.' },
    criteria: [
      { id: 'sports', description: 'Sports news.' },
      { id: 'business', description: 'Business news.' },
    ],
  }));
  console.log(result);
} finally {
  await model.release();
}
```

Node uses `onnxruntime-node` on CPU and reads the ONNX filename from the manifest.
Pass the directory explicitly; the default path is intended for local parity
fixtures. `predict` also accepts an already-rendered request.

Choice returns `selected_id`; Noul returns `probability_yes`; Score returns
`score` and `normalized_score`. All include `probabilities` keyed by candidate
ID. Noul requires `true`/`false` or `yes`/`no` IDs. Score requires distinct finite
numeric values; `normalized_score` maps the minimum and maximum supplied values
to 0 and 1.

## Export a checkpoint

Set up the Python environment as described in the [repository README](../README.md).
Then run these commands from the repository root:

```sh
uv run --with onnx==1.23.0 --with onnxscript==0.7.2 \
  python browser/scripts/export_onnx.py /path/to/checkpoint \
  --output /path/to/checkpoint/onnx

uv run --with onnx==1.23.0 --with onnxruntime==1.30.0 \
  python browser/scripts/quantize_onnx.py /path/to/checkpoint/onnx \
  --output /path/to/checkpoint/onnx --embedding-only

node browser/scripts/verify-export.js /path/to/checkpoint/onnx
```

The checkpoint must contain `0_SharedPrefix/` and `1_DecisionHeads/`. The exporter
supports full-weight ModernBERT shared-prefix models with balanced query
truncation and three scalar decision heads. LoRA, task markers, and Choice
interaction heads are not supported by this exporter.

Export preserves shared-prefix attention and produces Python parity fixtures.
Quantization writes `model_int8.onnx` and updates the manifest, while retaining
the FP32 `model.onnx` for verification. `verify-export.js` checks FP32 logits
against the fixtures and bounds prediction differences after quantization.

Publish only the selected ONNX file, manifest, and tokenizer files for inference.
Add the model URL and pinned revision to [`src/models.js`](src/models.js).
Training checkpoints, training data, and parity fixtures are not needed by the
browser app. Custom model hosting must allow cross-origin downloads from the app.

## Verification

From this directory, the following checks work without local model files:

```sh
node --test \
  test/batched-inference.test.js \
  test/context.test.js \
  test/decision.test.js \
  test/download.test.js \
  test/hf-model-proxy.test.js \
  test/inference-time.test.js \
  test/runtime-assets.test.js \
  test/runtime.test.js
npm run build
```

`npm test` additionally runs model parity checks. These require an FP32 export in
`public/model/` and its embedding-INT8 variant in
`public/model/quantized/embedding-int8/`, generated from the same checkpoint.
For an export elsewhere, use `scripts/verify-export.js` with its directory.

For UI checks with private models, start `npm run dev` with `HF_TOKEN`, open its URL using
Playwright CLI, and run:

```sh
playwright-cli run-code --filename=scripts/check-browser.js
playwright-cli run-code --filename=scripts/check-model-loading.js
playwright-cli run-code --filename=scripts/check-runtime.js
playwright-cli run-code --filename=scripts/check-hub-models.js
```

These checks download the configured models through the development server and cover forms, model reuse,
error recovery, and backend selection. `check-runtime.js` compares CPU and WebGPU
results when an adapter is available, and explicitly reports when it is not.
`check-hub-models.js` exercises all three decision types on the 17M and 68M models
using CPU. WebGPU validation should also be performed on the intended hardware.

## Deploy to a private Hugging Face Space

The authenticated model-download proxy is available only through `npm run dev`.
A static build uses direct Hub URLs and contains no token. It cannot load these
private models as configured. Do not deploy this configuration until the model
repositories are publicly accessible or a separate authenticated download service
is provided. Never embed `HF_TOKEN` in a static build.

Export a ready-to-upload static directory:

```sh
npm run export:space
```

This rebuilds the app and recreates `export/space/` with the contents of `dist/`
and the Space README. Local model files, training artifacts, and application
source files are not included.

With the Hugging Face CLI (`hf`) installed and authenticated using `hf auth login`
or `HF_TOKEN`, export and deploy in one command:

```sh
npm run deploy:space -- YOUR_ACCOUNT/YOUR_SPACE
hf spaces wait YOUR_ACCOUNT/YOUR_SPACE --timeout 5m
```

Always pass your own Space ID. Omitting it targets the maintainer's default
Space. The script creates a **private Static Space**, or checks that an existing
Space is private and static before uploading. It never changes visibility.
Previously uploaded assets are retained so existing browser sessions can finish
using their original build.

The Space serves static assets; inference still runs in the visitor's browser.
After the models are publicly accessible, static deployments download them
directly from the pinned Hub revisions. Space visibility and model repository
visibility are independent. Deployment credentials belong
in the CLI environment, never in frontend files.

## References

- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [Hugging Face Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
