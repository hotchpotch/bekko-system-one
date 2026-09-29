# Bekko browser inference

A React interface for running Choice, Noul (yes/no), and Score locally with
ONNX Runtime. Select a model, choose an example, and run it. Inputs stay on your
device; model files are downloaded from the public Hugging Face Hub.

## Run

Requires Node.js 22 or later. From this directory:

```sh
npm ci
npm run dev
```

The default server binds to localhost. Build a static site with:

```sh
npm run build
npm run preview
```

Serve all of `dist/` over HTTP(S), not `file://`. No inference server is needed.
Model weights are fetched on demand, not included in the static build.
The ONNX Runtime WASM binary is fetched from jsDelivr at the exact pinned
`onnxruntime-web` version, without authentication. This also works when the
static app is hosted in a private Space. The first
run needs network access to Hugging Face; subsequent requests depend on normal
browser HTTP caching. The current page reuses its loaded inference session.

## Models

Models are hosted in [hotchpotch/tmp-BS1-onnx](https://huggingface.co/hotchpotch/tmp-BS1-onnx).
`src/models.js` defines the available models and pins the Hub revision.

| Model | ONNX download | Availability |
| --- | ---: | --- |
| 17M | 29.0 MB | Available |
| 68M | 196.3 MB | Available |
| 400M | — | Coming soon |

Both available models use INT8 token embeddings and FP32 Transformer blocks and
heads. Each directory contains `model_int8.onnx`, `manifest.json`,
`tokenizer.json`, and `tokenizer_config.json`. The manifest identifies the model
file, hashes, quantization policy, task heads, and token budgets.

Use **Load model** to preload, or run an example to load automatically.
A compact progress bar shows received bytes; initialization is displayed separately.
Switching model or execution device releases the current worker and loads the
new session on the next run. Failed downloads can be retried.

## Inference device

Choose **CPU** or **WebGPU**. WebGPU is selected initially when an adapter is
available; otherwise CPU is selected. WebGPU requires HTTPS or localhost
and a usable GPU adapter. The interface explains when it is unavailable.
Unsupported GPU operations can use CPU assistance. If WebGPU fails, select CPU
and retry. Speed depends on your device and input length.

The runtime uses a Web Worker and scores candidates sequentially. Query and
candidate limits are read from the model manifest; long inputs are truncated.
Short inputs are best for interactive use.

## Inputs and results

Choose a decision type on the left and search for an example on the right.
Examples are editable. **Reset example** restores its inputs. Run buttons are
available above and below the input fields.

- **Yes / No:** edit the meanings of both answers directly in the form.
- **Choice:** enter one option per line, as a name or `id | description`.
- **Score:** enter one `number | description` per line. The result is a
  probability-weighted score and can fall between levels.

Context uses ordinary text fields. Results show probabilities and the selected
execution device. Input JSON, rendered model input, and Output JSON are
available in expandable panels. There is no system prompt field.

## Export your own checkpoint

From the Python repository root:

```sh
uv run --with onnx==1.23.0 --with onnxscript==0.7.2 python browser/scripts/export_onnx.py /path/to/checkpoint --output /path/to/checkpoint/onnx
uv run --with onnx==1.23.0 --with onnxruntime==1.30.0 python browser/scripts/quantize_onnx.py /path/to/checkpoint/onnx --output /path/to/checkpoint/onnx --embedding-only
node browser/scripts/verify-export.js /path/to/checkpoint/onnx
```

The checkpoint must contain `0_SharedPrefix/` and `1_DecisionHeads/`.
The exporter supports full-weight ModernBERT shared-prefix models with balanced
query truncation, no LoRA or task markers, and three scalar decision heads.
It preserves the shared-prefix attention structure and checks Python parity.
The quantizer writes `model_int8.onnx` and updates the manifest; the FP32
`model.onnx` remains available for validation. The tokenizer is not quantized.

To publish a runtime directory, include only the selected ONNX file, manifest,
and tokenizer files, then add its URL to `src/models.js`. No checkpoint or
training data is required for browser inference.

## Node API

Use a local export directory from this directory:

```sh
npm run infer -- examples/choice.json /path/to/checkpoint/onnx
```

```js
import { loadModel } from './src/node.js';
import { renderDecision } from './src/decision.js';
const model = await loadModel('/path/to/checkpoint/onnx');
try {
  const result = await model.predict(renderDecision({
    task: 'choice', instruction: 'Which topic describes this article?',
    state: { article: 'The player won the tennis championship.' },
    criteria: [
      { id: 'sports', description: 'Sports news.' },
      { id: 'business', description: 'Business news.' },
    ],
  }));
  console.log(result);
} finally { await model.release(); }
```

Node uses CPU and reads the ONNX filename from the manifest. The low-level API
also accepts already-rendered requests. Choice returns `selected_id`; Noul
returns `probability_yes`; Score returns `score` and `normalized_score`. All
include candidate probabilities. Noul requires `true`/`false` or `yes`/`no` IDs;
Score requires distinct finite numeric values.

## Verification

From a fresh checkout, run the checks that do not need model files:

```sh
node --test test/context.test.js test/decision.test.js test/download.test.js test/inference-time.test.js test/runtime-assets.test.js test/runtime.test.js
npm run build
```

`npm test` also checks runtime parity and requires generated model files.
Local parity tests require the FP32 export in `public/model/` and its
embedding-INT8 variant in `public/model/quantized/embedding-int8/`.
For any other export directory, use `scripts/verify-export.js` as shown above.

Open the preview in Playwright CLI, then run:

```sh
playwright-cli run-code --filename=scripts/check-browser.js
playwright-cli run-code --filename=scripts/check-model-loading.js
playwright-cli run-code --filename=scripts/check-runtime.js
playwright-cli run-code --filename=scripts/check-hub-models.js
```

These check forms, model reuse, error recovery, and CPU/WebGPU behavior.
GPU unavailability is reported explicitly.

## Hugging Face Static Space

Export a ready-to-upload directory:

```sh
npm run export:space
```

This builds the app and recreates `export/space/` with static assets and the Space
README. Model files and local source files are not included.

With the Hugging Face CLI (`hf`) installed and authenticated via `hf auth login`
or `HF_TOKEN`, export and deploy in one command:

```sh
npm run deploy:space -- YOUR_ACCOUNT/YOUR_SPACE
```

The script creates a **private Static Space**, or verifies that an existing Space
is private and static before uploading. It never switches visibility. Always pass
your own Space ID; omitting it uses the maintainer's default Space. Uploaded assets use hashed
filenames; previous assets are retained so existing browser sessions keep working.

Check deployment status with:

```sh
hf spaces wait YOUR_ACCOUNT/YOUR_SPACE --timeout 5m
```

The app downloads models directly from the pinned public Hub revision. No Python
service, GPU server, or host build step is required. A private Space does not
change the visibility of the model repository. Never include access tokens in
frontend files.

References: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html),
[WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html),
[Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static).
