import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjsSource from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?raw";
import { download } from "./download.js";
import { renderDecision } from "./decision.js";
import { createTokenizer, predict } from "./core.js";

ort.env.wasm.numThreads = 1;
// Bundle the runtime module: dynamic imports from classic workers can omit the
// cookies required by private static hosts. The blob needs no authenticated request.
const runtimeModuleUrl = URL.createObjectURL(new Blob([mjsSource], { type: "text/javascript" }));
ort.env.wasm.wasmPaths = { mjs: runtimeModuleUrl };
let wasmReady = false;
async function prepareWasm() {
  if (wasmReady) return;
  const response = await fetch(wasmUrl, { credentials: "same-origin" });
  if (!response.ok) throw Error(`Could not load inference runtime (HTTP ${response.status}). Reload the page and check your Space access.`);
  ort.env.wasm.wasmBinary = await response.arrayBuffer();
  wasmReady = true;
}
import { detectWebGPU, executionProviders, deviceLabel } from "./runtime.js";
let runtime;
let loadedDevice;
async function load(base, device) {
  if (device === "webgpu") {
    const support = await detectWebGPU();
    if (!support.available) throw Error(support.reason);
  }
  const asset = (name) =>
    download(new URL(name, base).href, (progress) =>
      self.postMessage({ progress: { file: name, ...progress } }),
    );
  const json = async (name) =>
    JSON.parse(new TextDecoder().decode(await asset(name)));
  const [manifest, config, data] = await Promise.all(
    ["manifest.json", "tokenizer_config.json", "tokenizer.json"].map(json),
  );
  const tokenizer = createTokenizer(data, config);
  const bytes = await asset(manifest.model_file || "model.onnx");
  self.postMessage({
    status: `Preparing model for ${deviceLabel(device)} inference…`,
  });
  await prepareWasm();
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: executionProviders(device),
  });
  self.postMessage({ loadedBytes: bytes.byteLength });
  return { manifest, tokenizer, session, ort };
}
self.onmessage = async ({ data }) => {
  try {
    const device = data.device || "cpu";
    executionProviders(device);
    if (runtime && loadedDevice !== device) {
      await runtime.session.release();
      runtime = null;
    }
    if (!runtime) {
      self.postMessage({ status: "Loading model…" });
      runtime = await load(data.base, device);
      loadedDevice = device;
    }
    if (data.type === "load") {
      self.postMessage({ ready: true });
      return;
    }
    self.postMessage({ status: `Running on ${deviceLabel(device)}…` });
    let inferenceMilliseconds;
    const result = await predict(
      data.decision ? renderDecision(data.decision) : data.request,
      runtime,
      { onInferenceTime: (elapsed) => { inferenceMilliseconds = elapsed; } },
    );
    self.postMessage({
      result,
      device,
      inferenceMilliseconds,
    });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
