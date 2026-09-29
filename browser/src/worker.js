import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { download } from "./download.js";
import { renderDecision } from "./decision.js";
import { createTokenizer, predict } from "./core.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
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
    const start = performance.now();
    const result = await predict(
      data.decision ? renderDecision(data.decision) : data.request,
      runtime,
    );
    self.postMessage({
      result,
      device,
      milliseconds: performance.now() - start,
    });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
