import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { download } from "./download.js";
import { renderDecision } from "./decision.js";
import { createTokenizer, predict } from "./core.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
let runtime;
async function load(base) {
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
  const bytes = await asset("model.onnx");
  self.postMessage({ status: "Preparing model for CPU inference…" });
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  self.postMessage({ loadedBytes: bytes.byteLength });
  return { manifest, tokenizer, session, ort };
}
self.onmessage = async ({ data }) => {
  try {
    if (!runtime) {
      self.postMessage({ status: "Loading model…" });
      runtime = await load(data.base);
    }
    if (data.type === "load") {
      self.postMessage({ ready: true });
      return;
    }
    self.postMessage({ status: "Running on CPU…" });
    const start = performance.now();
    const result = await predict(
      data.decision ? renderDecision(data.decision) : data.request,
      runtime,
    );
    self.postMessage({ result, milliseconds: performance.now() - start });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
