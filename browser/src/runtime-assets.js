// Must match the bundled onnxruntime-web package. The public runtime needs no
// Space cookie or token, including when the app runs inside a private iframe.
export const RUNTIME_VERSION = '1.30.0';
export const WASM_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${RUNTIME_VERSION}/dist/ort-wasm-simd-threaded.asyncify.wasm`;
