export async function detectWebGPU(environment = globalThis) {
  if (!environment.isSecureContext)
    return {
      available: false,
      reason: "WebGPU needs HTTPS or localhost. CPU works on this connection.",
    };
  if (!environment.navigator?.gpu)
    return {
      available: false,
      reason: "WebGPU is not supported by this browser. Use CPU.",
    };
  try {
    const adapter = await environment.navigator.gpu.requestAdapter();
    return adapter
      ? {
          available: true,
          reason:
            "WebGPU available. Speed depends on your GPU and input length.",
        }
      : {
          available: false,
          reason: "No WebGPU adapter is available. Use CPU.",
        };
  } catch {
    return {
      available: false,
      reason: "WebGPU could not be initialized. Use CPU.",
    };
  }
}
export function executionProviders(device) {
  if (device === "cpu") return ["wasm"];
  if (device === "webgpu") return ["webgpu", "wasm"];
  throw Error(`Unknown inference device: ${device}`);
}
export const deviceLabel = (device) => (device === "webgpu" ? "WebGPU" : "CPU");
