export const MODEL_REPOSITORY = "hotchpotch/tmp-BS1-onnx";
export const MODEL_REVISION = "b07e1b7ff136ee97cb0f022ddd4aebd6e16f7d31";
const base = `https://huggingface.co/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}`;
export const DEFAULT_MODEL_ID = "17m";
export const MODELS = [
  {
    value: "17m",
    name: "Bekko 17M",
    label: "Bekko 17M · 29 MB",
    base: `${base}/17m/`,
  },
  {
    value: "68m",
    name: "Bekko 68M",
    label: "Bekko 68M · 196 MB",
    base: `${base}/68m/`,
  },
  {
    value: "400m",
    name: "Bekko 400M",
    label: "Bekko 400M · Coming soon",
    disabled: true,
  },
];
