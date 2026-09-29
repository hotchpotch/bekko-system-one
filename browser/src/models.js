const releases = [
  ['17m', 29023607, 'cf92c2f76214e764b9c4e3054125cdb146b0732e'],
  ['68m', 196342909, '995dda388ac0f6735d8dc3cd4932d380c6fe06fc'],
  ['400m', 1426199724, '93c993e2fbf8ed480b49d410fe52347062616c6e'],
];
export const DEFAULT_MODEL_ID = '17m';
export const MODELS = releases.map(([value, bytes, revision]) => {
  const repository = `hotchpotch/bekko-system-one-v0-${value}`;
  return {
    value,
    name: `bekko-s1-v1-${value}`,
    label: `bekko-s1-v1-${value} (${bytes >= 1e9 ? "1.4 GB download" : `size ${Math.round(bytes / 1e6)} MB`})`,
    repository,
    bytes,
    revision,
    base: import.meta.env?.DEV
      ? `/__hf_models/${value}/`
      : `https://huggingface.co/${repository}/resolve/${revision}/onnx_browser/`,
  };
});
