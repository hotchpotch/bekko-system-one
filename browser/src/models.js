const releases = [
  ['17m', '17M', 29, 'cf92c2f76214e764b9c4e3054125cdb146b0732e'],
  ['68m', '68M', 196, '995dda388ac0f6735d8dc3cd4932d380c6fe06fc'],
  ['400m', '400M', 1426, '93c993e2fbf8ed480b49d410fe52347062616c6e'],
];
export const DEFAULT_MODEL_ID = '17m';
export const MODELS = releases.map(([value, size, megabytes, revision]) => {
  const repository = `hotchpotch/bekko-system-one-v0-${value}`;
  return {
    value,
    name: `Bekko-s1-v0-${size}`,
    label: `Bekko-s1-v0-${size} (size ${megabytes} MB)`,
    repository,
    revision,
    base: import.meta.env?.DEV
      ? `/__hf_models/${value}/`
      : `https://huggingface.co/${repository}/resolve/${revision}/onnx_browser/`,
  };
});
