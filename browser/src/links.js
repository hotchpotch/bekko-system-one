import { MODELS } from './models.js';

export const RESOURCE_LINKS = [
  ...MODELS.map(model => ({ kind: 'model', label: `🤗 ${model.name}`, href: `https://huggingface.co/${model.repository}` })),
  { label: 'Technical article', href: 'https://huggingface.co/blog/hotchpotch/bekko-system-one-v0-release/' },
  { label: 'Source code', href: 'https://github.com/hotchpotch/bekko-system-one' },
];
