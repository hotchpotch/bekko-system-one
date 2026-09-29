import { MODELS } from './models.js';

export const RESOURCE_LINKS = [
  ...MODELS.map(model => ({ kind: 'model', label: `🤗 ${model.name}`, href: `https://huggingface.co/${model.repository}` })),
  { label: 'Technical article', href: 'https://example.com/bekko/article', placeholder: true },
  { label: 'Source code', href: 'https://example.com/bekko/source', placeholder: true },
];
