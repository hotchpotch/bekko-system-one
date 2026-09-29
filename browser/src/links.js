import { MODEL_REPOSITORY, MODEL_REVISION } from './models.js';

// Replace placeholder URLs as public releases and documentation become available.
const models = `https://huggingface.co/${MODEL_REPOSITORY}/tree/${MODEL_REVISION}`;
export const RESOURCE_LINKS = [
  { label: '🤗 Bekko-s1-v0-17M', href: `${models}/17m` },
  { label: '🤗 Bekko-s1-v0-68M', href: `${models}/68m` },
  { label: '🤗 Bekko-s1-v0-400M', href: 'https://example.com/bekko/models/400m', placeholder: true },
  { label: 'Technical article', href: 'https://example.com/bekko/article', placeholder: true },
  { label: 'Source code', href: 'https://example.com/bekko/source', placeholder: true },
];
