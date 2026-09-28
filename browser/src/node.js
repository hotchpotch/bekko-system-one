import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';
import { DEFAULT_MODEL_PATH } from './default-model.js';
import { createTokenizer, predict } from './core.js';

export async function loadModel(directory = new URL(`../public/${DEFAULT_MODEL_PATH}`, import.meta.url)) {
  const base = directory instanceof URL ? directory : `${resolve(directory)}/`;
  const path = name => base instanceof URL ? new URL(name, base) : base + name;
  const json = async name => JSON.parse(await readFile(path(name), 'utf8'));
  const [manifest, config, data] = await Promise.all(['manifest.json', 'tokenizer_config.json', 'tokenizer.json'].map(json));
  const tokenizer = createTokenizer(data, config);
  const session = await ort.InferenceSession.create(new Uint8Array(await readFile(path('model.onnx'))), { executionProviders: ['cpu'], intraOpNumThreads: 4 });
  const runtime = { manifest, tokenizer, session, ort };
  return { ...runtime, predict: request => predict(request, runtime), release: () => session.release() };
}
