import { copyFile, mkdir, readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { DEFAULT_MODEL_PATH } from './src/default-model.js';

const root = fileURLToPath(new URL('.', import.meta.url));
const assets = ['model.onnx', 'manifest.json', 'tokenizer.json', 'tokenizer_config.json', 'parity.json'];

export default defineConfig(({ command }) => ({
  base: './',
  publicDir: command === 'build' ? false : 'public',
  build: { target: 'es2022' },
  plugins: [{
    name: 'selected-model-assets',
    apply: 'build',
    async buildStart() {
      const source = resolve(root, 'public', DEFAULT_MODEL_PATH);
      for (const name of assets) await stat(resolve(source, name));
      const manifest = JSON.parse(await readFile(resolve(source, 'manifest.json'), 'utf8'));
      if (manifest.quantization?.embedding !== 'rowwise_symmetric_int8' || manifest.quantization.blocks !== 'fp32' || manifest.quantization.heads !== 'fp32') {
        throw Error('The default model must use INT8 embeddings with FP32 blocks and heads');
      }
    },
    async closeBundle() {
      const target = resolve(root, 'dist', DEFAULT_MODEL_PATH);
      await mkdir(target, { recursive: true });
      for (const name of assets) await copyFile(resolve(root, 'public', DEFAULT_MODEL_PATH, name), resolve(target, name));
    },
  }],
}));
