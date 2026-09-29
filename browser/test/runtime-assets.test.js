import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { RUNTIME_VERSION, WASM_URL } from '../src/runtime-assets.js';

test('public WASM version matches the installed and pinned JS runtime', async () => {
  const installed = JSON.parse(await readFile(new URL('../node_modules/onnxruntime-web/package.json', import.meta.url)));
  const project = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
  assert.equal(RUNTIME_VERSION, installed.version);
  assert.equal(RUNTIME_VERSION, project.dependencies['onnxruntime-web']);
  assert.equal(new URL(WASM_URL).protocol, 'https:');
});
