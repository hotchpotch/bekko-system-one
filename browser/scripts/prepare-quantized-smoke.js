// node scripts/prepare-quantized-smoke.js [model-directory]
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { loadModel } from '../src/node.js';
const directory = resolve(process.argv[2] || 'public/model');
const fixtures = JSON.parse(await readFile(resolve(directory, 'parity.json')));
const variants = JSON.parse(await readFile(resolve(directory, 'quantized/variants.json')));
const results = {};
for (const name of Object.keys(variants)) {
  const model = await loadModel(resolve(directory, 'quantized', name));
  try {
    results[name] = [];
    for (const fixture of fixtures) results[name].push({ request: fixture.request, expected: await model.predict(fixture.request) });
  } finally { await model.release(); }
}
await writeFile(resolve(directory, 'quantized/browser-smoke.json'), JSON.stringify(results));
