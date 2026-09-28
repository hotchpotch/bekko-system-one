import { readFile } from 'node:fs/promises';
import { loadModel } from './node.js';

try {
  if (!process.argv[2]) throw Error('Usage: npm run infer -- request.json [model-directory]');
  const request = JSON.parse(await readFile(process.argv[2], 'utf8'));
  const model = await loadModel(process.argv[3]);
  try { console.log(JSON.stringify(await model.predict(request), null, 2)); }
  finally { await model.release(); }
} catch (error) { console.error(error.message); process.exitCode = 1; }
