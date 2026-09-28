// node scripts/compare_quantization.js cases.json model-directory output.json
import { readFile, writeFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { loadModel } from '../src/node.js';
import { tokenize, feedsFor, interpret } from '../src/core.js';

const [casesPath, directory, output] = process.argv.slice(2);
if (!output) throw Error('Usage: node scripts/compare_quantization.js cases.json model-directory output.json');
const cases = JSON.parse(await readFile(casesPath, 'utf8'));
const names = process.argv[5]?.split(',') || ['fp32', 'embedding-int8', 'blocks-int8', 'embedding-blocks-int8'];
if (!names.includes('fp32')) throw Error('Comparison must include fp32');
const models = {}, sizes = {}, hashes = {};
const average = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const percentile = (xs, q) => xs.length ? [...xs].sort((a, b) => a - b)[Math.ceil(q * xs.length) - 1] : null;
const argmax = xs => xs.indexOf(Math.max(...xs));
const softmax = xs => { const e = xs.map(x => Math.exp(x - Math.max(...xs))); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
async function infer(runtime, tokens, task) {
  const column = runtime.manifest.tasks.indexOf(task);
  const values = [];
  for (const doc of tokens.doc_ids) {
    const result = await runtime.session.run(feedsFor({ prefix_ids: tokens.prefix_ids, doc_ids: [doc] }, runtime.ort, runtime.manifest));
    values.push(Number(result.logits.data[column]));
  }
  return values;
}
const records = [];
try {
  for (const name of names) {
    const path = resolve(directory, name === 'fp32' ? '.' : `quantized/${name}`);
    models[name] = await loadModel(path);
    sizes[name] = (await stat(resolve(path, 'model.onnx'))).size;
    hashes[name] = models[name].manifest.model_sha256;
    const tokens = tokenize(cases[0].request, models[name].tokenizer, models[name].manifest);
    for (let i = 0; i < 3; i++) await infer(models[name], tokens, cases[0].request.task);
  }
  for (const [index, sample] of cases.entries()) {
    const { request, target } = sample;
    const tokens = tokenize(request, models.fp32.tokenizer, models.fp32.manifest);
    const row = { dataset: sample.dataset, case_id: sample.case_id, decision_id: sample.decision_id, task: request.task, layout: request.layout, target, lengths: { prefix: tokens.prefix_ids[0].length, documents: tokens.doc_ids.map(x => x.length) }, variants: {} };
    // Rotate execution order to reduce systematic warm-up/order effects.
    for (let offset = 0; offset < names.length; offset++) {
      const name = names[(index + offset) % names.length];
      const start = performance.now();
      const logits = await infer(models[name], tokens, request.task);
      const milliseconds = performance.now() - start;
      const p = softmax(logits);
      row.variants[name] = { logits, p, result: interpret(request, logits), milliseconds, cross_entropy: -target.reduce((sum, t, i) => sum + t * Math.log(Math.max(p[i], 1e-30)), 0), correct: argmax(p) === argmax(target) };
    }
    records.push(row);
    if ((index + 1) % 30 === 0) console.log(`${index + 1}/${cases.length}`);
  }
  const summary = {};
  for (const name of names) {
    summary[name] = { bytes: sizes[name], reduction_percent: 100 * (1 - sizes[name] / sizes.fp32), model_sha256: hashes[name] };
    for (const task of ['all', 'choice', 'noul', 'score']) {
      const subset = records.filter(row => task === 'all' || row.task === task);
      const pd = [], ld = [], score = [], scoreRaw = [], yes = [];
      let flips = 0;
      for (const row of subset) {
        const base = row.variants.fp32, current = row.variants[name];
        pd.push(...base.p.map((p, i) => Math.abs(p - current.p[i])));
        ld.push(...base.logits.map((p, i) => Math.abs(p - current.logits[i])));
        flips += Number(argmax(base.p) !== argmax(current.p));
        if (row.task === 'score') { score.push(Math.abs(base.result.normalized_score - current.result.normalized_score)); scoreRaw.push(Math.abs(base.result.score - current.result.score)); }
        if (row.task === 'noul') yes.push(Math.abs(base.result.probability_yes - current.result.probability_yes));
      }
      summary[name][task] = { judgments: subset.length, top1_flips: flips, top1_agreement: 1 - flips / subset.length, probability_abs_mean: average(pd), probability_abs_p95: percentile(pd, .95), probability_abs_max: Math.max(...pd), logit_abs_mean: average(ld), logit_abs_max: Math.max(...ld), normalized_score_abs_mean: average(score), normalized_score_abs_max: score.length ? Math.max(...score) : null, raw_score_abs_mean: average(scoreRaw), probability_yes_abs_mean: average(yes), probability_yes_abs_max: yes.length ? Math.max(...yes) : null, cross_entropy: average(subset.map(row => row.variants[name].cross_entropy)), top1_target_accuracy: average(subset.map(row => Number(row.variants[name].correct))), median_ms: percentile(subset.map(row => row.variants[name].milliseconds), .5) };
    }
  }
  await writeFile(output, JSON.stringify({ node: process.version, runtime: 'onnxruntime-node CPU, 4 intra-op threads, sequential candidates', summary, records }, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally { for (const runtime of Object.values(models)) await runtime.release(); }
