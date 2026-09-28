import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadModel } from '../src/node.js';
import { tokenize, feedsFor, interpret, validate } from '../src/core.js';

test('Python token IDs and all head logits match Node CPU, including dynamic shapes', async () => {
  const fixtures = JSON.parse(await readFile(new URL('../public/model/parity.json', import.meta.url)));
  const model = await loadModel(new URL('../public/model/', import.meta.url));
  try {
    const truncation = JSON.parse(await readFile(new URL('../public/model/tokenization-parity.json', import.meta.url)));
    for (const fixture of truncation) {
      assert.deepEqual(tokenize(fixture.request, model.tokenizer, model.manifest), { prefix_ids: fixture.prefix_ids, doc_ids: fixture.doc_ids });
    }
    for (const fixture of fixtures) {
      const tokens = tokenize(fixture.request, model.tokenizer, model.manifest);
      assert.deepEqual(tokens.prefix_ids, fixture.prefix_ids);
      assert.deepEqual(tokens.doc_ids, fixture.doc_ids);
      const { logits } = await model.session.run(feedsFor(tokens, model.ort, model.manifest));
      let maxError = 0;
      fixture.logits.flat().forEach((value, i) => {
        maxError = Math.max(maxError, Math.abs(value - logits.data[i]));
        assert.ok(Math.abs(value - logits.data[i]) < 1e-4, `logit ${i}: ${value} vs ${logits.data[i]}`);
      });
      const expected = interpret(fixture.request, fixture.logits.map(row => row[model.manifest.tasks.indexOf(fixture.request.task)]));
      const actual = await model.predict(fixture.request);
      for (const id of Object.keys(expected.probabilities)) assert.ok(Math.abs(expected.probabilities[id] - actual.probabilities[id]) < 1e-5);
      console.log(`${fixture.request.task}: maximum logit error ${maxError}`);
    }
  } finally { await model.release(); }
});

test('typed interpretation respects IDs and explicit numeric values', () => {
  const base = { instruction: 'Test', state: '' };
  assert.equal(interpret({ ...base, task: 'noul', candidates: [{ id: 'no', text: 'No' }, { id: 'yes', text: 'Yes' }] }, [0, 0]).probability_yes, 0.5);
  const result = interpret({ ...base, task: 'score', candidates: [{ id: 'low', text: 'Low', value: -10 }, { id: 'high', text: 'High', value: 30 }] }, [0, 0]);
  assert.equal(result.score, 10); assert.equal(result.normalized_score, 0.5);
  assert.throws(() => validate({ ...base, task: 'noul', candidates: [{ id: 'a', text: 'Yes' }, { id: 'b', text: 'No' }] }));
  assert.throws(() => validate({ ...base, task: 'score', candidates: [{ id: 'a', text: '1' }, { id: 'b', text: '2' }] }));
});


test('default embedding INT8 model retains FP32 blocks/heads and bounded prediction drift', async () => {
  const model = await loadModel();
  const reference = await loadModel(new URL('../public/model/', import.meta.url));
  try {
    assert.equal(model.manifest.quantization.embedding, 'rowwise_symmetric_int8');
    assert.equal(model.manifest.quantization.blocks, 'fp32');
    assert.equal(model.manifest.quantization.heads, 'fp32');
    assert.equal(model.manifest.source_model_sha256, reference.manifest.model_sha256);
    const fixtures = JSON.parse(await readFile(new URL('../public/model/parity.json', import.meta.url)));
    for (const { request } of fixtures) {
      const actual = await model.predict(request);
      const expected = await reference.predict(request);
      for (const id of Object.keys(expected.probabilities)) {
        assert.ok(Math.abs(expected.probabilities[id] - actual.probabilities[id]) < 0.03);
      }
    }
  } finally { await model.release(); await reference.release(); }
});
