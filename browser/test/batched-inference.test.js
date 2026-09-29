import { test } from 'node:test';
import assert from 'node:assert/strict';
import { predict, interpret } from '../src/core.js';

const manifest = { tasks: ['choice', 'noul', 'score'], query_length: 4096, document_length: 2048, cls_token_id: 1, sep_token_id: 2, pad_token_id: 0 };
class Tensor {
  constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
}
for (const task of manifest.tasks) {
  test(`${task}: candidate rows use the correct head and preserve input order`, async () => {
    const candidates = task === 'noul'
      ? [{ id: 'no', text: '3' }, { id: 'yes', text: '7' }]
      : [{ id: 'last', text: '3', value: 10 }, { id: 'first', text: '7', value: -2 }, { id: 'middle', text: '5', value: 0 }];
    const request = { task, instruction: 'Choose', state: '', candidates };
    let runs = 0;
    const actual = await predict(request, {
      tokenizer: { encode: text => ({ ids: [Number(text) || 1] }) }, manifest, ort: { Tensor },
      session: { async run(feeds) {
        runs++;
        assert.equal(feeds.prefix_ids.dims[0], 1);
        assert.equal(feeds.doc_ids.dims[0], candidates.length);
        assert.deepEqual(Array.from(feeds.owners.data), candidates.map(() => 0n));
        const data = candidates.flatMap(c => [Number(c.text), -Number(c.text), Number(c.text) / 2]);
        return { logits: { data, dims: [candidates.length, 3] } };
      } },
    });
    const column = manifest.tasks.indexOf(task);
    const expected = interpret(request, candidates.map(c => [Number(c.text), -Number(c.text), Number(c.text) / 2][column]));
    assert.equal(runs, 1);
    assert.deepEqual(actual, expected);
    assert.deepEqual(Object.keys(actual.probabilities), candidates.map(c => c.id));
  });
}

test('long candidates use bounded batches and combine logits before normalizing', async () => {
  const request = { task: 'choice', instruction: 'Choose', state: '', candidates: Array.from({ length: 5 }, (_, i) => ({ id: `c${i}`, text: `${i + 1}` })) };
  const sizes = [];
  let elapsed;
  const result = await predict(request, {
    manifest, ort: { Tensor },
    tokenizer: { encode: text => ({ ids: Array(Number(text) ? 600 : 1).fill(Number(text) || 1) }) },
    session: { async run(feeds) {
      const [rows, width] = feeds.doc_ids.dims;
      sizes.push(rows);
      const data = Array.from({ length: rows }, (_, i) => [Number(feeds.doc_ids.data[i * width]), 0, 0]).flat();
      return { logits: { data, dims: [rows, 3] } };
    } },
  }, { onInferenceTime: ms => { elapsed = ms; } });
  assert.deepEqual(sizes, [2, 2, 1]);
  assert.deepEqual(result, interpret(request, [1, 2, 3, 4, 5]));
  assert.ok(Number.isFinite(elapsed) && elapsed >= 0);
});
