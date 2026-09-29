import { test } from 'node:test';
import assert from 'node:assert/strict';
import { predict } from '../src/core.js';

test('inference time sums awaited model runs and excludes tokenization and tensor preparation', async (t) => {
  let clock = 0;
  t.mock.method(performance, 'now', () => clock);
  let measured;
  let runs = 0;
  const result = await predict({
    task: 'choice', instruction: 'Choose', state: '{}',
    candidates: [{ id: 'a', text: 'A' }, { id: 'b', text: 'B' }],
  }, {
    tokenizer: { encode() { clock += 100; return { ids: [1] }; } },
    manifest: { tasks: ['choice'], query_length: 64, document_length: 16, cls_token_id: 1, sep_token_id: 2, pad_token_id: 0 },
    ort: { Tensor: class { constructor() { clock += 50; } } },
    session: { async run() { await Promise.resolve(); clock += 7; runs++; return { logits: { data: [runs] } }; } },
  }, { onInferenceTime: elapsed => { measured = elapsed; } });
  assert.equal(runs, 2);
  assert.equal(measured, 14);
  assert.ok(clock > measured);
  assert.equal(result.selected_id, 'b');
});
