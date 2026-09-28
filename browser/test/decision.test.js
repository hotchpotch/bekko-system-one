import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseCriteria, renderDecision } from '../src/decision.js';
const json = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));

test('UI decisions match the native Python training renderer', async () => {
  for (const { decision, request } of await json('./fixtures/decision-rendering.json')) {
    assert.deepEqual(renderDecision(decision), request);
    assert.ok(!('system' in request));
  }
});
test('line input supports labels, descriptions, explicit score values and defaults', () => {
  assert.deepEqual(parseCriteria('choice', 'Billing\nshipping | Package delivery'), [{ id: 'Billing', description: 'Billing' }, { id: 'shipping', description: 'Package delivery' }]);
  assert.deepEqual(parseCriteria('score', '-1 | Negative\n0 | Neutral\n1 | Positive').map(c => c.value), [-1, 0, 1]);
  assert.equal(parseCriteria('noul', '')[0].id, 'true');
  for (const lines of ['0 | Zero\n0 | Duplicate', '| Missing number\n1 | One', 'NaN | Invalid\n1 | One', '0 |\n1 | One']) assert.throws(() => parseCriteria('score', lines));
  assert.throws(() => parseCriteria('choice', 'One'));
  assert.throws(() => parseCriteria('choice', 'A\nA'));
});
