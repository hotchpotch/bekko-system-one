import { test } from 'node:test';
import assert from 'node:assert/strict';
import { contextKeyError, contextObject } from '../src/context.js';

test('key collisions cannot overwrite either context value', () => {
  const rows = [['review', 'First value'], ['review', 'Second value']];
  assert.ok(contextKeyError(rows, 0));
  assert.ok(contextKeyError(rows, 1));
  assert.throws(() => contextObject(rows), /already used/);
  rows[1][0] = 'comment';
  assert.deepEqual(contextObject(rows), { review: 'First value', comment: 'Second value' });
  rows[0][0] = '  ';
  assert.throws(() => contextObject(rows), /Enter a context key/);
});

test('serialization preserves literal keys and supports empty context', () => {
  const result = contextObject([['__proto__', 'text'], ['constructor', 'other']]);
  assert.equal(Object.getPrototypeOf(result), Object.prototype);
  assert.equal(JSON.stringify(result), '{"__proto__":"text","constructor":"other"}');
  assert.deepEqual(contextObject([]), {});
});
