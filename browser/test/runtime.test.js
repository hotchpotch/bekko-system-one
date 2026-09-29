import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectWebGPU, executionProviders } from '../src/runtime.js';
test('GPU availability requires a secure context and an actual adapter', async () => {
  assert.equal((await detectWebGPU({isSecureContext:false})).available,false);
  assert.equal((await detectWebGPU({isSecureContext:true,navigator:{}})).available,false);
  for (const result of [null, {}]) {
    const gpu={requestAdapter:async()=>result};
    assert.equal((await detectWebGPU({isSecureContext:true,navigator:{gpu}})).available,!!result);
  }
  assert.equal((await detectWebGPU({isSecureContext:true,navigator:{gpu:{requestAdapter:async()=>{throw Error('blocked');}}}})).available,false);
});
test('explicit device selection preserves CPU assistance only for WebGPU', () => {
  assert.deepEqual(executionProviders('cpu'),['wasm']);
  assert.deepEqual(executionProviders('webgpu'),['webgpu','wasm']);
  assert.throws(()=>executionProviders('unknown'));
});
