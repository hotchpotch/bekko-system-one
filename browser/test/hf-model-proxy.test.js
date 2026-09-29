import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { modelProxy } from '../scripts/hf-model-proxy.js';

async function serve(t, token, fetcher) {
  const middleware = modelProxy(token, fetcher);
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end(); }));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('proxy streams only approved assets and never forwards HF credentials to CDN redirects', async t => {
  const calls = [];
  const base = await serve(t, 'test-only-token', async (url, options) => {
    calls.push({ url: String(url), options });
    return calls.length === 1
      ? new Response(null, { status: 302, headers: { location: 'https://cdn.example/model' } })
      : new Response('model data', { headers: { 'content-length': '10' } });
  });
  const response = await fetch(`${base}/__hf_models/400m/model.onnx`);
  assert.equal(await response.text(), 'model data');
  assert.equal(response.headers.get('cache-control'), 'private, no-store');
  assert.match(calls[0].url, /bekko-system-one-v0-400m\/resolve\/[a-f0-9]+\/onnx_browser\/model.onnx$/);
  assert.equal(calls[0].options.headers.Authorization, 'Bearer test-only-token');
  assert.deepEqual(calls[1].options.headers, {});
  for (const path of ['other/model.onnx', '17m/config.json', '17m/model.onnx?url=https://example.com']) {
    assert.equal((await fetch(`${base}/__hf_models/${path}`)).status, 404);
  }
  assert.equal((await fetch(`${base}/__hf_models/17m/model.onnx`, { headers: { Origin: 'https://elsewhere.example' } })).status, 403);
  assert.equal((await fetch(`${base}/__hf_models/17m/model.onnx`, { method: 'POST' })).status, 405);
  assert.equal(calls.length, 2);
});

test('missing credentials and upstream errors do not expose credentials or upstream bodies', async t => {
  const missing = await serve(t, '', () => { throw Error('Must not fetch'); });
  assert.equal((await fetch(`${missing}/__hf_models/17m/manifest.json`)).status, 503);
  const denied = await serve(t, 'test-only-token', async () => new Response('sensitive upstream body', { status: 401 }));
  const response = await fetch(`${denied}/__hf_models/17m/manifest.json`);
  assert.equal(response.status, 401);
  assert.doesNotMatch(await response.text(), /test-only-token|sensitive upstream/);
});
