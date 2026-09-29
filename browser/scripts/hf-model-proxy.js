import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { MODELS } from '../src/models.js';

const files = new Set(['manifest.json', 'tokenizer.json', 'tokenizer_config.json', 'model.onnx']);

export function modelProxy(token, fetcher = fetch) {
  return async (req, res, next) => {
    if (!req.url.startsWith('/__hf_models/')) return next();
    res.setHeader('Cache-Control', 'private, no-store');
    const match = /^\/__hf_models\/([^/]+)\/([^/?]+)$/.exec(req.url);
    const model = MODELS.find(model => model.value === match?.[1]);
    if (!model || !files.has(match?.[2])) { res.statusCode = 404; res.end('Unknown model asset'); return; }
    if (!['GET', 'HEAD'].includes(req.method)) { res.statusCode = 405; res.end('Method not allowed'); return; }
    if (req.headers['sec-fetch-site'] === 'cross-site' ||
        (req.headers.origin && req.headers.origin !== `http://${req.headers.host}` && req.headers.origin !== `https://${req.headers.host}`)) {
      res.statusCode = 403; res.end('Same-origin requests only'); return;
    }
    if (!token) { res.statusCode = 503; res.end('Set HF_TOKEN in the local server environment or .env.local and restart.'); return; }
    const controller = new AbortController();
    const abort = () => { if (!res.writableFinished) controller.abort(); };
    res.on('close', abort);
    try {
      let url = new URL(`https://huggingface.co/${model.repository}/resolve/${model.revision}/onnx_browser/${match[2]}`);
      let response;
      for (let redirects = 0; redirects <= 5; redirects++) {
        response = await fetcher(url, {
          method: req.method, redirect: 'manual', signal: controller.signal,
          headers: url.origin === 'https://huggingface.co' ? { Authorization: `Bearer ${token}` } : {},
        });
        if (![301, 302, 303, 307, 308].includes(response.status)) break;
        const location = response.headers.get('location');
        await response.body?.cancel();
        if (!location || redirects === 5) throw Error('Invalid redirect');
        url = new URL(location, url);
        if (url.protocol !== 'https:' || url.username || url.password) throw Error('Invalid redirect');
      }
      if (!response.ok) { res.statusCode = response.status; res.end(`Model download failed (HTTP ${response.status}). Check HF_TOKEN access.`); return; }
      res.statusCode = response.status;
      res.setHeader('Content-Type', response.headers.get('content-type') || 'application/octet-stream');
      // fetch decompresses responses; only forward lengths for unencoded bodies.
      if (!response.headers.get('content-encoding') && response.headers.get('content-length')) {
        res.setHeader('Content-Length', response.headers.get('content-length'));
      }
      if (req.method === 'HEAD' || !response.body) res.end();
      else await pipeline(Readable.fromWeb(response.body), res);
    } catch {
      if (!res.headersSent) { res.statusCode = 502; res.end('Model download failed. Check network access and retry.'); }
      else res.destroy();
    } finally {
      res.off('close', abort);
    }
  };
}
