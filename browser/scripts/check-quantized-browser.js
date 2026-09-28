// Open the built preview and run its default example once before this check.
// playwright-cli run-code --filename=scripts/check-quantized-browser.js
async (page) => {
  const workerUrl = page.workers()[0]?.url();
  if (!workerUrl) throw Error('Run the default example first to start the inference worker');
  return await page.evaluate(async (workerUrl) => {
    const fixtures = await (await fetch('./model/quantized/browser-smoke.json')).json();
    const results = {};
    for (const [name, cases] of Object.entries(fixtures)) {
      const worker = new Worker(workerUrl, { type: 'module' });
      const errors = [], timings = [];
      try {
        for (const fixture of cases) {
          const reply = await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(Error('Timed out')), 60000);
            worker.onerror = error => { clearTimeout(timeout); reject(Error(error.message)); };
            worker.onmessage = ({ data }) => {
              if (data.status || data.progress || data.loadedBytes !== undefined) return;
              clearTimeout(timeout);
              if (data.error) reject(Error(data.error)); else resolve(data);
            };
            worker.postMessage({ request: fixture.request, base: new URL(`./model/quantized/${name}/`, document.baseURI).href });
          });
          for (const [id, expected] of Object.entries(fixture.expected.probabilities)) errors.push(Math.abs(expected - reply.result.probabilities[id]));
          timings.push(reply.milliseconds);
        }
      } finally { worker.terminate(); }
      results[name] = { cases: cases.length, maxProbabilityErrorVsNode: Math.max(...errors), milliseconds: timings };
      results[name].withinTolerance = results[name].maxProbabilityErrorVsNode <= 1e-4;
      if (!Number.isFinite(results[name].maxProbabilityErrorVsNode)) throw Error(`${name}: non-finite output`);
    }
    return results;
  }, workerUrl);
}
