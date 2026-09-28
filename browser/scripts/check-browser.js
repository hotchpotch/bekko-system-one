// Open the built preview, then: playwright-cli run-code --filename=scripts/check-browser.js
async (page) => {
  await page.reload();
  const modelRequests = [];
  const track = request => { if (request.url().endsWith('/model.onnx')) modelRequests.push(request.url()); };
  page.on('request', track);
  const results = [];
  try {
    if (await page.locator('#system').count()) throw Error('System prompt must not be in the UI');
    for (const task of ['noul', 'choice', 'score']) {
      await page.locator('#task').selectOption(task);
      if (await page.locator('#request-json').isVisible()) throw Error('Request JSON should start collapsed');
      await page.locator('#run').click();
      await page.waitForFunction(() => !document.querySelector('#run').disabled, null, { timeout: 60000 });
      const response = JSON.parse(await page.locator('#response-json').textContent());
      if (response.error) throw Error(response.error);
      if (!(await page.locator('#model-status').textContent()).includes('Model loaded · 29.0 MB')) throw Error('Missing measured model-load status');
      if (await page.locator('#result .probability').count() < 2) throw Error('Missing probability bars');
      const request = JSON.parse(await page.locator('#request-json').textContent());
      if ('system' in request || request.task !== task) throw Error('Unexpected request structure');
      results.push({ task, result: response, headline: await page.locator('.result-value').textContent() });
    }
    if (modelRequests.length !== 1 || !modelRequests[0].includes('embedding-int8')) throw Error('Model should load once, using the selected INT8 model');
    await page.locator('#options').fill('0 | First\n0 | Duplicate');
    await page.locator('#run').click();
    if (!(await page.locator('#result').textContent()).includes('different')) throw Error('Duplicate score values must be rejected');
    await page.locator('#task').selectOption('noul');
    await page.locator('#example').selectOption('custom');
    await page.locator('#instruction').fill('Is this a request for a refund?');
    await page.locator('#state-0').fill('Please refund my purchase.');
    await page.locator('#run').click();
    await page.waitForFunction(() => !document.querySelector('#run').disabled, null, { timeout: 60000 });
    const custom = JSON.parse(await page.locator('#response-json').textContent());
    if (typeof custom.probability_yes !== 'number') throw Error('Default Yes/No criteria must work');
    await page.getByText('Request JSON', { exact: true }).click();
    if (!await page.locator('#request-json').isVisible()) throw Error('Cannot expand request');
    await page.getByText('Response JSON', { exact: true }).click();
    if (!await page.locator('#response-json').isVisible()) throw Error('Cannot expand response');
    await page.setViewportSize({ width: 390, height: 844 });
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('Mobile page overflows');
    await page.setViewportSize({ width: 1280, height: 1000 });
    return { results, modelRequests, custom };
  } finally { page.off('request', track); }
}
