// Run against the deployed or built app. Private hosts must already be authenticated.
// Inference must work even when private-host runtime module and WASM requests are unavailable.
async (page) => {
  const moduleRoute = `${await page.evaluate(() => location.origin)}/**/ort-wasm-*`;
  const blocked = [];
  const rejectModule = route => {
    blocked.push(route.request().url());
    return route.abort('accessdenied');
  };
  await page.context().route(moduleRoute, rejectModule);
  try {
    await page.reload();
    await page.locator('#run-example').click();
    await page.waitForFunction(() => !document.querySelector('#run').disabled, null, { timeout: 180000 });
    const result = JSON.parse(await page.locator('#response-json').textContent());
    if (result.error || !result.probabilities) throw Error(result.error || 'No inference result');
    if (blocked.length) throw Error('Runtime still requests private-host runtime assets');
    return { separateModuleRequests: blocked.length, result };
  } finally {
    await page.context().unroute(moduleRoute, rejectModule);
  }
}
