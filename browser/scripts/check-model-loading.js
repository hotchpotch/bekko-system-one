// Open the preview, then run with playwright-cli run-code --filename=scripts/check-model-loading.js
async (page) => {
  const context = page.context();
  let attempts = 0;
  const route = async (request) => {
    attempts++;
    if (attempts === 1)
      await request.fulfill({ status: 503, body: "Temporarily unavailable" });
    else {
      await page.waitForTimeout(700);
      await request.continue();
    }
  };
  await context.route("**/model.onnx", route);
  try {
    await page.reload();
    await page.locator("#load-model").click();
    await page.waitForFunction(() =>
      document.querySelector('[role="alert"]')?.textContent.includes("503"),
    );
    if (await page.locator("#load-model").isDisabled())
      throw Error("Load failure must allow retry");
    await page.locator("#load-model").click();
    await page.locator(".load-progress").waitFor();
    if (!(await page.locator("#run").isDisabled()))
      throw Error("Run must be disabled during model loading");
    await page.waitForFunction(
      () =>
        document
          .querySelector("#model-status")
          ?.textContent.includes("Model loaded"),
      null,
      { timeout: 60000 },
    );
    await page.waitForFunction(() => !document.querySelector("#run").disabled);
    if (attempts !== 2)
      throw Error("Expected one failed download and one retry");
    await page.locator("#run").click();
    await page.waitForFunction(() => !document.querySelector("#run").disabled);
    if (attempts !== 2) throw Error("Preloaded model must be reused");
    const result = JSON.parse(
      await page.locator("#response-json").textContent(),
    );
    if (typeof result.probability_yes !== "number")
      throw Error("Inference after retry failed");
    const original = await page.locator("#instruction").inputValue();
    await page.locator("#instruction").fill("Edited question");
    if (await page.locator("#result .result-value").count())
      throw Error("Editing must clear stale results");
    await page.getByRole("button", { name: "Reset example" }).click();
    if ((await page.locator("#instruction").inputValue()) !== original)
      throw Error("Reset must restore example");
    return { attempts, preloadAndRetry: true, reset: true };
  } finally {
    await context.unroute("**/model.onnx", route);
  }
};
