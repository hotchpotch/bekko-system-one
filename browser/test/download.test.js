import { test } from "node:test";
import assert from "node:assert/strict";
import { download } from "../src/download.js";

const url = "https://example.test/model.onnx";
function response(headers = {}) {
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2]));
        controller.enqueue(new Uint8Array([3]));
        controller.close();
      },
    }),
    { headers },
  );
}
test("streamed downloads preserve bytes and report measurable progress", async () => {
  const events = [];
  const bytes = await download(
    url,
    (e) => events.push(e),
    async () => response({ "content-length": "3" }),
  );
  assert.deepEqual([...bytes], [1, 2, 3]);
  assert.deepEqual(
    events.map((e) => e.loaded),
    [0, 2, 3, 3],
  );
  assert.ok(events.every((e) => e.total === 3));
  assert.equal(events.at(-1).done, true);
});
test("unknown and compressed content lengths stay indeterminate until complete", async () => {
  for (const headers of [
    {},
    { "content-length": "2", "content-encoding": "gzip" },
  ]) {
    const events = [];
    await download(
      url,
      (e) => events.push(e),
      async () => response(headers),
    );
    assert.ok(events.slice(0, -1).every((e) => e.total === null));
    assert.deepEqual(events.at(-1), { loaded: 3, total: 3, done: true });
  }
});
test("HTTP and interrupted download failures can be retried", async () => {
  await assert.rejects(
    download(url, undefined, async () => new Response("", { status: 503 })),
    /503/,
  );
  await assert.rejects(
    download(
      url,
      undefined,
      async () =>
        new Response(
          new ReadableStream({
            start(c) {
              c.error(Error("Connection lost"));
            },
          }),
        ),
    ),
    /Connection lost/,
  );
  assert.deepEqual(
    [...(await download(url, undefined, async () => response()))],
    [1, 2, 3],
  );
});
