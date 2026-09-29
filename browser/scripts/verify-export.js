// node scripts/verify-export.js /path/to/checkpoint/onnx
import { readFile } from "node:fs/promises";
import { loadModel } from "../src/node.js";
import { tokenize, feedsFor } from "../src/core.js";
const directory = process.argv[2];
if (!directory) throw Error("Supply an ONNX export directory");
const fixtures = JSON.parse(await readFile(`${directory}/parity.json`));
const fp32 = await loadModel(directory, { modelFile: "model.onnx" });
const int8 = await loadModel(directory);
let logitError = 0,
  probabilityDifference = 0;
try {
  for (const fixture of fixtures) {
    const tokens = tokenize(fixture.request, fp32.tokenizer, fp32.manifest);
    const { logits } = await fp32.session.run(
      feedsFor(tokens, fp32.ort, fp32.manifest),
    );
    fixture.logits.flat().forEach((x, i) => {
      logitError = Math.max(logitError, Math.abs(x - logits.data[i]));
    });
    const a = await fp32.predict(fixture.request),
      b = await int8.predict(fixture.request);
    for (const id of Object.keys(a.probabilities))
      probabilityDifference = Math.max(
        probabilityDifference,
        Math.abs(a.probabilities[id] - b.probabilities[id]),
      );
  }
  if (logitError > 1e-4 || probabilityDifference > 0.03)
    throw Error(`Parity failed: ${logitError}, ${probabilityDifference}`);
  console.log(
    JSON.stringify(
      {
        fixtures: fixtures.length,
        logitError,
        probabilityDifference,
        manifest: int8.manifest,
      },
      null,
      2,
    ),
  );
} finally {
  await fp32.release();
  await int8.release();
}
