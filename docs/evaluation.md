# Evaluation and limitations

Smoke runs establish that the pipeline executes. They do not establish accuracy,
calibration, or generalization. The [September 30, 2026 S1MB snapshot](benchmarks.md)
reports release-model comparisons separately from smoke checks. For current
results, see the [S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard)
and the model card for the exact revision used.

## Interpret predictions

Choice compares the supplied options. Noul measures the authored positive
meaning. Score reports an expectation on explicit numeric levels. These outputs
are not guarantees of correctness; agreement with target distributions alone
does not establish calibration against real-world outcomes.

Language, domain, candidate wording, and truncation affect results. The available
documentation does not establish a supported-language guarantee or a quality
threshold for every domain. Evaluate representative held-out data for your use.

## Report reproducible results

Record model and dataset commits, configurations, explicit split roles, sample
IDs or selection policy, and code/configuration versions. Include prefix order,
token limits, truncation, precision, runtime, and attention backend. Keep test
data separate from model selection.

Report metrics with eligible example counts. The
[inference reference](inference.md#typed-predictions-and-numerical-evaluation)
describes accuracy, Brier score, numeric Score errors, and macro aggregation.
Missing metrics are not zero. Small balanced validation samples change the
evaluation distribution and do not replace a full benchmark.

Bekko supplies training and validation data in the release recipes. S1MB is a
separate test source; identify its access conditions and revision when reporting
results. Runs that omit S1MB are not S1MB evaluations.

## Measure performance

State hardware, memory, software versions, model revision, input lengths,
candidate counts, batch/token budgets, and number of requests. Separate downloads,
initialization, and compilation from warmed inference. Report end-to-end latency
separately from model execution time, and include peak memory in comparisons.

For browser results, also record browser version, WASM or WebGPU backend, and
quantization. The UI inference timer measures model execution, not the full wait.

## Model cards

Each release model card should identify its base model, training data and revisions,
training settings, evaluation results, limitations, intended uses, license, and
a runnable example. Link dataset cards for provenance and data-specific terms.
The repository's MIT license covers code, not all model and dataset assets.
