# Decision Index evaluation

The [official reproduction kit](https://github.com/apolinario/decision-index)
implements the pinned Decision Index 0.2.1 scoring rules. This is a different
evaluation from S1MB: do not compare their headline numbers directly.

The adapter in `examples/decision_index_engine.py` loads the public standalone
17M release at commit `b886a1f9b91f4e8368d7830080d52d955e2c8dfa`, including its
published runtime. It preserves question and candidate order, IDs and full
probability distributions, and uses the native Choice and Noul heads. Gold
answers and provenance are never passed to the model. One fixed rendering is
used across benchmarks.

For Noul questions without explicit criteria, the fixed descriptions are
"The statement in the instructions is true." and "The statement in the
instructions is false." Explicit criteria are preserved. This conversion is
recorded in engine provenance and is independent of the benchmark.

The kit forbids truncation. Before inference, the adapter compares the runtime's
prepared token IDs against uncapped tokenization, including task markers and
balanced instruction/state rendering. If any branch would lose tokens, it
refuses the entire request as `unsupported`. The scorer counts unsupported
requests as unanswered, and retains linked-case completeness rules. Native v0
limits are recorded in `environment.json`; they are not silently enlarged.

For a separate exploratory run, `--option allow_truncation=true` permits the
release runtime's standard query/candidate truncation instead of refusing long
requests. All original candidate IDs are retained. `raw_output` records which
decisions were truncated, and provenance marks the policy. Use a separate
output directory: these results do not satisfy the leaderboard's no-truncation
rule and must not be submitted as a compliant run. This option does not extend
the backbone's positional capacity; it measures predictions on shortened input.

## Local run

Run from this repository root after `uv sync --locked`. The example defaults to
CPU with SDPA, without FlashAttention; select an available GPU explicitly with
`--option device=cuda:1`. The public model requires no authentication. Rebuilding the
suite requires an authenticated Hub account with access to
[cais/hle](https://huggingface.co/datasets/cais/hle), approximately 7 GB of
downloads and 17 GB of working space. Other source-specific terms are listed
in the kit's `docs/suite.md`. Do not redistribute the rebuilt suite.

```sh
git clone https://github.com/apolinario/decision-index.git output/decision-index-kit
git -C output/decision-index-kit checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf

PYTHONPATH=output/decision-index-kit uv run --locked \
  --with 'mido>=1.3' --with 'python-chess>=1.10' --with 'tiktoken>=0.7' \
  python -m decision_index suite rebuild --work output/decision-index-work

PYTHONPATH=output/decision-index-kit:examples uv run --locked \
  python -m decision_index suite import \
  --rows output/decision-index-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
  --added-rows output/decision-index-work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
  --dir output/decision-index-suite

PYTHONPATH=output/decision-index-kit:examples uv run --locked \
  python -m decision_index pipeline --suite-dir output/decision-index-suite \
  --engine decision_index_engine:BekkoEngine \
  --option device=cuda:1 \
  --out output/decision-index-17m
```

If the HoVer university server is slow, build the base suite with
`suite rebuild --edition 0.1 --work output/decision-index-work`, then replace
the default added-suite acquisition with:

```sh
PYTHONPATH=output/decision-index-kit uv run --locked --with 'tiktoken>=0.7' \
  python examples/decision_index_added_mirror.py --work output/decision-index-work
```

This uses the 4,475 needed Wikipedia documents from
`Seungjuhan/hover_original` at commit `e04edde79f89d5664589f02240057409065a9a7a`
instead of downloading the entire 2.2 GB database. The official row builders
remain unchanged. Both base and added corpus hashes must match the official
suite before import; a differing mirror is rejected. This route was verified
to reproduce the official added-row hash
`7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89`.

The temporary `--with` dependencies leave `pyproject.toml` and `uv.lock`
unchanged. The import command verifies the official corpus hashes. Runs resume
from their existing results; errors are retried. Loading and warmup precede
request timing, while request timing includes rendering, length checks,
tokenization and synchronized inference. Local latency depends on hardware and
concurrent workloads and does not establish the leaderboard's latency result.

## Leaderboard submission

The official kit's README documents the submission workflow:

1. Complete the full suite. `scores.json` must contain `"complete": true`.
2. Upload the run artifacts to a Hugging Face dataset repository: results,
   `scores.json`, `benchmark-summary.json`, `index.json`, `environment.json`
   and `status.json`. Include the reproducible adapter and pinned model/kit
   revisions; declare all capacity limits.
3. Open a PR to `apolinario/decision-index`, adding an entry to
   `submissions/README.md` with the model name, results link, engine/commit and
   hardware. The result path must include `runs/<name>/scores.json`.

Maintainers rescore the untouched results and compare answers on a private
held-out sample. They measure single-process latency on one RTX PRO 6000;
models with median latency above 1,000 ms per request are not added. A local
RTX 5090 measurement is only supporting evidence. Uploads and PR submission
are separate actions from running the local evaluation.

## Release comparison

To evaluate another release, pass both `--model` and `--option revision=...`
and give each model its own output directory. Pinned public releases used for
the comparison:

| Size | Model | Revision |
| --- | --- | --- |
| 17M | `hotchpotch/bekko-system-one-v0-17m` | `b886a1f9b91f4e8368d7830080d52d955e2c8dfa` |
| 68M | `hotchpotch/bekko-system-one-v0-68m` | `ab7685f23e5edbc1acb12ced4f2c4e12591efa69` |
| 400M | `hotchpotch/bekko-system-one-v0-400m` | `1960df5602bd93cc8d336fdebd9fb68a30926e13` |

Use the same verified suite and record the device/backend per run. Finish all
three strict runs before the three optional native-truncation runs. For the
latter, append `--option allow_truncation=true` and change the output directory.
The official scorer can compute these exploratory scores, but a completion
flag alone does not imply compliance with the no-truncation rule.
