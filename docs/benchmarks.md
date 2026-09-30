# S1MB benchmark snapshot — September 30, 2026

For current rankings, filters, and per-benchmark results, visit the
[S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard).
This page preserves the **2026-09-30** comparison of 23 models; it is not a live
leaderboard. The collection date does not mean every model was evaluated that day.

## Scope and provenance

S1MB (System One Mosaic Benchmark) combines evaluation tasks derived from public
NLP datasets, Open-Jev/Laya tasks, and synthetic tasks. The standard view includes
106 subsets, 137 benchmarks, 14,009 cases, and 26,269 decisions. Benchmarks comprise
59 Noul, 57 Choice, and 21 Score tasks. A subset can contain multiple benchmarks,
and a case can contain multiple decisions; these counts are different units.

These tables reproduce the supplied release snapshot at its original two-decimal
precision. No evaluations were rerun to prepare this page. The snapshot did not
include an exact result-dataset commit, all model revisions, or all run settings,
so it is a historical comparison rather than a fully reproducible run manifest.
Jev is identified as version 1.13 (`jev-1.13.0`).

Public project resources:

- [Evaluation code](https://github.com/hotchpotch/S1MB)
- [Evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
- [Result repository](https://huggingface.co/datasets/hotchpotch/s1mb-result)
- [Scoring definitions](https://github.com/hotchpotch/S1MB/blob/main/evaluator/SCORING.md)

## Overall results: 137 benchmarks

The README focuses on models with at most 500M total parameters, plus Jev.
This full snapshot includes all 23 models, each covering 137 benchmarks, sorted
by Task Avg. Parameter counts reproduce the supplied total counts; they are not
independently recounted. Jev's count was not disclosed in the snapshot.

| Model | Total parameters | Task Avg | Noul | Choice | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Jev 1.13](https://docs.typesafe.ai/models) | Not disclosed | 59.59 | 64.63 | 67.22 | 46.92 |
| Open-Jev-27B-v1.1 (ZefanCai) | 25.64B | 56.88 | 51.48 | 66.91 | 52.26 |
| Open-Jev-9B (ZefanCai) | 7.94B | 52.10 | 55.76 | 55.51 | 45.04 |
| [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 395M | 50.60 | 51.24 | 61.32 | 39.25 |
| Tev1-4B-experimental | 4.54B | 47.07 | 50.73 | 53.36 | 37.11 |
| kev-9b | 7.94B | 42.40 | 50.21 | 56.36 | 20.64 |
| kev-4b | 4.21B | 42.33 | 48.48 | 52.03 | 26.47 |
| JevK5 | 4.21B | 41.14 | 48.93 | 52.84 | 21.65 |
| [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 68M | 40.46 | 42.91 | 51.62 | 26.85 |
| OpenJev-4B (Alex) | 4.54B | 36.46 | 42.77 | 47.49 | 19.11 |
| Open-Jev-2B (ZefanCai) | 1.88B | 34.64 | 38.83 | 39.82 | 25.27 |
| [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 17M | 27.57 | 31.43 | 35.57 | 15.70 |
| OpenJev-2B (Alex) | 2.21B | 25.21 | 29.72 | 37.80 | 8.11 |
| decider-0.8b | 752M | 22.57 | 28.97 | 32.36 | 6.39 |
| Tev1-0.8B-experimental | 853M | 20.43 | 25.97 | 27.15 | 8.16 |
| kev-0.8b | 753M | 18.68 | 23.61 | 27.36 | 5.07 |
| [von](https://huggingface.co/wfzyx/von) | 395M | 16.21 | 20.15 | 23.99 | 4.48 |
| OpenJev-0.8B (Alex) | 853M | 16.02 | 18.90 | 23.79 | 5.36 |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | 15.00 | 20.06 | 18.93 | 5.99 |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | 13.36 | 20.19 | 16.31 | 3.58 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | 9.28 | 14.01 | 13.03 | 0.79 |
| minojev | 1.72B | 7.93 | 8.57 | 14.56 | 0.67 |
| JevForge-0.8B | 854M | 1.92 | 0.00 | 5.75 | 0.00 |

## How to read the scores

Task columns and Task Avg are baseline-adjusted scores on a 0–100 scale:

- **Noul:** skill is `2 × balanced accuracy − 1`.
- **Choice:** skill is `(target mass at the selected option − b) / (1 − b)`;
  `b` is the best expected result from uniform random, fixed candidate ID, or
  fixed position baselines.
- **Score:** skill is `1 − MAE / baseline MAE`, where the constant baseline
  predicts the median target expected value. Values are normalized to each
  decision's numeric scale before computing the expected-value error.

Per-benchmark skill is clipped to [0, 1], averaged within each task, then scaled
by 100. Task Avg gives the three tasks equal weight. A score of 50 means half
of the baseline-to-reference-ceiling gap, not 50% accuracy. Clipping hides how far
below baseline a model performs; inspect raw metrics as well. With soft targets,
the reference ceiling need not be attainable.

These tables do not establish statistical significance for small differences.

## Generalization results: six adaptation benchmarks

Each task has a Diverse and Contextual benchmark, with 100 cases apiece.
Diverse varies instructions, states, and criteria across tasks. Contextual uses
20 groups of five cases per task: Choice and Score retain the question and
criteria while changing state; Noul varies state and the authored yes/no criteria.
The 300 Contextual cases therefore represent 60 groups, not 300 independent
scenarios.

For example, a contextual Choice case asks what to serve a guest who is fasting
until sunset. With a noon context and an explicit refusal of food, the intended
choice is to defer the meal. This tests whether the supplied conditions govern
the choice, rather than whether a food option sounds plausible in isolation.

All rows below cover six benchmarks and are sorted by Task Avg for this view.
The six are included once in the overall view; this is a filtered evaluation,
not a separate training run.

| Model | Total parameters | Task Avg | General Noul | General Choice | General Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Jev 1.13](https://docs.typesafe.ai/models) | Not disclosed | 96.27 | 99.00 | 98.68 | 91.14 |
| Open-Jev-9B (ZefanCai) | 7.94B | 85.63 | 93.00 | 96.70 | 67.21 |
| JevK5 | 4.21B | 85.01 | 92.00 | 93.39 | 69.64 |
| kev-9b | 7.94B | 84.58 | 87.00 | 96.05 | 70.69 |
| Tev1-4B-experimental | 4.54B | 84.07 | 88.00 | 94.05 | 70.15 |
| Open-Jev-27B-v1.1 (ZefanCai) | 25.64B | 82.97 | 93.00 | 100.00 | 55.91 |
| kev-4b | 4.21B | 77.50 | 83.00 | 93.44 | 56.06 |
| OpenJev-4B (Alex) | 4.54B | 76.75 | 82.00 | 94.12 | 54.13 |
| Open-Jev-2B (ZefanCai) | 1.88B | 65.57 | 71.00 | 85.56 | 40.15 |
| OpenJev-2B (Alex) | 2.21B | 64.08 | 70.00 | 88.85 | 33.39 |
| decider-0.8b | 752M | 55.92 | 57.00 | 82.90 | 27.86 |
| Tev1-0.8B-experimental | 853M | 54.61 | 59.00 | 76.40 | 28.43 |
| [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 395M | 54.48 | 52.00 | 80.30 | 31.14 |
| OpenJev-0.8B (Alex) | 853M | 52.33 | 57.00 | 77.09 | 22.89 |
| kev-0.8b | 753M | 51.22 | 53.00 | 75.15 | 25.51 |
| [von](https://huggingface.co/wfzyx/von) | 395M | 43.19 | 41.00 | 65.89 | 22.67 |
| [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 68M | 32.48 | 31.00 | 57.52 | 8.90 |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | 31.82 | 22.00 | 58.80 | 14.67 |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | 26.50 | 9.00 | 54.83 | 15.67 |
| minojev | 1.72B | 25.08 | 11.00 | 63.37 | 0.87 |
| [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 17M | 18.96 | 15.00 | 39.11 | 2.76 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | 14.47 | 6.00 | 36.67 | 0.75 |
| JevForge-0.8B | 854M | 3.74 | 0.00 | 11.22 | 0.00 |

The 400M model reaches an overall Task Avg of 50.60, but trails Jev substantially
on adaptation (54.48 versus 96.27). Its General Choice score is substantially
higher than its General Noul and General Score scores, but all three trail
Jev 1.13. The smaller Bekko models have larger gaps. This supports a
limited-generalization description of v0 even when individual tasks or task
families perform well.

GPT-6-Astra generated the questions, intended answers, and self-review. There
was no independent human validation. Results measure agreement with that test
design, including its numeric rubrics, rather than a universal definition of
correct judgment. They do not establish generalization to all unseen tasks or
absence of training overlap for every model.

## Training exposure

The [training manifest](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0/blob/c998cebc8ebd55253e43afb729c14ab470b0fc92/training-manifest.json)
and [evaluation manifest](https://huggingface.co/datasets/hotchpotch/s1mb-dataset/blob/61c988adcee34f259a8e00eac6647fc41e023a5f/training-manifest.json)
have **77 exactly matching subset names**, out of 153 training and 106 evaluation
subsets in those revisions. This indicates related task families, not duplicated
test examples. Sampling and caps also determine which training rows are used.
The six synthetic adaptation subsets are absent by name from that training
manifest; different names do not prove absence of semantic overlap.

This exposure must be considered when interpreting Bekko's overall results, but
it is not a controlled experiment establishing the cause of any score. Jev's
training-data composition is not documented in this snapshot. Neither its
training overlap nor its absence can be inferred from the scores.

For evaluating your own checkpoint, follow the
[S1MB evaluation guide](https://github.com/hotchpotch/S1MB/blob/main/docs/evaluation.md).
For reporting your results, see [Evaluation and limitations](evaluation.md).
