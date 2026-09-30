# Concepts and architecture

Bekko System One scores explicit candidates for an instruction and context.
It returns structured decisions rather than generating answer text.

## Choose a task

| Task | Example | Interpretation |
| --- | --- | --- |
| Choice | Route a request to billing or technical support | Select a candidate ID and inspect the option distribution |
| Noul | Determine whether a request asks for a refund | Estimate the probability of the authored positive meaning |
| Score | Rate urgency at levels 0, 2, and 4 | Compute an expected value on the supplied numeric scale |
| Reranking | Order documents for a search query | Compare relevance within the supplied candidate set |

Noul is a yes/no decision with descriptions of both outcomes. Define the negative
meaning carefully: “not supported by the evidence” and “contradicted by the
evidence” are different conditions.

Score values are explicit, not inferred from labels or positions. Probabilities
of 0.2, 0.5, and 0.3 over values 0, 2, and 4 yield a score of 2.2. This arithmetic
example illustrates the output, not a measured prediction.

Choice and relative ranking probabilities depend on the candidate set. Adding
or removing options changes that comparison. A high probability does not by
itself establish calibration or correctness.

## Shared-prefix attention

The instruction and state form a shared query prefix. Each candidate attends
to that prefix and its own branch; it cannot attend to other candidates.

```text
Instruction + state -> shared prefix -> Candidate A -> pooling -> task head
                                    -> Candidate B -> pooling -> task head
                                    -> Candidate C -> pooling -> task head
```

A conventional pairwise cross-encoder processes the query again for each
candidate. Shared-prefix attention reuses prefix computation within a batch.
Actual speed depends on prefix length, candidate count, batching, and hardware;
this is not a cache across unrelated requests or a fixed speedup guarantee.

The optional Choice interaction head allows pooled candidate representations
to interact within one decision. Encoder branches remain independent. This
head is not supported by the browser exporter.

## Inputs, targets, and rendering

A native inference request contains `state_json` and `decisions`. Fields ending
in `_json` hold JSON-encoded strings. Candidate IDs identify options independently
of order; numeric criterion values define Score scales. See the complete
[input example](../examples/v0-input.json).

Training rows keep targets and provenance separate from model input. Pass only
the input object at inference and preserve IDs, descriptions, order, and values.
Do not insert labels or target probabilities into the text.

Rendering and truncation are part of the model interface. Match the checkpoint's
prefix layout and input policy. See [training](training.md) for data formats and
[compatibility](compatibility.md) for runtime differences.
