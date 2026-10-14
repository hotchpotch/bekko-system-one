# Candidate encoder architectures

`build_model()` and the training YAML `model` section accept `architecture`:

| Architecture | Pooling | Attention scope |
| --- | --- | --- |
| `shared_prefix` (default) | candidate mean | Query cannot attend to candidates; each candidate sees its query and itself. |
| `cross_encoder` | `cls`, `mean` (default) | Each query/candidate pair is encoded bidirectionally. Mean includes the full pair and its special tokens. |
| `listwise` | `mean` (default), `token_before`, `token_after` | One sequence per complete decision; candidates can attend to each other. |

All variants retain the backbone's global/local attention layer schedule. The
joint encoders use packed sequence attention with either FlashAttention 2 on CUDA
or SDPA. Separate decisions never share listwise attention, even when their query
text is identical. Candidate order is preserved. A listwise prediction depends on
the complete candidate set, so changing that set may change existing scores.

For example, replace the model options in an existing training configuration:

```yaml
model:
  model_name_or_path: cross-encoder/ettin-reranker-17m-v1
  architecture: listwise
  pooling: token_after
  lora: null
  frozen_linear_bf16: false
  query_length: 4096
  document_length: 2048
  query_truncation: balanced
  attention_backend: flash_attention_2
```

`token_before` places a marker immediately before each candidate's text;
`token_after` places it immediately after that text, before SEP. By default the
marker is the tokenizer's MASK. Set `candidate_token: "[CANDIDATE]"` to register a
new special token, initialized from the SEP embedding and trained with the model.
An existing ordinary vocabulary token cannot be repurposed as a marker. Marker
pooling currently requires full training (`lora: null`). Listwise mean pooling
excludes the delimiter SEP, with SEP used as a fallback for empty candidate text.
Configured task markers remain part of candidate content.

## Length limits

`max_length` is the complete joint sequence limit, defaulting to the backbone's
`max_position_embeddings`. It cannot exceed that capacity. The query and candidate
limits still apply first, using the same renderer and tokenizer as training.
Cross-encoder limits apply independently to every pair. Listwise limits apply to
the complete decision, including all candidate delimiters and pooling markers.

The allocator reserves a delimiter (and pooling marker, when enabled) for every
candidate and at least one content token for each nonempty candidate. It retains
the query up to the remaining capacity, then divides candidate capacity with a
max-min fair allocation: short candidates donate unused capacity to longer ones.
Candidates are right-truncated; none is silently dropped. If even the minimum
candidate representation and query CLS/SEP cannot fit, a clear error is raised.
When that reservation forces additional query truncation, both ends of the
already-rendered query are retained. Balanced initial query truncation requires
`QueryParts`, as for the shared-prefix architecture.

## Inference and checkpoints

Use `predict`, `predict_typed`, or `rank` from `bekko_system_one.model`.
`SentenceTransformer.save_pretrained()` checkpoints retain architecture, pooling,
length limits, marker vocabulary and weights; load with the installed package
and `trust_remote_code=True`. Training and inference use identical collation and
truncation. Joint models use their normal packed forward for all inference modes;
shared-prefix caching/CUDA Graph optimizations do not apply. `rank` retains the
complete candidate group for joint models regardless of `chunk_size`.

The standalone v0 exporter and browser runtime support shared-prefix models only.
Exporting a joint model raises an error instead of silently changing its attention
semantics. Training token-budget counters are conservative shared-prefix work
estimates, not measurements of joint sequence tokens or attention FLOPs. Adaptive
OOM replay still preserves complete decisions.
