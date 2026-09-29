"""Standalone Bekko v0 typed-decision inference and remote-code interface.

Install the exported requirements.txt (and a CUDA-compatible PyTorch build for
GPU use). The exported file includes its helpers and needs no Bekko training
package, datasets, PEFT, W&B, or external FlashAttention extension.

Load once, reuse for many requests
---------------------------------
From a downloaded export, import BekkoSentenceTransformer from inference_v0.
To load the class directly from a Hugging Face model repository::

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    repo = "YOUR_ORG/YOUR_MODEL"
    revision = "FULL_COMMIT_HASH"  # Pin the same revision for code and weights.
    Model = get_class_from_dynamic_module(
        "inference_v0.BekkoSentenceTransformer", repo,
        revision=revision, token=True,
    )
    model = Model(repo, revision=revision, token=True,
                  trust_remote_code=True, device="cuda")  # Or device="cpu".

Authenticate with Hugging Face before accessing a private model. token=True
uses your saved token. Plain SentenceTransformer(..., trust_remote_code=True)
also loads the export, but its typed entry point is model[0].predict(...).
BekkoSentenceTransformer exposes predict() directly. Raw-text encode() is not
a typed-decision API.

Input and output interface
--------------------------
Pass only a native input dict with exactly state_json and decisions; do not
pass a dataset row, targets, labels, or provenance. Fields ending in _json are
JSON-encoded strings, including when they contain a plain string. For example::

    import json
    request = {
        "state_json": json.dumps({"message": "Please refund a duplicate charge."}),
        "decisions": [{
            "id": "department", "kind": "judgment", "type": "choice",
            "instructions_json": json.dumps("Which department should respond?"),
            "system_prompt": "",
            "criteria": [
                {"id": "billing", "description_json": json.dumps("Payments and refunds"),
                 "value": None},
                {"id": "technical", "description_json": json.dumps("Technical failures"),
                 "value": None},
            ],
            "documents": [], "scoring": None,
        }],
    }
    result = model.predict(request, show_progress_bar=False)
    selected = result["department"]["selected_id"]

Decision IDs must be nonempty and unique within each request; they may repeat
across requests. Candidate IDs must be nonempty and distinct within a decision.
Judgments use kind="judgment", criteria, empty documents, and scoring=None:
* choice: returns selected_id and probabilities keyed by candidate ID.
* noul: use exactly true/false or yes/no IDs and authored descriptions of both
  meanings; returns probability_yes and probabilities.
* score: supply numeric value for each criterion (at least two distinct values);
  returns score (expected value on that scale), normalized_score in [0, 1],
  probabilities, and values. Values are never inferred from descriptions.
Relative ranking uses kind="ranking", type=None, scoring="relative", empty
criteria, and documents containing id and content_json. It returns probabilities
and order (document IDs sorted by descending probability).

Choose the batching interface
-----------------------------
Use predict(request) for one request: it returns a dict keyed by decision ID.
Use predict(requests) for a list of requests: it returns a list of result dicts
in input order. This is the usual throughput interface, including mixed tasks::

    results = model.predict(
        [request, request], batch_size=128, token_budget=64000,
        show_progress_bar=False,
    )

Prefer this list call to a Python loop of single-request calls. batch_size limits
both the requests rendered per window and the decisions tokenized per window;
it is not a fixed candidate count or GPU microbatch size. token_budget estimates
candidate_count * (max_query_tokens + max_document_tokens) for each microbatch.
An oversized decision runs alone, so this is not a strict memory limit. Candidate
groups are never split. Length bucketing reduces padding; original request and
candidate order is restored. Tokenization deduplicates text within each window,
and encoder prefixes are shared within each microbatch, not cached across calls.

Use predict_groups(groups) only if you already have rendered Group objects,
for example from input_groups(request). It returns one CPU FP32 probability
tensor per decision, in group/candidate order, without typed interpretation.
It is not required for ordinary list batching. Empty request lists return [];
a request with no decisions returns {}. Inputs are materialized, not streamed;
chunk very large datasets into lists in the caller. Progress defaults to stderr;
predict counts requests, predict_groups counts decisions.

Fast execution and memory tuning
--------------------------------
Reuse a loaded model, batch requests, and select device="cuda" when available.
CUDA encoder execution uses BF16 autocast and PyTorch SDPA; CPU uses FP32.
Heads and per-decision softmax use FP32. Small CPU/GPU differences are expected.
Tune batch_size for the tokenization/sorting window and token_budget for GPU
work per microbatch; reduce the latter when memory is tight. A single oversized
decision still runs alone and may require shorter inputs or fewer candidates.
For repeated workloads, optionally enable compilation before warmup::

    model.compile_inference()  # Lazy encoder-only torch.compile, default Inductor.
    model.predict([request, request], show_progress_bar=False)  # Warmup.
    results = model.predict([request, request], show_progress_bar=False)
    model.disable_compile()  # Return to eager encoder execution.

Compilation leaves rendering, tokenization, heads, packing, and output processing
eager. First calls include compilation cost; new shapes can recompile despite
dynamic=True. Benchmark representative warmed batches, synchronizing CUDA when
timing; no universal speedup is guaranteed and compile errors are not silently
converted to eager execution.

Context limits are separate from the microbatch token_budget. Fresh v0 exports
use adaptive-v1: reserve half the context for query and candidate (candidate gets
the odd token), then lend unused capacity subject to branch caps. The default
context is the backbone positional capacity (7,999 for v0 17M); candidates default
to min(3800, context - 3) tokens. Special tokens count. All candidates in a decision
share one truncated query. Queries follow saved balanced/right truncation and
candidates truncate on the right. Per-call context_length, query_length, and
document_length override limits without modifying the checkpoint. context_length
must not exceed positional capacity. prefix_layout overrides instruction_state
or state_instruction rendering; normally keep the exported default.

CLI: python inference_v0.py --model PATH_OR_HUB_ID --input requests.json
Add --device cuda, --compile, --batch-size 128, --token-budget 64000, or
--no-show-progress-bar as needed. Input JSON can be one request or an array;
results are JSON on stdout and progress is on stderr.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import InputModule
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from .choice import ChoiceInteraction
from .data import DecisionMetadata, Group, PreparedGroup, collate_groups, to_device
from .decisions import interpret_prediction
from .query_budget import QueryParts, allocate_query_budget, balanced_query_ids


def _text(value):
    """Return strings unchanged; serialize other decoded JSON values deterministically."""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def input_groups(case_input, *, prefix_layout="instruction_state"):
    """Render one native request into ordered Group objects for predict_groups().

    Accept exactly state_json and decisions, with JSON strings for state,
    instructions and candidate descriptions/content. See the module input schema.
    This performs no tokenization or inference. Noul descriptions are embedded
    into the query; numeric Score values remain output metadata. Labels and row
    metadata are not accepted. prefix_layout controls instruction/state order.
    """
    if not isinstance(case_input, dict) or set(case_input) != {"state_json", "decisions"}:
        raise ValueError("Pass the input object only: state_json and decisions")
    state = case_input["state_json"]
    json.loads(state)
    groups, seen = [], set()
    for decision in case_input["decisions"]:
        did = decision["id"]
        if not isinstance(did, str) or not did or did in seen:
            raise ValueError("Decision IDs must be nonempty and unique")
        seen.add(did)
        kind, task = decision["kind"], decision["type"]
        if kind not in {"judgment", "ranking"}:
            raise ValueError("Unknown decision kind")
        ranking = kind == "ranking"
        if ranking and (task is not None or decision.get("scoring") != "relative"):
            raise ValueError("Ranking requires null type and scoring=relative")
        if ranking:
            task = "score"
        if not ranking and (decision.get("documents") or decision.get("scoring") is not None):
            raise ValueError("Judgments cannot contain ranking documents or scoring")
        candidates = decision["documents"] if ranking else decision["criteria"]
        if ranking and decision.get("criteria"):
            raise ValueError("Ranking cannot contain judgment criteria")
        ids = tuple(c["id"] for c in candidates)
        descriptions = [
            _text(json.loads(c["content_json" if ranking else "description_json"]))
            for c in candidates
        ]
        values = None if ranking or task != "score" else tuple(c["value"] for c in candidates)
        if values is not None and any(
            not isinstance(v, (int, float)) or isinstance(v, bool) for v in values
        ):
            raise ValueError("Score criteria require numeric values")
        context = state
        if task == "noul":
            if set(ids) not in ({"true", "false"}, {"yes", "no"}):
                raise ValueError("Noul requires authored true/false or yes/no criteria")
            mapping = dict(zip(ids, descriptions, strict=True))
            context = json.dumps(
                {
                    "noul": {
                        "yes": mapping["true" if "true" in ids else "yes"],
                        "no": mapping["false" if "false" in ids else "no"],
                    },
                    "state": json.loads(state),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        parts = QueryParts(
            _text(json.loads(decision["instructions_json"])),
            context,
            decision.get("system_prompt") or "",
            prefix_layout,
        )
        group = Group(
            parts.render(),
            [
                f"Document: {d}" if ranking else f"Candidate: {i}: {d}"
                for i, d in zip(ids, descriptions, strict=True)
            ],
            task,
            query_parts=parts,
            metadata=DecisionMetadata(
                candidate_ids=ids, candidate_values=values, kind=kind, decision_id=did
            ),
        )
        if not ranking:
            from_types = {"choice", "noul", "score"}
            if task not in from_types or (task == "score" and len(candidates) < 2):
                raise ValueError("Expected Choice, Noul or numeric Score judgment")
        groups.append(group)
    return groups


class _SDPAPrefix(nn.Module):
    """Independent prefix self-attention and suffix-aligned document attention."""

    def __init__(self, backbone):
        """Wrap a ModernBERT backbone for independent shared-prefix SDPA execution."""
        super().__init__()
        self.backbone = backbone

    def qkv(self, layer, hidden, positions):
        """Project one layer and apply rotary positions; return query/key/value tensors."""
        attn = layer.attn
        q, k, v = (
            attn.Wqkv(layer.attn_norm(hidden))
            .view(*hidden.shape[:2], 3, -1, attn.head_dim)
            .unbind(2)
        )
        cos, sin = self.backbone.rotary_emb(hidden, positions, layer.attention_type)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q.transpose(1, 2), k.transpose(1, 2), v

    @staticmethod
    def attend(q, k, v, qmask, kmask, window):
        """Apply masked SDPA, optionally windowed, and zero padded query positions."""
        qp = qmask.long().cumsum(1) - 1 + (kmask.sum(1) - qmask.sum(1))[:, None]
        kp = kmask.long().cumsum(1) - 1
        allowed = kmask[:, None, :].expand(-1, q.shape[1], -1)
        if window is not None:
            allowed = allowed & ((qp[:, :, None] - kp[:, None, :]).abs() <= window)
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed[:, None],
            dropout_p=0.0,
        ).transpose(1, 2)
        return result * qmask[:, :, None, None]

    @staticmethod
    def update(layer, hidden, attended):
        """Apply attention output and MLP residual updates for one encoder layer."""
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(2)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
        """Encode unique prefixes once and attend each candidate to its owner prefix."""
        prefix = self.backbone.embeddings(prefix_ids)
        hidden = self.backbone.embeddings(doc_ids)
        pp = (prefix_mask.long().cumsum(1) - 1).clamp_min(0)
        pm = prefix_mask[owners]
        dp = (doc_mask.long().cumsum(1) - 1).clamp_min(0) + pm.sum(1)[:, None]
        combined = torch.cat((pm, doc_mask), dim=1)
        for layer in self.backbone.layers:
            pq, pk, pv = self.qkv(layer, prefix, pp)
            window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
            prefix = self.update(
                layer, prefix, self.attend(pq, pk, pv, prefix_mask, prefix_mask, window)
            )
            q, k, v = self.qkv(layer, hidden, dp)
            k, v = torch.cat((pk[owners], k), 1), torch.cat((pv[owners], v), 1)
            hidden = self.update(layer, hidden, self.attend(q, k, v, doc_mask, combined, window))
        return self.backbone.final_norm(hidden)


class BekkoInference(InputModule):
    """Self-contained ST module with typed predict() and optional torch.compile()."""

    tokenizer: PreTrainedTokenizerBase
    config_file_name = "inference_config.json"
    save_in_root = False
    budget_policy = "adaptive-v1"

    def __init__(
        self,
        backbone,
        tokenizer,
        *,
        tasks,
        query_length=None,
        document_length=None,
        context_length=None,
        query_truncation="balanced",
        task_tokens=None,
        choice_interaction=None,
        prefix_layout="instruction_state",
    ):
        """Build a runtime from a ModernBERT backbone, tokenizer, heads and saved limits."""
        super().__init__()
        if backbone.config.model_type != "modernbert":
            raise ValueError("Expected ModernBERT-compatible weights")
        maximum = backbone.config.max_position_embeddings
        capacity = maximum if context_length is None else context_length
        if not 5 <= capacity <= maximum:
            raise ValueError("Context length exceeds the backbone positional capacity")
        self.context_length = capacity
        query_length = capacity - 2 if query_length is None else query_length
        document_length = min(3800, capacity - 3) if document_length is None else document_length
        self._validate_limits(query_length, document_length, capacity)
        if query_truncation not in {"right", "balanced"} or prefix_layout not in {
            "instruction_state",
            "state_instruction",
        }:
            raise ValueError("Invalid truncation or prefix layout")
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or set(tasks) - {"choice", "noul", "score", "reranker"}
        ):
            raise ValueError("Expected distinct supported task heads")
        self.encoder = _SDPAPrefix(backbone)
        self.tokenizer = tokenizer
        self.query_length, self.document_length = query_length, document_length
        self.query_truncation, self.tasks = query_truncation, list(tasks)
        self.prefix_layout = prefix_layout
        self.task_tokens = dict(task_tokens or {})
        self.task_token_ids = {
            task: tokenizer.convert_tokens_to_ids(marker)
            for task, marker in self.task_tokens.items()
        }
        if any(marker not in tokenizer.get_vocab() for marker in self.task_tokens.values()):
            raise ValueError("Checkpoint tokenizer is missing a task token")
        self.hidden_size = backbone.config.hidden_size
        self.heads = nn.ModuleDict({task: nn.Linear(self.hidden_size, 1) for task in tasks})
        self.choice_interaction_config = choice_interaction
        self.choice_interaction = (
            ChoiceInteraction(self.hidden_size, **choice_interaction)
            if choice_interaction
            else None
        )
        self._compiled_forward = None
        self.settings = dict(
            tasks=list(tasks),
            query_length=query_length,
            document_length=document_length,
            context_length=capacity,
            query_truncation=query_truncation,
            task_tokens=self.task_tokens,
            choice_interaction=choice_interaction,
            prefix_layout=prefix_layout,
        )

    def compile_inference(self, *, mode="default", dynamic=True, backend="inductor"):
        """Enable lazy compilation of encoder.forward and return this runtime.

        mode, dynamic and backend are forwarded to torch.compile; defaults are
        "default", True and "inductor". Heads, tokenization, rendering and output
        interpretation stay eager. First calls pay compilation cost and changing
        shapes may recompile. Warm representative batches before timing. Compile
        failures propagate; call disable_compile() to explicitly use eager mode.
        """
        self.eval().requires_grad_(False)
        self._compiled_forward = torch.compile(
            self.encoder.forward, mode=mode, dynamic=dynamic, backend=backend
        )
        return self

    def disable_compile(self):
        """Clear the compiled encoder callable and restore eager execution; return None."""
        self._compiled_forward = None

    def preprocess(self, inputs, prompt=None, **kwargs):
        """Reject raw-text encode(); typed decisions require predict() or predict_groups()."""
        raise ValueError(
            "Typed decisions require candidate groups; use model.predict(input_object)"
        )

    @staticmethod
    def _validate_limits(query_length, document_length, capacity):
        """Validate branch caps including special tokens against shared positional capacity."""
        if not 3 <= query_length <= capacity - 2 or not 2 <= document_length <= capacity - 3:
            raise ValueError("Invalid query/document limits for this backbone")

    def _limits(self, query_length, document_length, context_length=None):
        """Resolve per-call caps, clamping saved defaults to a smaller requested context."""
        capacity = self.context_length if context_length is None else context_length
        if not 5 <= capacity <= self.encoder.backbone.config.max_position_embeddings:
            raise ValueError("Context length exceeds the backbone positional capacity")
        q = min(self.query_length, capacity - 2) if query_length is None else query_length
        d = min(self.document_length, capacity - 3) if document_length is None else document_length
        self._validate_limits(q, d, capacity)
        return q, d, capacity

    def _documents(self, documents, tasks, limit):
        """Tokenize candidates with task markers and final SEP, right-truncated to limit."""
        if len(tasks) != len(documents):
            raise ValueError("Tasks and candidates must align")
        if not documents:
            return []
        encoded = self.tokenizer(
            documents,
            add_special_tokens=False,
            truncation=True,
            max_length=limit - 1,
        )["input_ids"]
        return [
            [self.task_token_ids[t], *d[: limit - 2], self.tokenizer.sep_token_id]
            if t in self.task_token_ids
            else [*d, self.tokenizer.sep_token_id]
            for t, d in zip(tasks, encoded, strict=True)
        ]

    def _queries(self, queries, parts, limits):
        """Tokenize queries under per-query caps with balanced or right truncation."""
        if not queries:
            return []
        if self.query_truncation == "balanced":
            if (
                parts is None
                or len(parts) != len(queries)
                or any(p is None or p.render() != q for p, q in zip(parts, queries, strict=True))
            ):
                raise ValueError("Balanced truncation requires matching QueryParts")
            return balanced_query_ids(self.tokenizer, parts, limits)
        encoded = self.tokenizer(
            queries,
            add_special_tokens=False,
            truncation=True,
            max_length=max(limits) - 2,
        )["input_ids"]
        return [
            [self.tokenizer.cls_token_id, *q[: limit - 2], self.tokenizer.sep_token_id]
            for q, limit in zip(encoded, limits, strict=True)
        ]

    def tokenize_branches(
        self,
        queries,
        documents,
        document_tasks=None,
        *,
        query_parts=None,
        query_length=None,
        document_length=None,
        context_length=None,
    ):
        """Tokenize separate branch lists and return (query_ids, document_ids).

        This low-level helper has no decision ownership, so allocation uses the
        longest branches conservatively. Use prepare_groups() for per-decision
        adaptive allocation or predict() for the full native-input interface.
        Balanced truncation requires query_parts matching each rendered query.
        """
        qlimit, dlimit, capacity = self._limits(query_length, document_length, context_length)
        dids = self._documents(documents, document_tasks or ["reranker"] * len(documents), dlimit)
        qids = self._queries(queries, query_parts, [qlimit] * len(queries))
        dsize, qsize = allocate_query_budget(
            max(map(len, dids), default=2),
            max(map(len, qids), default=3),
            capacity,
        )
        if any(len(q) > qsize for q in qids):
            qids = self._queries(queries, query_parts, [qsize] * len(queries))
        return qids, [d[: dsize - 1] + [d[-1]] if len(d) > dsize else d for d in dids]

    def prepare_groups(
        self, groups, *, query_length=None, document_length=None, context_length=None
    ):
        """Share positions equally, then lend unused capacity; candidates win odd tokens.

        Candidate caps apply before sharing. All candidates in a decision see the
        same query. Allocation does not depend on neighboring decisions or batches.
        Special tokens count toward limits. Training settings are not modified.
        """
        qlimit, dlimit, capacity = self._limits(query_length, document_length, context_length)
        documents = list(dict.fromkeys((g.task, d) for g in groups for d in g.candidates))
        dids = self._documents([d for _, d in documents], [t for t, _ in documents], dlimit)
        dmap = dict(zip(documents, dids, strict=True))
        queries = list(dict.fromkeys((g.query, g.query_parts) for g in groups))
        qids = self._queries(
            [q for q, _ in queries], [p for _, p in queries], [qlimit] * len(queries)
        )
        original = dict(zip(queries, qids, strict=True))
        keys, document_limits = [], []
        for g in groups:
            if not g.candidates:
                raise ValueError("A decision requires at least one candidate")
            ds, qs = allocate_query_budget(
                max(len(dmap[g.task, d]) for d in g.candidates),
                len(original[g.query, g.query_parts]),
                capacity,
            )
            keys.append((g.query, g.query_parts, qs))
            document_limits.append(ds)
        unique = list(dict.fromkeys(keys))
        qmap = {key: original[key[:2]] for key in unique if len(original[key[:2]]) <= key[2]}
        reduced = [key for key in unique if key not in qmap]
        encoded = self._queries(
            [q for q, _, _ in reduced], [p for _, p, _ in reduced], [n for _, _, n in reduced]
        )
        qmap.update(zip(reduced, encoded, strict=True))
        return [
            PreparedGroup(
                tuple(qmap[key]),
                g.task,
                qmap[key],
                [
                    dmap[g.task, d][: limit - 1] + [dmap[g.task, d][-1]]
                    if len(dmap[g.task, d]) > limit
                    else dmap[g.task, d]
                    for d in g.candidates
                ],
                g.target,
                g.metadata,
            )
            for g, key, limit in zip(groups, keys, document_limits, strict=True)
        ]

    def collate_tokens(self, queries, documents, owners):
        """Pad token lists on CPU and map each document to a unique prefix via owners."""

        def pad(rows):
            ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r, dtype=torch.long) for r in rows],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            return ids, torch.arange(ids.shape[1])[None] < torch.tensor([len(r) for r in rows])[
                :, None
            ]

        pi, pm = pad(queries)
        di, dm = pad(documents)
        return dict(
            prefix_ids=pi,
            prefix_mask=pm,
            doc_ids=di,
            doc_mask=dm,
            owners=torch.tensor(owners, dtype=torch.long),
        )

    def forward(self, features, **kwargs):
        """Score collated tensor features and return the same dict with raw logits.

        Requires prefix_ids, prefix_mask, doc_ids, doc_mask and owners; multiple
        heads require head_indices. Optional Choice interaction also uses its
        group indices/mask. Adds scores and sentence_embedding, both (N, 1).
        These are logits, not typed predictions or normalized probabilities.
        Prefer predict()/predict_groups(), which also manage batching and
        inference mode. CUDA encoder autocast is BF16; task heads use FP32.
        """
        device = next(self.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            execute = self._compiled_forward or self.encoder
            hidden = execute(
                **{
                    k: features[k]
                    for k in ("prefix_ids", "prefix_mask", "doc_ids", "doc_mask", "owners")
                }
            )
            mask = features["doc_mask"]
            hidden = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1)[:, None]
        with torch.autocast(device.type, enabled=False):
            indices = features.get("head_indices")
            if indices is None:
                if len(self.tasks) != 1:
                    raise ValueError("Multi-task forward requires head_indices")
                indices = {self.tasks[0]: None}
            scores = hidden.new_zeros((hidden.shape[0], 1), dtype=torch.float32)
            for task, index in indices.items():
                if task not in self.heads:
                    raise ValueError(f"Checkpoint has no {task} head")
                if index is None:
                    scores = self.heads[task](hidden.float())
                else:
                    ix = torch.as_tensor(index, device=device, dtype=torch.long)
                    scores = scores.index_copy(
                        0, ix, self.heads[task](hidden.index_select(0, ix).float())
                    )
            if self.choice_interaction is not None and "choice" in indices:
                scores = scores + self.choice_interaction(
                    hidden.float(), features["choice_indices"], features["choice_mask"]
                )
        features["scores"] = features["sentence_embedding"] = scores
        return features

    @torch.inference_mode()
    def predict_groups(
        self,
        groups,
        *,
        batch_size=128,
        token_budget=64000,
        query_length=None,
        document_length=None,
        context_length=None,
        show_progress_bar=True,
    ):
        """Predict probability tensors for already-rendered decision groups.

        Use predict(list_of_requests) for ordinary native-input batching. This
        lower-level API accepts an iterable of Group objects (materialized),
        such as input_groups(request), and skips typed result interpretation.
        With balanced query truncation, each group needs matching QueryParts.

        Returns a list of CPU FP32 tensors, one per group, each shaped
        (number_of_candidates,) and normalized within that group. Group order
        and candidate order match the input, regardless of internal bucketing.
        Supported tasks must have heads in this checkpoint; empty input gives [].

        batch_size (128) bounds decisions tokenized/sorted per window.
        token_budget (64,000) bounds estimated padded microbatch work; a decision
        exceeding it runs alone, with all candidates together. Length overrides
        are per call; see predict() and the module's adaptive context description.
        show_progress_bar counts completed decisions on stderr, not requests.
        """
        if batch_size < 1 or token_budget < 1:
            raise ValueError("batch_size and token_budget must be positive")
        groups = list(groups)
        self.eval()
        # Validate overrides even for empty requests.
        _, _, capacity = self._limits(query_length, document_length, context_length)
        if any(g.task not in self.tasks for g in groups):
            raise ValueError("Requested task is absent from this checkpoint")
        device = next(self.parameters()).device
        result = [None] * len(groups)
        with tqdm(total=len(groups), desc="Batches", disable=not show_progress_bar) as progress:
            for start in range(0, len(groups), batch_size):
                prepared = self.prepare_groups(
                    groups[start : start + batch_size],
                    query_length=query_length,
                    document_length=document_length,
                    context_length=context_length,
                )
                ordered = sorted(
                    enumerate(prepared),
                    key=lambda pair: (
                        len(pair[1].query),
                        max(map(len, pair[1].documents)),
                    ),
                    reverse=True,
                )
                batch, positions = [], []
                count = max_query = max_document = 0

                def flush():
                    scores = self(to_device(collate_groups(batch, self), device))[
                        "scores"
                    ].flatten()
                    lengths = [len(g.documents) for g in batch]
                    # One host transfer/synchronization per microbatch, including
                    # the finite check; preserve FP32 per-decision softmax.
                    probabilities = torch.cat(
                        [p.float().softmax(0) for p in scores.split(lengths)]
                    ).cpu()
                    if not torch.isfinite(probabilities).all():
                        raise FloatingPointError("Nonfinite inference probabilities")
                    for index, values in zip(positions, probabilities.split(lengths), strict=True):
                        result[start + index] = values
                    progress.update(len(batch))

                for index, group in ordered:
                    nq = max(max_query, len(group.query))
                    nd = max(max_document, max(map(len, group.documents)))
                    nc = count + len(group.documents)
                    if batch and (nc * (nq + nd) > token_budget or nq + nd > capacity):
                        flush()
                        batch, positions = [], []
                        nq, nd, nc = (
                            len(group.query),
                            max(map(len, group.documents)),
                            len(group.documents),
                        )
                    batch.append(group)
                    positions.append(index)
                    max_query, max_document, count = nq, nd, nc
                if batch:
                    flush()
        return result

    @staticmethod
    def _interpret(group, probabilities):
        """Convert one group distribution into typed fields or stable relative ranking."""
        assert group.metadata is not None and group.metadata.candidate_ids is not None
        if group.metadata.kind == "ranking":
            return {
                "probabilities": dict(
                    zip(
                        group.metadata.candidate_ids,
                        probabilities.tolist(),
                        strict=True,
                    )
                ),
                "order": [
                    group.metadata.candidate_ids[i]
                    for i in probabilities.argsort(descending=True, stable=True).tolist()
                ],
            }
        return asdict(interpret_prediction(group, probabilities))

    def predict(
        self,
        inputs,
        *,
        batch_size=128,
        token_budget=64000,
        query_length=None,
        document_length=None,
        context_length=None,
        prefix_layout=None,
        show_progress_bar=True,
    ):
        """Predict typed decisions for one request or a batch of native requests.

        Args:
            inputs: Dict with exactly state_json and decisions, or an iterable
                of those dicts (materialized as a list). See the module example.
                Use a list for ordinary batched inference, including mixed tasks.
            batch_size: Positive limit on requests rendered per window and on
                decisions tokenized per window (default 128), not candidate count.
            token_budget: Positive padded-work estimate per microbatch (64,000).
                Complete decisions stay together; oversized ones run alone.
                Lower this to reduce microbatch work; it is not a memory cap.
            query_length: Optional query token cap, including special tokens.
            document_length: Optional candidate token cap, including special tokens.
            context_length: Optional shared query/candidate positional limit.
                All length overrides apply only to this call; see adaptive-v1
                allocation in the module docstring.
            prefix_layout: Optional instruction_state or state_instruction;
                None uses the exported rendering order.
            show_progress_bar: Show completed requests on stderr (default True).

        Returns:
            One dict keyed by decision ID for a dict input, otherwise a list of
            those dicts in input order. Each value contains the typed fields
            documented above (Choice, Noul, Score, or relative ranking).
            No decisions yields {}; an empty request list yields [].

        For already-rendered Group objects and raw probability tensors, use
        predict_groups(). For repeated GPU workloads, compile_inference() is
        optional; warm representative batches before measuring throughput.
        """
        single = isinstance(inputs, dict)
        cases = [inputs] if single else list(inputs)
        options = dict(
            batch_size=batch_size,
            token_budget=token_budget,
            query_length=query_length,
            document_length=document_length,
            context_length=context_length,
            show_progress_bar=False,
        )
        self.predict_groups([], **options)
        results = []
        with tqdm(total=len(cases), desc="Predict", disable=not show_progress_bar) as progress:
            for start in range(0, len(cases), batch_size):
                groups, owners = [], []
                chunk = cases[start : start + batch_size]
                outputs = [{} for _ in chunk]
                for index, case in enumerate(chunk):
                    rendered = input_groups(case, prefix_layout=prefix_layout or self.prefix_layout)
                    groups.extend(rendered)
                    owners.extend([index] * len(rendered))
                probabilities = self.predict_groups(groups, **options)
                for owner, group, p in zip(owners, groups, probabilities, strict=True):
                    outputs[owner][group.metadata.decision_id] = self._interpret(group, p)
                results.extend(outputs)
                progress.update(len(chunk))
        return results[0] if single else results

    def get_sentence_embedding_dimension(self):
        """Return the ST-compatible scalar logit dimension; this is not a text embedding."""
        return 1

    def get_config_dict(self):
        """Return serializable runtime settings used by the ST module configuration."""
        return self.settings

    def save(self, output_path, *args, **kwargs):
        """Save module config, backbone config, safetensors and tokenizer to a directory."""
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_config(str(path))
        config = self.encoder.backbone.config.to_dict()
        config.pop("_name_or_path", None)
        (path / "backbone_config.json").write_text(json.dumps(config, indent=2))
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(path / "model.safetensors"),
        )
        self.save_tokenizer(str(path / "tokenizer"))

    @classmethod
    def load(
        cls,
        model_name_or_path,
        subfolder="",
        token=None,
        cache_folder=None,
        revision=None,
        local_files_only=False,
        init_defaults=None,
        **kwargs,
    ):
        """Load this ST module from a local directory or Hugging Face repository.

        Called by SentenceTransformer with subfolder, authentication, revision,
        cache and offline settings. Restores FP32 weights with strict matching,
        SDPA attention, eval mode and gradients disabled. Prefer constructing
        BekkoSentenceTransformer for the public typed interface and device setup.
        """
        hub: dict[str, Any] = dict(
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        settings = cls.load_config(model_name_or_path, **hub)
        config_path = cls.load_file_path(model_name_or_path, "backbone_config.json", **hub)
        weights = cls.load_file_path(model_name_or_path, "model.safetensors", **hub)
        tokenizer_hub: dict[str, Any] = {**hub, "subfolder": f"{subfolder}/tokenizer".lstrip("/")}
        tok = cls.load_dir_path(model_name_or_path, **tokenizer_hub)
        if config_path is None or weights is None or tok is None:
            raise FileNotFoundError("Incomplete portable checkpoint")
        config = json.loads(Path(config_path).read_text())
        backbone = AutoModel.from_config(
            AutoConfig.for_model(config.pop("model_type"), **config),
            attn_implementation="sdpa",
            dtype=torch.float32,
        )
        model = cls(backbone, AutoTokenizer.from_pretrained(tok, local_files_only=True), **settings)
        model.load_state_dict(load_file(weights), strict=True)
        return model.eval().requires_grad_(False)


class BekkoSentenceTransformer(SentenceTransformer):
    """SentenceTransformer with a public typed-decision inference interface.

    Load an exported v0 checkpoint using the usual SentenceTransformer constructor
    arguments, including device, revision, local_files_only and trust_remote_code.
    The same class is included in the standalone exported inference_v0.py.
    """

    def __init__(self, *args, **kwargs):
        """Load an exported v0 model with standard ST arguments, then validate its module."""
        super().__init__(*args, **kwargs)
        # Remote-code loading creates a distinct Python class identity, so check
        # the portable module contract instead of using isinstance.
        if len(self) != 1 or getattr(self[0], "config_file_name", None) != "inference_config.json":
            raise ValueError("BekkoSentenceTransformer requires an exported v0 checkpoint")

    def predict(
        self,
        inputs,
        *,
        batch_size=128,
        token_budget=64000,
        query_length=None,
        document_length=None,
        context_length=None,
        prefix_layout=None,
        show_progress_bar=True,
    ):
        """Predict typed decisions for one request or a batch of native requests.

        Args:
            inputs: Dict with exactly state_json and decisions, or an iterable
                of those dicts (materialized as a list). See the module example.
                Use a list for ordinary batched inference, including mixed tasks.
            batch_size: Positive limit on requests rendered per window and on
                decisions tokenized per window (default 128), not candidate count.
            token_budget: Positive padded-work estimate per microbatch (64,000).
                Complete decisions stay together; oversized ones run alone.
                Lower this to reduce microbatch work; it is not a memory cap.
            query_length: Optional query token cap, including special tokens.
            document_length: Optional candidate token cap, including special tokens.
            context_length: Optional shared query/candidate positional limit.
                All length overrides apply only to this call; see adaptive-v1
                allocation in the module docstring.
            prefix_layout: Optional instruction_state or state_instruction;
                None uses the exported rendering order.
            show_progress_bar: Show completed requests on stderr (default True).

        Returns:
            One dict keyed by decision ID for a dict input, otherwise a list of
            those dicts in input order. Each value contains the typed fields
            documented above (Choice, Noul, Score, or relative ranking).
            No decisions yields {}; an empty request list yields [].

        For already-rendered Group objects and raw probability tensors, use
        predict_groups(). For repeated GPU workloads, compile_inference() is
        optional; warm representative batches before measuring throughput.
        """
        return cast(Any, self[0]).predict(
            inputs,
            batch_size=batch_size,
            token_budget=token_budget,
            query_length=query_length,
            document_length=document_length,
            context_length=context_length,
            prefix_layout=prefix_layout,
            show_progress_bar=show_progress_bar,
        )

    def predict_groups(
        self,
        groups,
        *,
        batch_size=128,
        token_budget=64000,
        query_length=None,
        document_length=None,
        context_length=None,
        show_progress_bar=True,
    ):
        """Predict probability tensors for already-rendered decision groups.

        Use predict(list_of_requests) for ordinary native-input batching. This
        lower-level API accepts an iterable of Group objects (materialized),
        such as input_groups(request), and skips typed result interpretation.
        With balanced query truncation, each group needs matching QueryParts.

        Returns a list of CPU FP32 tensors, one per group, each shaped
        (number_of_candidates,) and normalized within that group. Group order
        and candidate order match the input, regardless of internal bucketing.
        Supported tasks must have heads in this checkpoint; empty input gives [].

        batch_size (128) bounds decisions tokenized/sorted per window.
        token_budget (64,000) bounds estimated padded microbatch work; a decision
        exceeding it runs alone, with all candidates together. Length overrides
        are per call; see predict() and the module's adaptive context description.
        show_progress_bar counts completed decisions on stderr, not requests.
        """
        return cast(Any, self[0]).predict_groups(
            groups,
            batch_size=batch_size,
            token_budget=token_budget,
            query_length=query_length,
            document_length=document_length,
            context_length=context_length,
            show_progress_bar=show_progress_bar,
        )

    def compile_inference(self, *, mode="default", dynamic=True, backend="inductor"):
        """Enable lazy encoder compilation and return self for optional chaining.

        For repeated inference, call once before representative warmup batches.
        mode="default", dynamic=True and backend="inductor" pass to torch.compile.
        Shapes may recompile; first-call latency includes compilation. Rendering,
        tokenization, heads and typed outputs stay eager. See the module guide.
        """
        cast(Any, self[0]).compile_inference(mode=mode, dynamic=dynamic, backend=backend)
        return self

    def disable_compile(self):
        """Restore eager tensor execution and return this model."""
        cast(Any, self[0]).disable_compile()
        return self


def main():
    """Run standalone inference from a JSON request or request array.

    --model accepts a local export or Hub ID; private Hub access uses saved
    authentication. --compile enables lazy encoder compilation. Writes results
    to stdout and optional request progress to stderr. See --help for controls.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", required=True, type=Path, help="JSON file containing only native inference input"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--token-budget", type=int, default=64000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-length", type=int)
    parser.add_argument("--document-length", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--show-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model = BekkoSentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.compile:
        model.compile_inference()
    print(
        json.dumps(
            model.predict(
                json.loads(args.input.read_text()),
                token_budget=args.token_budget,
                batch_size=args.batch_size,
                query_length=args.query_length,
                document_length=args.document_length,
                context_length=args.context_length,
                show_progress_bar=args.show_progress_bar,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
