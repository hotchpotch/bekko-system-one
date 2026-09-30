"""Standalone Bekko v0 typed-decision inference and remote-code interface.

Maintainer note: this source file is already the standalone runtime.
export_v0.py:runtime_source() copies it verbatim into exported model directories;
it does not convert imports or inline helpers. The same file can be distributed
with compatible v0 weights on Hugging Face. Do not add relative imports or
imports of bekko_system_one: consumers must not need the training package.
Keep embedded helpers aligned with their training counterparts and preserve the
isolated-process standalone loading test in tests/test_inference_v0.py.

Install the exported requirements.txt (and a CUDA-compatible PyTorch build for
GPU use). This source file includes its helpers and needs no Bekko training
package, datasets, PEFT or W&B. SDPA needs no external FlashAttention extension;
FA2 is an optional acceleration backend selected at model loading.

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
BekkoSentenceTransformer(..., attn_implementation="auto") is the default:
CUDA capability 8.0+ with a compatible flash-attn library selects FlashAttention 2;
otherwise it selects PyTorch SDPA. Force a backend at model load time::

    model = Model(repo, revision=revision, device="cuda", trust_remote_code=True,
                  attn_implementation="flash_attention_2")
    # Or attn_implementation="sdpa" to require SDPA without importing flash-attn.
    # model_kwargs={"attn_implementation": "flash_attention_2"} is also accepted.
    print(model[0].attn_implementation)  # The backend actually selected.

Explicit FA2 requires CUDA capability 8.0+ and a compatible flash-attn wheel;
missing or binary-incompatible libraries raise during model loading, never silently
fall back. Install the optional fa2 extra in the training package, or a flash-attn
wheel matching your inference environment's Python, PyTorch and CUDA versions.
SDPA has no external attention dependency. Auto selection happens at loading;
load again with the desired device/backend when switching devices. Plain ST can
load the format, but use BekkoSentenceTransformer for backend selection.

For 17M models, SDPA and FA2 generally have little speed difference; SDPA is a
reasonable dependency-free choice. For 68M and larger models, prefer FA2 for
throughput, especially on long inputs. This is a sizing guideline, not a speed
guarantee for every larger checkpoint: measure representative inputs on your GPU,
excluding loading and warmup. BF16 backend rounding can change probabilities and
occasionally the selected candidate; the backends are not bitwise interchangeable.
The SDPA path reuses per-forward masks/rotary tensors and blocks local prefix
attention; FA2 keeps valid tokens packed through attention and feed-forward layers.
Both retain the same rendering, input budgets, candidate order and task heads.
CUDA encoder execution uses BF16 autocast; CPU uses FP32.
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
Add --device cuda, --attn-implementation flash_attention_2 (or sdpa/auto),
--compile, --batch-size 128, --token-budget 64000, or
--no-show-progress-bar as needed. Input JSON can be one request or an array;
results are JSON on stdout and progress is on stderr.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class QueryParts:
    """Explicit boundaries: never infer them from text inside the context."""

    instruction: str
    context: str
    system: str = ""
    layout: str = "instruction_state"

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be nonempty text")
        if not isinstance(self.context, str) or not isinstance(self.system, str):
            raise ValueError("context and system must be text")
        if self.layout not in {"instruction_state", "state_instruction"}:
            raise ValueError("Unsupported query layout")

    def render(self):
        prefix = f"{self.system}\n\n" if self.system.strip() else ""
        instruction = f"Instruction: {self.instruction}"
        context = f"State: {self.context}"
        body = (
            (instruction, context) if self.layout == "instruction_state" else (context, instruction)
        )
        return prefix + "\n".join(body)


def allocate_query_budget(instruction_length, context_length, budget):
    """Reserve half per field, transfer unused capacity, favor instruction on odd budgets."""
    if min(instruction_length, context_length, budget) < 0:
        raise ValueError("Lengths and budget must be nonnegative")
    instruction = min(instruction_length, (budget + 1) // 2)
    context = min(context_length, budget // 2)
    instruction += min(instruction_length - instruction, budget - instruction - context)
    context += min(context_length - context, budget - instruction - context)
    return instruction, context


def balanced_query_ids(tokenizer, parts, query_length):
    """Reserve system/markers first, then share the remaining budget between fields.

    Tokenize components independently so both layouts retain exactly the same
    content tokens. Truncate field tails, without decoding and re-tokenizing.
    """
    if tokenizer.truncation_side != "right":
        raise ValueError("balanced query truncation requires a right-truncating tokenizer")
    if any(not isinstance(p, QueryParts) for p in parts):
        raise ValueError("balanced query truncation requires QueryParts for every query")
    if not parts:
        return []
    limits = [query_length] * len(parts) if isinstance(query_length, int) else list(query_length)
    if len(limits) != len(parts) or any(limit < 3 for limit in limits):
        raise ValueError("One valid query budget is required per query")
    markers = tokenizer(["Instruction: ", "State: ", "\n"], add_special_tokens=False)["input_ids"]
    instruction_marker, context_marker, separator = markers
    texts = []
    for p in parts:
        texts.extend([f"{p.system}\n\n" if p.system.strip() else "", p.instruction, p.context])
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max(limits))[
        "input_ids"
    ]
    result = []
    overhead = 2 + sum(map(len, markers))
    for i, p in enumerate(parts):
        system, instruction, context = encoded[3 * i : 3 * i + 3]
        budget = limits[i] - overhead - len(system)
        if budget < 2:
            raise ValueError("System prompt and query markers leave fewer than two content tokens")
        ni, nc = allocate_query_budget(len(instruction), len(context), budget)
        ins = instruction_marker + instruction[:ni]
        ctx = context_marker + context[:nc]
        body = ins + separator + ctx if p.layout == "instruction_state" else ctx + separator + ins
        result.append([tokenizer.cls_token_id, *system, *body, tokenizer.sep_token_id])
    return result


@dataclass(frozen=True)
class DecisionMetadata:
    """Non-tokenized candidate alignment and source identity; never model features."""

    candidate_ids: tuple[str, ...] | None = None
    candidate_values: tuple[float, ...] | None = None
    kind: str | None = None
    case_id: str | None = None
    group_id: str | None = None
    decision_id: str | None = None

    def __post_init__(self):
        if self.kind not in {None, "judgment", "ranking"}:
            raise ValueError("metadata kind must be judgment or ranking")
        if self.candidate_ids is not None:
            ids = tuple(self.candidate_ids)
            if not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("candidate_ids must be nonempty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("candidate_ids must be unique")
            object.__setattr__(self, "candidate_ids", ids)
        if self.candidate_values is not None:
            values = tuple(self.candidate_values)
            if not values or any(not math.isfinite(v) for v in values):
                raise ValueError("candidate_values must be finite")
            if len(set(values)) != len(values):
                raise ValueError("candidate_values must be unique")
            object.__setattr__(self, "candidate_values", values)
        for identity in (self.case_id, self.group_id, self.decision_id):
            if identity is not None and not isinstance(identity, str):
                raise ValueError("metadata identities must be strings")

    @property
    def yes_index(self):
        ids = self.candidate_ids
        if ids is not None and set(ids) in ({"true", "false"}, {"yes", "no"}):
            return ids.index("true" if "true" in ids else "yes")
        return None

    @property
    def score_values(self):
        if self.kind == "judgment" and self.candidate_values is not None:
            if len(self.candidate_values) >= 2:
                return self.candidate_values
        return None


@dataclass(frozen=True)
class Group:
    query: str
    candidates: list[str]
    task: str = "reranker"
    target: list[float] | None = None
    query_parts: QueryParts | None = None
    metadata: DecisionMetadata | None = None

    def __post_init__(self):
        if self.metadata is not None:
            for values in (self.metadata.candidate_ids, self.metadata.candidate_values):
                if values is not None and len(values) != len(self.candidates):
                    raise ValueError("candidate metadata must align with candidates")
        if self.query_parts is not None and self.query_parts.render() != self.query:
            raise ValueError("query must match query_parts.render()")
        if self.task not in {"reranker", "choice", "noul", "score"}:
            raise ValueError(f"Unknown task: {self.task}")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be nonempty text")
        if not self.candidates or any(not isinstance(c, str) for c in self.candidates):
            raise ValueError("candidates must contain text")
        if self.target is not None and (
            len(self.target) != len(self.candidates)
            or any(not math.isfinite(t) or t < 0 for t in self.target)
            or abs(sum(self.target) - 1) > 1e-5
        ):
            raise ValueError("target must be a normalized distribution aligned with candidates")

    @classmethod
    def from_dict(cls, row):
        # Metadata and targets are never concatenated into inference inputs.
        return cls(
            query=row["query"],
            candidates=row["candidates"],
            task=row.get("task", "reranker"),
            target=row.get("target"),
            query_parts=QueryParts(**row["query_parts"]) if row.get("query_parts") else None,
            metadata=DecisionMetadata(**row["metadata"]) if row.get("metadata") else None,
        )


@dataclass
class PreparedGroup:
    key: str | tuple[int, ...]
    task: str
    query: list[int]
    documents: list[list[int]]
    target: list[float] | None
    metadata: DecisionMetadata | None = None

    @property
    def cost(self):
        return len(self.query) * len(self.documents) + sum(map(len, self.documents))


def prepare_groups(groups, encoder):
    queries = list(dict.fromkeys((g.query, g.query_parts) for g in groups))
    documents = list(dict.fromkeys((g.task, d) for g in groups for d in g.candidates))
    qids, dids = encoder.tokenize_branches(
        [q for q, _ in queries],
        [d for _, d in documents],
        [t for t, _ in documents],
        query_parts=[p for _, p in queries],
    )
    qmap, dmap = dict(zip(queries, qids, strict=True)), dict(zip(documents, dids, strict=True))
    return [
        PreparedGroup(
            tuple(qmap[g.query, g.query_parts]),
            g.task,
            qmap[g.query, g.query_parts],
            [dmap[g.task, d] for d in g.candidates],
            g.target,
            g.metadata,
        )
        for g in groups
    ]


def collate_groups(groups, encoder):
    queries, docs, owners, seen, indices = [], [], [], {}, {}
    for group in groups:
        if group.key not in seen:
            seen[group.key] = len(queries)
            queries.append(group.query)
        indices.setdefault(group.task, []).extend(
            range(len(docs), len(docs) + len(group.documents))
        )
        docs.extend(group.documents)
        owners.extend([seen[group.key]] * len(group.documents))
    features = encoder.collate_tokens(queries, docs, owners)
    features["head_indices"] = {next(iter(indices)): None} if len(indices) == 1 else indices
    choice_rows, offset = [], 0
    for group in groups:
        count = len(group.documents)
        if group.task == "choice":
            choice_rows.append(list(range(offset, offset + count)))
        offset += count
    if choice_rows:
        width = max(map(len, choice_rows))
        features["choice_indices"] = torch.tensor(
            [row + [0] * (width - len(row)) for row in choice_rows], dtype=torch.long
        )
        features["choice_mask"] = torch.tensor(
            [[True] * len(row) + [False] * (width - len(row)) for row in choice_rows]
        )
    return features


def to_device(features, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}


def prepare_batch(model, groups):
    """Build a feature dictionary accepted by an ordinary SentenceTransformer."""
    return to_device(collate_groups(prepare_groups(groups, model[0]), model[0]), model.device)


def work_tokens(items, padded_documents=True):
    cost = sum(x.cost for x in items)
    if padded_documents and items:
        lengths = [len(d) for x in items for d in x.documents]
        cost += len(lengths) * max(lengths) - sum(lengths)
    return cost


def pack_groups(items, budget, padded_documents=True):
    """First-fit decreasing; never split the candidates of a training group."""
    if budget < 1:
        raise ValueError("Token budget must be positive")
    groups, statistics, ordered = [], [], []
    for item in items:
        lengths = [len(d) for d in item.documents]
        ordered.append((item, item.cost, len(lengths), sum(lengths), max(lengths)))
    for item, cost, count, total, maximum in sorted(
        ordered, key=lambda x: (x[4], x[1]), reverse=True
    ):
        for i, (old_cost, old_count, old_total, old_max) in enumerate(statistics):
            combined = old_cost + cost, old_count + count, old_total + total, max(old_max, maximum)
            estimate = combined[0]
            if padded_documents:
                estimate += combined[1] * combined[3] - combined[2]
            if estimate <= budget:
                groups[i].append(item)
                statistics[i] = combined
                break
        else:
            groups.append([item])
            statistics.append((cost, count, total, maximum))
    return groups


@dataclass(frozen=True)
class ChoicePrediction:
    selected_id: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class NoulPrediction:
    probability_yes: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScorePrediction:
    score: float
    normalized_score: float
    probabilities: dict[str, float]
    values: dict[str, float]


def validate_typed_group(group: Group):
    metadata = group.metadata
    if metadata is None or metadata.candidate_ids is None or metadata.kind == "ranking":
        raise ValueError(
            "Typed prediction requires judgment candidate IDs; use predict for ranking"
        )
    if group.task == "noul" and metadata.yes_index is None:
        raise ValueError("Noul requires true/false or yes/no candidate IDs")
    if group.task == "score" and metadata.score_values is None:
        raise ValueError(
            "Score requires judgment metadata and at least two distinct numeric values"
        )
    if group.task not in {"choice", "noul", "score"}:
        raise ValueError("Typed prediction supports choice, noul and ordinal score")


def interpret_prediction(
    group: Group, probabilities
) -> ChoicePrediction | NoulPrediction | ScorePrediction:
    """Map probabilities to IDs and values; never infer numbers from candidate text."""
    validate_typed_group(group)
    p = torch.as_tensor(probabilities, dtype=torch.float64).detach().cpu()
    if (
        p.ndim != 1
        or len(p) != len(group.candidates)
        or not torch.isfinite(p).all()
        or (p < 0).any()
        or abs(p.sum().item() - 1) > 1e-5
    ):
        raise ValueError("Expected normalized probabilities aligned with candidates")
    metadata = group.metadata
    assert metadata is not None and metadata.candidate_ids is not None
    mapping = dict(zip(metadata.candidate_ids, p.tolist(), strict=True))
    if group.task == "choice":
        return ChoicePrediction(metadata.candidate_ids[int(p.argmax())], mapping)
    if group.task == "noul":
        assert metadata.yes_index is not None
        return NoulPrediction(p[metadata.yes_index].item(), mapping)
    values = metadata.score_values
    assert values is not None
    score = sum(prob * value for prob, value in zip(p.tolist(), values, strict=True))
    return ScorePrediction(
        score,
        (score - min(values)) / (max(values) - min(values)),
        mapping,
        dict(zip(metadata.candidate_ids, values, strict=True)),
    )


class ChoiceInteraction(nn.Module):
    """One small attention block, isolated by decision rather than shared prefix.

    Zero-initializing only the final projection preserves the existing scorer.
    No candidate position embeddings or dropout are used.
    """

    def __init__(self, hidden_size, width=128, heads=4):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (width, heads)):
            raise ValueError("Choice width and heads must be positive integers")
        if width % heads:
            raise ValueError("Choice width must be divisible by heads")
        self.width, self.heads = width, heads
        self.project = nn.Linear(hidden_size, width)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_out = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, indices, mask):
        x = self.project(hidden)[indices]
        batch, count, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, count, 3, self.heads, -1)
        q, k, v = qkv.unbind(2)
        attended = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask[:, None, None, :],
            )
            .transpose(1, 2)
            .reshape(batch, count, self.width)
        )
        x = x + self.attention_out(attended)
        x = x + self.ffn(x)
        delta = self.output(x).masked_fill(~mask.unsqueeze(-1), 0)
        return hidden.new_zeros((hidden.shape[0], 1)).index_add(
            0, indices.flatten(), delta.flatten(0, 1)
        )


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
    """Shared-prefix SDPA with per-forward layouts and blocked local attention."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.local_block_size = 256

    def qkv(self, layer, hidden, rotary):
        attn = layer.attn
        q, k, v = (
            attn.Wqkv(layer.attn_norm(hidden))
            .view(*hidden.shape[:2], 3, -1, attn.head_dim)
            .unbind(2)
        )
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *rotary)
        return q.transpose(1, 2), k.transpose(1, 2), v

    @staticmethod
    def layout(qmask, kmask, window):
        allowed = kmask[:, None, :]
        if window is not None:
            qp = qmask.long().cumsum(1) - 1 + (kmask.sum(1) - qmask.sum(1))[:, None]
            kp = kmask.long().cumsum(1) - 1
            allowed = allowed & ((qp[:, :, None] - kp[:, None, :]).abs() <= window)
        return allowed[:, None]

    @staticmethod
    def attend(q, k, v, qmask, allowed):
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed,
            dropout_p=0.0,
        ).transpose(1, 2)
        return result * qmask[:, :, None, None]

    def blocked_layout(self, mask, window):
        """Bound local attention work using exact overlapping key windows."""
        length = mask.shape[1]
        block = self.local_block_size
        count = (length + block - 1) // block
        starts = torch.arange(count, device=mask.device)[:, None] * block
        queries = starts + torch.arange(block, device=mask.device)[None]
        keys = starts + torch.arange(-window, block + window, device=mask.device)[None]
        indices = keys.clamp(0, length - 1)
        valid = (keys >= 0) & (keys < length)
        allowed = mask[:, indices][:, :, None, :] & valid[None, :, None, :]
        allowed = allowed & ((queries[:, :, None] - keys[:, None, :]).abs() <= window)[None]
        return indices, allowed[:, :, None], count * block - length

    def blocked_attend(self, q, k, v, mask, layout):
        indices, allowed, padding = layout
        batch, length, heads, dim = q.shape
        count, width = indices.shape
        q = (
            F.pad(q, (0, 0, 0, 0, 0, padding))
            .reshape(batch, count, self.local_block_size, heads, dim)
            .permute(0, 1, 3, 2, 4)
        )
        k = k[:, indices].permute(0, 1, 3, 2, 4)
        v = v[:, indices].permute(0, 1, 3, 2, 4)
        result = F.scaled_dot_product_attention(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            attn_mask=allowed.flatten(0, 1),
            dropout_p=0.0,
        )
        result = (
            result.reshape(batch, count, heads, self.local_block_size, dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, -1, heads, dim)[:, :length]
        )
        return result * mask[:, :, None, None]

    @staticmethod
    def update(layer, hidden, attended):
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(2)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
        prefix = self.backbone.embeddings(prefix_ids)
        hidden = self.backbone.embeddings(doc_ids)
        pp = (prefix_mask.long().cumsum(1) - 1).clamp_min(0)
        pm = prefix_mask[owners]
        dp = (doc_mask.long().cumsum(1) - 1).clamp_min(0) + pm.sum(1)[:, None]
        combined = torch.cat((pm, doc_mask), dim=1)
        layouts, rotations = {}, {}
        for layer in self.backbone.layers:
            kind = layer.attention_type
            window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
            blocked = window is not None and prefix.shape[1] > self.local_block_size + 2 * window
            if window not in layouts:
                pl = (
                    self.blocked_layout(prefix_mask, window)
                    if blocked
                    else self.layout(prefix_mask, prefix_mask, window)
                )
                layouts[window] = pl, self.layout(doc_mask, combined, window)
            if kind not in rotations:
                rotations[kind] = (
                    self.backbone.rotary_emb(prefix, pp, kind),
                    self.backbone.rotary_emb(hidden, dp, kind),
                )
            pq, pk, pv = self.qkv(layer, prefix, rotations[kind][0])
            pl, dl = layouts[window]
            attended = (
                self.blocked_attend(pq, pk, pv, prefix_mask, pl)
                if blocked
                else self.attend(pq, pk, pv, prefix_mask, pl)
            )
            prefix = self.update(layer, prefix, attended)
            q, k, v = self.qkv(layer, hidden, rotations[kind][1])
            k, v = torch.cat((pk[owners], k), 1), torch.cat((pv[owners], v), 1)
            hidden = self.update(layer, hidden, self.attend(q, k, v, doc_mask, dl))
        return self.backbone.final_norm(hidden)


def _load_fa2():
    """Load the optional native wheel, including its binary compatibility check."""
    try:
        return importlib.import_module("flash_attn").flash_attn_varlen_func
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise RuntimeError(
            "flash_attention_2 requires a compatible flash-attn wheel for this PyTorch/CUDA "
            "installation; install the fa2 extra or select attn_implementation='sdpa'"
        ) from error


class _TokenLayout:
    def __init__(self, mask):
        self.indices = mask.flatten().nonzero().flatten()
        self.lengths = mask.sum(1)
        self.cumulative = F.pad(self.lengths.cumsum(0).to(torch.int32), (1, 0))
        self.positions = (mask.long().cumsum(1) - 1).flatten()[self.indices]
        self.maximum = int(self.lengths.max())


class _FA2Prefix(nn.Module):
    """Keep real tokens packed through attention, projections and feed-forward layers."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.attention = _load_fa2()

    def rotary(self, hidden, positions):
        return {
            kind: tuple(
                x.squeeze(0) for x in self.backbone.rotary_emb(hidden, positions[None], kind)
            )
            for kind in set(self.backbone.config.layer_types)
        }

    @staticmethod
    def qkv(layer, hidden, rotary):
        q, k, v = (
            layer.attn.Wqkv(layer.attn_norm(hidden))
            .view(hidden.shape[0], 3, -1, layer.attn.head_dim)
            .unbind(1)
        )
        q, k = apply_rotary_pos_emb(q, k, *rotary, unsqueeze_dim=1)
        return q, k, v

    @staticmethod
    def update(layer, hidden, attended):
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(1)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
        if not prefix_ids.is_cuda:
            raise RuntimeError("FA2 encoder requires CUDA")
        pl, dl = _TokenLayout(prefix_mask), _TokenLayout(doc_mask)
        prefix_lengths = pl.lengths[owners]
        kv_lengths = prefix_lengths + dl.lengths
        cuk = F.pad(kv_lengths.cumsum(0).to(torch.int32), (1, 0))
        total, maximum = int(cuk[-1]), int(kv_lengths.max())
        branches = torch.repeat_interleave(
            torch.arange(len(owners), device=owners.device), kv_lengths, output_size=total
        )
        offsets = torch.arange(total, device=owners.device) - cuk[branches]
        gather = torch.where(
            offsets < prefix_lengths[branches],
            pl.cumulative[owners[branches]] + offsets,
            pl.indices.numel() + dl.cumulative[branches] + offsets - prefix_lengths[branches],
        ).long()
        documents = torch.repeat_interleave(
            torch.arange(len(owners), device=owners.device),
            dl.lengths,
            output_size=dl.indices.numel(),
        )
        prefix = self.backbone.embeddings(prefix_ids.flatten()[pl.indices])
        hidden = self.backbone.embeddings(doc_ids.flatten()[dl.indices])
        pr, dr = (
            self.rotary(prefix, pl.positions),
            self.rotary(hidden, dl.positions + prefix_lengths[documents]),
        )
        for layer in self.backbone.layers:
            kind = layer.attention_type
            window = (
                (-1, -1)
                if layer.attn.sliding_window is None
                else (layer.attn.sliding_window - 1,) * 2
            )
            pq, pk, pv = self.qkv(layer, prefix, pr[kind])
            attended = self.attention(
                pq,
                pk,
                pv,
                pl.cumulative,
                pl.cumulative,
                pl.maximum,
                pl.maximum,
                dropout_p=0.0,
                causal=False,
                window_size=window,
            )
            prefix = self.update(layer, prefix, attended)
            q, k, v = self.qkv(layer, hidden, dr[kind])
            k = torch.cat((pk, k)).index_select(0, gather)
            v = torch.cat((pv, v)).index_select(0, gather)
            attended = self.attention(
                q,
                k,
                v,
                dl.cumulative,
                cuk,
                dl.maximum,
                maximum,
                dropout_p=0.0,
                causal=False,
                window_size=window,
            )
            hidden = self.update(layer, hidden, attended)
        hidden = self.backbone.final_norm(hidden)
        output = hidden.new_zeros((doc_ids.numel(), hidden.shape[-1]))
        return output.index_copy(0, dl.indices, hidden).view(*doc_ids.shape, hidden.shape[-1])


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
        self.attn_implementation = "sdpa"
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

    def set_attention_implementation(self, implementation="auto", *, device=None):
        """Select an encoder without changing weights, budgets or the saved configuration.

        Auto prefers compatible FA2 on CUDA capability 8+ and otherwise uses SDPA.
        Explicit FA2 never falls back. Selection clears any compiled encoder.
        """
        if implementation not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError("attn_implementation must be auto, sdpa or flash_attention_2")
        device = torch.device(device) if device is not None else next(self.parameters()).device
        if implementation != "sdpa":
            supported = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            if not supported:
                if implementation == "flash_attention_2":
                    raise ValueError("flash_attention_2 requires a CUDA GPU with capability 8.0+")
                implementation = "sdpa"
            else:
                try:
                    _load_fa2()
                except RuntimeError:
                    if implementation == "flash_attention_2":
                        raise
                    implementation = "sdpa"
                else:
                    implementation = "flash_attention_2"
        if implementation != self.attn_implementation:
            encoder_class = _FA2Prefix if implementation == "flash_attention_2" else _SDPAPrefix
            self.encoder = encoder_class(self.encoder.backbone).train(self.training)
            self._compiled_forward = None
            self.attn_implementation = implementation
        return self

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
            # Reconstruct at the boundary: callers may use training-side dataclasses.
            parts = [QueryParts(p.instruction, p.context, p.system, p.layout) for p in parts]
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

    def __init__(self, *args, attn_implementation="auto", **kwargs):
        """Load native weights, then select SDPA or optional FA2 on the final device."""
        model_kwargs = dict(kwargs.get("model_kwargs") or {})
        nested = model_kwargs.pop("attn_implementation", None)
        if nested is not None:
            if attn_implementation != "auto" and attn_implementation != nested:
                raise ValueError("Conflicting attn_implementation arguments")
            attn_implementation = nested
        if attn_implementation not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError("attn_implementation must be auto, sdpa or flash_attention_2")
        kwargs["model_kwargs"] = model_kwargs
        super().__init__(*args, **kwargs)
        # Remote-code loading creates a distinct Python class identity, so check
        # the portable module contract instead of using isinstance.
        if len(self) != 1 or getattr(self[0], "config_file_name", None) != "inference_config.json":
            raise ValueError("BekkoSentenceTransformer requires an exported v0 checkpoint")
        cast(Any, self[0]).set_attention_implementation(attn_implementation, device=self.device)

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
    parser.add_argument(
        "--attn-implementation", choices=["auto", "sdpa", "flash_attention_2"], default="auto"
    )
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
        attn_implementation=args.attn_implementation,
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
