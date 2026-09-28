"""Portable v0 inference. Exported checkpoints inline the small relative helpers.

Runtime dependencies are torch, transformers, sentence-transformers and safetensors.
No training package, datasets, PEFT, W&B or external attention extension is required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from sentence_transformers.base.modules import InputModule
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from .choice import ChoiceInteraction
from .data import DecisionMetadata, Group, collate_groups, prepare_groups, to_device, work_tokens
from .decisions import interpret_prediction
from .query_budget import QueryParts, balanced_query_ids


def _text(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def input_groups(case_input, *, prefix_layout="instruction_state"):
    """Render native inference input; labels and row metadata are never accepted."""
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
        super().__init__()
        self.backbone = backbone

    def qkv(self, layer, hidden, positions):
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
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(2)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
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

    def __init__(
        self,
        backbone,
        tokenizer,
        *,
        tasks,
        query_length=4096,
        document_length=2048,
        query_truncation="balanced",
        task_tokens=None,
        choice_interaction=None,
        prefix_layout="instruction_state",
    ):
        super().__init__()
        if backbone.config.model_type != "modernbert":
            raise ValueError("Expected ModernBERT-compatible weights")
        if (
            query_length < 3
            or document_length < 2
            or query_length + document_length > backbone.config.max_position_embeddings
        ):
            raise ValueError("Invalid query/document limits for this backbone")
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
            query_truncation=query_truncation,
            task_tokens=self.task_tokens,
            choice_interaction=choice_interaction,
            prefix_layout=prefix_layout,
        )

    def compile_inference(self, *, mode="default", dynamic=True, backend="inductor"):
        """Compile tensor execution lazily; first calls include compilation cost.

        Tokenization/rendering stay eager. No silent eager fallback is enabled.
        Changing shapes can trigger new compilations despite dynamic=True.
        """
        self.eval().requires_grad_(False)
        self._compiled_forward = torch.compile(
            self.encoder.forward, mode=mode, dynamic=dynamic, backend=backend
        )
        return self

    def disable_compile(self):
        self._compiled_forward = None

    def preprocess(self, inputs, prompt=None, **kwargs):
        raise ValueError(
            "Typed decisions require candidate groups; use model[0].predict(input_object)"
        )

    def tokenize_branches(self, queries, documents, document_tasks=None, *, query_parts=None):
        tasks = document_tasks or ["reranker"] * len(documents)
        if len(tasks) != len(documents):
            raise ValueError("Tasks and candidates must align")
        if self.query_truncation == "balanced":
            if (
                query_parts is None
                or len(query_parts) != len(queries)
                or any(
                    p is None or p.render() != q for p, q in zip(query_parts, queries, strict=True)
                )
            ):
                raise ValueError("Balanced truncation requires matching QueryParts")
            qids = balanced_query_ids(self.tokenizer, query_parts, self.query_length)
        else:
            encoded = self.tokenizer(
                queries, add_special_tokens=False, truncation=True, max_length=self.query_length - 2
            )["input_ids"]
            qids = [[self.tokenizer.cls_token_id, *q, self.tokenizer.sep_token_id] for q in encoded]
        encoded = self.tokenizer(
            documents,
            add_special_tokens=False,
            truncation=True,
            max_length=self.document_length - 1,
        )["input_ids"]
        dids = [
            [self.task_token_ids[t], *d[: self.document_length - 2], self.tokenizer.sep_token_id]
            if t in self.task_token_ids
            else [*d, self.tokenizer.sep_token_id]
            for t, d in zip(tasks, encoded, strict=True)
        ]
        return qids, dids

    def collate_tokens(self, queries, documents, owners):
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
    def predict_groups(self, groups, *, token_budget=16000):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        groups = list(groups)
        if not groups:
            return []
        if any(g.task not in self.tasks for g in groups):
            raise ValueError("Requested task is absent from this checkpoint")
        self.eval()
        prepared = prepare_groups(groups, self)
        device = next(self.parameters()).device
        result, batch = [], []

        def flush():
            scores = self(to_device(collate_groups(batch, self), device))["scores"].flatten()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite inference scores")
            result.extend(
                p.float().softmax(0).cpu() for p in scores.split([len(g.documents) for g in batch])
            )

        for group in prepared:
            if batch and work_tokens([*batch, group]) > token_budget:
                flush()
                batch = []
            batch.append(group)
        if batch:
            flush()
        return result

    def predict(self, case_input, *, prefix_layout=None, token_budget=16000):
        groups = input_groups(case_input, prefix_layout=prefix_layout or self.prefix_layout)
        probabilities = self.predict_groups(groups, token_budget=token_budget)
        result = {}
        for group, p in zip(groups, probabilities, strict=True):
            assert group.metadata is not None and group.metadata.candidate_ids is not None
            if group.metadata.kind == "ranking":
                result[group.metadata.decision_id] = {
                    "probabilities": dict(
                        zip(group.metadata.candidate_ids, p.tolist(), strict=True)
                    ),
                    "order": [
                        group.metadata.candidate_ids[i]
                        for i in p.argsort(descending=True, stable=True).tolist()
                    ],
                }
            else:
                result[group.metadata.decision_id] = asdict(interpret_prediction(group, p))
        return result

    def get_sentence_embedding_dimension(self):
        return 1

    def get_config_dict(self):
        return self.settings

    def save(self, output_path, *args, **kwargs):
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


def main():
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", required=True, type=Path, help="JSON file containing only native inference input"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--token-budget", type=int, default=16000)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    runtime = cast(Any, model[0])
    if args.compile:
        runtime.compile_inference()
    print(
        json.dumps(
            runtime.predict(json.loads(args.input.read_text()), token_budget=args.token_budget),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
