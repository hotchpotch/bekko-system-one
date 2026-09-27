"""Sentence Transformers modules for shared-prefix candidate scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from sentence_transformers.base.modules import InputModule, Module
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from .packed_prefix import PackedPrefixEncoder


class SharedPrefix(InputModule):
    """Encode each distinct query once; candidates attend to its per-layer K/V.

    Weights, backbone configuration, and tokenizer are saved together. Loading
    a local checkpoint needs neither the original base model nor network access.
    """

    save_in_root = False
    config_file_name = "shared_prefix.json"
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        model_name_or_path=None,
        *,
        revision=None,
        query_length=512,
        query_truncation="right",
        document_length=1024,
        attention_backend="flash_attention_2",
        gradient_checkpointing=False,
        frozen_linear_bf16=True,
        fused_rotary=True,
        lora=None,
        task_tokens=None,
        backbone=None,
        tokenizer=None,
    ):
        super().__init__()
        if query_length < 3 or document_length < 2:
            raise ValueError("query_length >= 3 and document_length >= 2 are required")
        if attention_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Choose sdpa or flash_attention_2")
        if backbone is None:
            if model_name_or_path is None:
                raise ValueError("Supply model_name_or_path or a backbone and tokenizer")
            backbone = AutoModel.from_pretrained(
                model_name_or_path,
                revision=revision,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
        if backbone.config.model_type != "modernbert":
            raise ValueError("SharedPrefix requires a ModernBERT-compatible encoder")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Expected a Transformers tokenizer")
        if any(
            getattr(tokenizer, k) is None for k in ("cls_token_id", "sep_token_id", "pad_token_id")
        ):
            raise ValueError("Tokenizer requires CLS, SEP, and PAD tokens")
        self.tokenizer = tokenizer
        self.task_tokens = dict(task_tokens or {})
        if (
            any(t not in {"choice", "noul", "score", "reranker"} for t in self.task_tokens)
            or any(not isinstance(v, str) or not v.strip() for v in self.task_tokens.values())
            or len(set(self.task_tokens.values())) != len(self.task_tokens)
        ):
            raise ValueError("task_tokens must map supported tasks to distinct nonempty tokens")
        if self.task_tokens and document_length < 3:
            raise ValueError("Task tokens require document_length >= 3")
        self.task_token_ids = {}
        if self.task_tokens:
            # Fork RNG so vocabulary extension does not change subsequent head initialization.
            # Initialize new markers identically from SEP to isolate learned task identity.
            existing_special = set(tokenizer.all_special_tokens)
            for marker in self.task_tokens.values():
                if marker in tokenizer.get_vocab() and marker not in existing_special:
                    raise ValueError("Task markers must not replace ordinary vocabulary tokens")
                if marker in {
                    tokenizer.cls_token,
                    tokenizer.sep_token,
                    tokenizer.pad_token,
                    tokenizer.unk_token,
                    tokenizer.mask_token,
                    tokenizer.bos_token,
                    tokenizer.eos_token,
                }:
                    raise ValueError("Task markers must be distinct from built-in special tokens")
            old_vocab = tokenizer.get_vocab()
            tokenizer.add_special_tokens(
                {"extra_special_tokens": list(self.task_tokens.values())},
                replace_extra_special_tokens=False,
            )
            self.task_token_ids = {
                task: tokenizer.convert_tokens_to_ids(marker)
                for task, marker in self.task_tokens.items()
            }
            with torch.random.fork_rng(devices=[]):
                if len(tokenizer) > backbone.get_input_embeddings().num_embeddings:
                    backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            with torch.no_grad():
                weight = backbone.get_input_embeddings().weight
                for task, marker in self.task_tokens.items():
                    if marker not in old_vocab:
                        weight[self.task_token_ids[task]].copy_(weight[tokenizer.sep_token_id])
        if query_truncation not in {"right", "balanced"}:
            raise ValueError("query_truncation must be right or balanced")
        self.query_truncation = query_truncation
        self.query_length, self.document_length = query_length, document_length
        self.hidden_size = backbone.config.hidden_size
        self.backbone_config = backbone.config.to_dict()
        # Local initialization paths are never part of the distributable artifact.
        self.backbone_config.pop("_name_or_path", None)
        if lora:
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=lora["rank"],
                    lora_alpha=lora["alpha"],
                    lora_dropout=lora.get("dropout", 0.0),
                    target_modules=lora.get("target_modules", ["Wqkv", "Wo", "Wi"]),
                    bias="none",
                    trainable_token_indices=list(self.task_token_ids.values()) or None,
                ),
            )
        elif frozen_linear_bf16:
            raise ValueError("frozen_linear_bf16 requires LoRA")
        if frozen_linear_bf16:
            from .cuda_training import precast_frozen_linears

            precast_frozen_linears(backbone)
        self.encoder = PackedPrefixEncoder(
            backbone,
            backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.encoder.fused_rotary = fused_rotary
        self.settings = dict(
            query_length=query_length,
            query_truncation=query_truncation,
            document_length=document_length,
            attention_backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
            frozen_linear_bf16=frozen_linear_bf16,
            fused_rotary=fused_rotary,
            lora=lora,
            task_tokens=self.task_tokens,
        )

    def preprocess(self, inputs, prompt=None, **kwargs):
        if prompt:
            raise ValueError("Include instructions in the query of each (query, candidate) pair")
        if not inputs:
            raise ValueError("At least one query/candidate pair is required")
        if any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in inputs):
            raise ValueError("Expected (query, candidate) pairs")
        queries = list(dict.fromkeys(p[0] for p in inputs))
        documents = list(dict.fromkeys(p[1] for p in inputs))
        query_ids, document_ids = self.tokenize_branches(queries, documents)
        lookup = {q: i for i, q in enumerate(queries)}
        docs = dict(zip(documents, document_ids, strict=True))
        return self.collate_tokens(
            query_ids, [docs[p[1]] for p in inputs], [lookup[p[0]] for p in inputs]
        )

    def tokenize_branches(self, queries, documents, document_tasks=None, *, query_parts=None):
        """Prefix each candidate with its configured task marker, within its token budget."""
        if document_tasks is None:
            document_tasks = ["reranker"] * len(documents)
        if len(document_tasks) != len(documents) or any(
            t not in {"choice", "noul", "score", "reranker"} for t in document_tasks
        ):
            raise ValueError("document_tasks must align with candidates and use supported tasks")
        if self.query_truncation == "balanced":
            from .query_budget import balanced_query_ids

            if query_parts is None or len(query_parts) != len(queries):
                raise ValueError("balanced query truncation requires aligned QueryParts")
            if any(p is None or p.render() != q for p, q in zip(query_parts, queries, strict=True)):
                raise ValueError("QueryParts must match the rendered query")
            qids = balanced_query_ids(self.tokenizer, query_parts, self.query_length)
        else:
            tokens = self.tokenizer(
                queries,
                add_special_tokens=False,
                truncation=True,
                max_length=self.query_length - 2,
            )["input_ids"]
            qids = [[self.tokenizer.cls_token_id, *q, self.tokenizer.sep_token_id] for q in tokens]
        dids = self.tokenizer(
            documents,
            add_special_tokens=False,
            truncation=True,
            max_length=self.document_length - 1,
        )["input_ids"]
        return (
            qids,
            [
                [
                    self.task_token_ids[task],
                    *d[: self.document_length - 2],
                    self.tokenizer.sep_token_id,
                ]
                if task in self.task_token_ids
                else [*d, self.tokenizer.sep_token_id]
                for d, task in zip(dids, document_tasks, strict=True)
            ],
        )

    def collate_tokens(self, queries, documents, owners):
        from torch.nn.utils.rnn import pad_sequence

        def pad(rows):
            ids = pad_sequence(
                [torch.tensor(r, dtype=torch.long) for r in rows],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            lengths = torch.tensor([len(r) for r in rows])
            return ids, torch.arange(ids.shape[1])[None] < lengths[:, None]

        pids, pmask = pad(queries)
        dids, dmask = pad(documents)
        return dict(
            prefix_ids=pids,
            prefix_mask=pmask,
            doc_ids=dids,
            doc_mask=dmask,
            owners=torch.tensor(owners, dtype=torch.long),
        )

    @staticmethod
    def pool(hidden, mask):
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1)[:, None]

    def forward(self, features, **kwargs):
        device = next(self.parameters()).device
        # Keep FP32 residuals/norms/adapters, and BF16 matrix multiplication.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if self.settings["frozen_linear_bf16"] and device.type != "cuda":
                raise RuntimeError("Load with model_kwargs={'frozen_linear_bf16': False} for CPU")
            hidden = self.encoder(
                **{
                    k: features[k]
                    for k in (
                        "prefix_ids",
                        "prefix_mask",
                        "doc_ids",
                        "doc_mask",
                        "owners",
                    )
                }
            )
            features["sentence_embedding"] = self.pool(hidden, features["doc_mask"])
        return features

    def get_sentence_embedding_dimension(self):
        return self.hidden_size

    def get_config_dict(self):
        return self.settings

    def save(self, output_path, *args, **kwargs):
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_config(str(path))
        (path / "backbone_config.json").write_text(json.dumps(self.backbone_config, indent=2))
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
        overrides = kwargs.get("model_kwargs") or {}
        for key in (
            "query_length",
            "query_truncation",
            "document_length",
            "attention_backend",
            "gradient_checkpointing",
            "frozen_linear_bf16",
            "fused_rotary",
        ):
            if key in overrides:
                settings[key] = overrides[key]
        config_path = cls.load_file_path(model_name_or_path, "backbone_config.json", **hub)
        if config_path is None:
            raise FileNotFoundError("Checkpoint is missing backbone_config.json")
        config = json.loads(Path(config_path).read_text())
        config = AutoConfig.for_model(config.pop("model_type"), **config)
        backbone = AutoModel.from_config(config, attn_implementation="sdpa", dtype=torch.float32)
        tokenizer_hub: dict[str, Any] = {**hub, "subfolder": f"{subfolder}/tokenizer".lstrip("/")}
        tokenizer_path = cls.load_dir_path(model_name_or_path, **tokenizer_hub)
        if tokenizer_path is None:
            raise FileNotFoundError("Checkpoint is missing its tokenizer directory")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = cls(backbone=backbone, tokenizer=tokenizer, **settings)
        weights = cls.load_file_path(model_name_or_path, "model.safetensors", **hub)
        if weights is None:
            raise FileNotFoundError("Checkpoint is missing model.safetensors")
        model.load_state_dict(load_file(weights), strict=True)
        return model


class DecisionHeads(Module):
    """Independent FP32 scalar heads over candidate representations."""

    config_file_name = "heads.json"

    def __init__(self, hidden_size, tasks=("reranker",)):
        super().__init__()
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or any(t not in {"reranker", "choice", "noul", "score"} for t in tasks)
        ):
            raise ValueError("Expected distinct supported tasks")
        self.hidden_size, self.tasks = hidden_size, list(tasks)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_size, 1) for t in tasks})

    def forward(self, features, **kwargs):
        hidden = features["sentence_embedding"]
        # CPU index lists are prepared by the collator, avoiding CUDA nonzero/sync.
        indices = features.get("head_indices")
        if indices is None:
            if len(self.tasks) != 1:
                raise ValueError("Multi-task models require head_indices; use prepare_batch")
            indices = {self.tasks[0]: None}
        with torch.autocast(hidden.device.type, enabled=False):
            if len(indices) == 1 and next(iter(indices.values())) is None:
                scores = self.heads[next(iter(indices))](hidden.float())
            else:
                scores = hidden.new_zeros((hidden.shape[0], 1), dtype=torch.float32)
                for task, index in indices.items():
                    index = torch.as_tensor(index, device=hidden.device, dtype=torch.long)
                    value = self.heads[task](hidden.index_select(0, index).float())
                    scores = scores.index_copy(0, index, value)
        features["scores"] = scores
        features["sentence_embedding"] = scores
        return features

    def get_sentence_embedding_dimension(self):
        return 1

    def get_config_dict(self):
        return dict(hidden_size=self.hidden_size, tasks=self.tasks)

    def save(self, output_path, *args, **kwargs):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=True)

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
        model = cls(**cls.load_config(model_name_or_path, **hub))
        model.load_torch_weights(model_name_or_path, model=model, **hub)
        return model
