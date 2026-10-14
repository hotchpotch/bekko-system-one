"""Bidirectional cross-encoder and decision-local listwise candidate encoders."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .modules import SharedPrefix


def fair_lengths(lengths, budget):
    """Max-min fair allocation; short candidates donate unused capacity to long ones."""
    if budget < 0:
        raise ValueError("Negative candidate token budget")
    result = [0] * len(lengths)
    remaining = list(range(len(lengths)))
    while remaining and budget:
        share = max(1, budget // len(remaining))
        for i in remaining:
            take = min(lengths[i] - result[i], share, budget)
            result[i] += take
            budget -= take
        remaining = [i for i in remaining if result[i] < lengths[i]]
    return result


class JointEncoder(SharedPrefix):
    """Use full bidirectional sequence attention (ModernBERT local layers retained).

    Cross encoders pool CLS or the complete query/candidate sequence. Listwise
    encoders pool each candidate's content, or a dedicated marker before/after
    it. SEP is excluded from candidate means except for empty candidate text.
    Complete decisions, including duplicate queries, must stay separate.
    """

    joint_candidates = True
    config_file_name = "joint_encoder.json"

    def __init__(
        self,
        model_name_or_path=None,
        *,
        architecture="listwise",
        pooling="mean",
        max_length=None,
        candidate_token=None,
        **kwargs,
    ):
        if architecture not in {"cross_encoder", "listwise"}:
            raise ValueError("JointEncoder architecture must be cross_encoder or listwise")
        supported = (
            {"mean", "cls"}
            if architecture == "cross_encoder"
            else {"mean", "token_before", "token_after"}
        )
        if pooling not in supported:
            raise ValueError(f"{architecture} pooling must be one of {sorted(supported)}")
        if candidate_token is not None and (
            not isinstance(candidate_token, str) or not candidate_token.strip()
        ):
            raise ValueError("candidate_token must be a nonempty string")
        if candidate_token is not None and not pooling.startswith("token_"):
            raise ValueError("candidate_token requires token_before or token_after pooling")
        super().__init__(model_name_or_path, **kwargs)
        self.architecture, self.pooling = architecture, pooling
        position_limit = self.backbone_config["max_position_embeddings"]
        self.max_length = position_limit if max_length is None else max_length
        if not isinstance(self.max_length, int) or not 4 <= self.max_length <= position_limit:
            raise ValueError("max_length must be between 4 and backbone max_position_embeddings")
        self.candidate_token_id = None
        if pooling.startswith("token_"):
            token = candidate_token or self.tokenizer.mask_token
            if token is None:
                raise ValueError("Tokenizer has no MASK; supply a candidate_token")
            if self.settings["lora"]:
                raise ValueError("Special-token pooling currently requires lora: null")
            vocab = self.tokenizer.get_vocab()
            if token in vocab and token not in self.tokenizer.all_special_tokens:
                raise ValueError("candidate_token must be a special token")
            if token in {self.tokenizer.pad_token, self.tokenizer.unk_token}:
                raise ValueError("candidate_token cannot be PAD or UNK")
            self.tokenizer.add_special_tokens(
                {"extra_special_tokens": [token]}, replace_extra_special_tokens=False
            )
            backbone = self.encoder.backbone
            with torch.random.fork_rng(devices=[]):
                if len(self.tokenizer) > backbone.get_input_embeddings().num_embeddings:
                    backbone.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            self.candidate_token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token not in vocab:
                with torch.no_grad():
                    weight = backbone.get_input_embeddings().weight
                    weight[self.candidate_token_id].copy_(weight[self.tokenizer.sep_token_id])
            self.backbone_config["vocab_size"] = backbone.config.vocab_size
            candidate_token = token
        self.settings.update(
            architecture=architecture,
            pooling=pooling,
            max_length=self.max_length,
            candidate_token=candidate_token,
        )

    def _sequence(self, query, documents):
        marker = self.candidate_token_id
        content = [d[:-1] for d in documents]  # retain task markers, remove final SEP
        overhead = len(documents) * (1 + (marker is not None))
        # Preserve one text token per nonempty candidate, plus query CLS/SEP.
        minimum = sum(bool(d) for d in content)
        query_budget = min(len(query), self.max_length - overhead - minimum)
        if query_budget < 2:
            raise ValueError("Candidate count exceeds max_length even at minimum token lengths")
        if len(query) > query_budget:
            # Preserve both ends of the already rendered/balanced prefix.
            left = (query_budget + 1) // 2
            query = query[:left] + query[-(query_budget - left) :]
        lengths = fair_lengths([len(d) for d in content], self.max_length - len(query) - overhead)
        sequence, spans = list(query), []
        for document, length in zip(content, lengths, strict=True):
            if marker is not None and self.pooling == "token_before":
                spans.append([len(sequence)])
                sequence.append(marker)
            start = len(sequence)
            sequence.extend(document[:length])
            if self.pooling == "token_after":
                spans.append([len(sequence)])
                sequence.append(marker)
            elif self.pooling == "mean":
                spans.append(list(range(start, len(sequence))) or [len(sequence)])
            sequence.append(self.tokenizer.sep_token_id)
        if self.architecture == "cross_encoder":
            spans = [[0]] if self.pooling == "cls" else [list(range(len(sequence)))]
        return sequence, spans

    def collate_tokens(self, queries, documents, owners):
        rows, spans, offset = [], [[] for _ in documents], 0
        groups = (
            [[i] for i in range(len(documents))]
            if self.architecture == "cross_encoder"
            else [[i for i, owner in enumerate(owners) if owner == q] for q in range(len(queries))]
        )
        for indices in groups:
            if not indices:
                continue
            sequence, local_spans = self._sequence(
                queries[owners[indices[0]]], [documents[i] for i in indices]
            )
            rows.append(torch.tensor(sequence, dtype=torch.long))
            for index, span in zip(indices, local_spans, strict=True):
                spans[index] = [offset + j for j in span]
            offset += len(sequence)
        ids = pad_sequence(rows, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = torch.arange(ids.shape[1])[None] < torch.tensor([len(row) for row in rows])[:, None]
        indices = [index for span in spans for index in span]
        segment_ids = [i for i, span in enumerate(spans) for _ in span]
        return dict(
            input_ids=ids,
            attention_mask=mask,
            pool_indices=torch.tensor(indices, dtype=torch.long),
            pool_segments=torch.tensor(segment_ids, dtype=torch.long),
            pool_lengths=torch.tensor([len(span) for span in spans], dtype=torch.long),
        )

    def forward(self, features, **kwargs):
        device = next(self.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = self.encoder.encode_bidirectional(
                features["input_ids"], features["attention_mask"]
            )
            selected = hidden.index_select(0, features["pool_indices"])
            pooled = hidden.new_zeros((len(features["pool_lengths"]), hidden.shape[-1]))
            pooled = pooled.index_add(0, features["pool_segments"], selected)
            features["sentence_embedding"] = pooled / features["pool_lengths"][:, None]
        return features
