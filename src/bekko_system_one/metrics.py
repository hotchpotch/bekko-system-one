"""Distribution and typed numerical metrics with explicit eligible counts."""

from collections import Counter, defaultdict

import torch

METRICS = (
    "cross_entropy",
    "accuracy",
    "brier",
    "target_mass_at_prediction",
    "kl_divergence",
    "score_mae",
    "score_normalized_mae",
    "binary_brier",
)


class DecisionMetrics:
    def __init__(self):
        self.groups = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.counts = Counter()

    def update(self, group, logits):
        scores = logits.detach().float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite evaluation scores")
        if group.target is None:
            raise ValueError("Evaluation requires targets")
        target = scores.new_tensor(group.target)
        logp = scores.log_softmax(0)
        p = logp.exp()
        ce = -(target * logp).sum()
        positive = target > 0
        kl = (target[positive] * (target[positive].log() - logp[positive])).sum()
        values = dict(
            cross_entropy=ce.item(),
            accuracy=float(target[scores.argmax()] == target.max()),
            brier=(p - target).square().sum().item(),
            target_mass_at_prediction=target[scores.argmax()].item(),
            kl_divergence=max(0.0, kl.item()),
        )
        metadata = group.metadata
        if metadata is not None and metadata.kind != "ranking":
            if group.task == "score" and metadata.score_values is not None:
                scale = scores.new_tensor(metadata.score_values)
                error = ((p - target) * scale).sum().abs().item()
                values["score_mae"] = error
                values["score_normalized_mae"] = error / (scale.max() - scale.min()).item()
            if group.task == "noul" and metadata.yes_index is not None:
                i = metadata.yes_index
                values["binary_brier"] = (p[i] - target[i]).square().item()
        self.groups += 1
        for key, value in values.items():
            self.sums[key] += value
            self.counts[key] += 1

    def result(self):
        return dict(
            groups=self.groups,
            **{k: self.sums[k] / self.counts[k] if self.counts[k] else None for k in METRICS},
            metric_counts={k: self.counts[k] for k in METRICS},
        )
