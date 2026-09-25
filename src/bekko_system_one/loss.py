"""FP32 groupwise cross entropy for hard or soft targets."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def distribution_loss_sum(scores, targets):
    """Targets must have been validated by Group before reaching the GPU."""
    logits = scores.flatten().split([len(t) for t in targets])
    padded = pad_sequence(logits, batch_first=True, padding_value=-1e9).float()
    target = torch.tensor(
        [list(t) + [0.0] * (padded.shape[1] - len(t)) for t in targets],
        device=scores.device,
        dtype=torch.float32,
    )
    return -(target * padded.log_softmax(-1)).sum()


class GroupedDistributionLoss(nn.Module):
    """A Sentence Transformers loss operating on collated complete groups.

    sentence_features contains one feature dictionary; labels contains the
    aligned target distributions. Use a group-aware collator with ST Trainer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        return distribution_loss_sum(self.model(sentence_features[0])["scores"], labels) / len(
            labels
        )
