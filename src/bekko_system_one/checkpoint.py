"""Checkpoint transformations that preserve the initialized scoring function."""

import copy

from peft import PeftModel


def merge_lora_for_full_training(model):
    """Return an independent FP32 model with merged LoRA and all parameters trainable.

    Load frozen-linear checkpoints with frozen_linear_bf16=False when preparing
    on CPU. Save the returned model normally to persist the non-LoRA settings.
    """
    if not isinstance(model[0].encoder.backbone, PeftModel):
        raise ValueError("Expected a LoRA checkpoint")
    merged = copy.deepcopy(model).float()
    encoder = merged[0]
    encoder.encoder.backbone = encoder.encoder.backbone.merge_and_unload(safe_merge=True)
    encoder.settings["lora"] = None
    encoder.settings["frozen_linear_bf16"] = False
    merged.requires_grad_(True)
    return merged
