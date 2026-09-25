"""Adaptive token budgets with complete-batch OOM replay."""

import gc

import torch

from .data import pack_groups, work_tokens


class TokenBudget:
    def __init__(self, initial, *, maximum, target_bytes):
        if not 0 < initial <= maximum or target_bytes <= 0:
            raise ValueError("Invalid adaptive token budget")
        self.limit, self.maximum, self.target_bytes = initial, maximum, target_bytes
        self.bytes_per_token = 0.0

    def observe(self, *, tokens, peak_bytes, base_bytes):
        # Retain the highest observed activation cost; allocator/workspace variability
        # gets another 20% reserve. Limit growth to 2x per successful logical batch.
        self.bytes_per_token = max(
            self.bytes_per_token, max(1, peak_bytes - base_bytes) / max(1, tokens)
        )
        estimate = int(max(1, self.target_bytes - base_bytes) / (self.bytes_per_token * 1.2))
        self.limit = max(1, min(self.maximum, self.limit * 2, estimate))

    def backoff(self):
        self.limit = max(1, self.limit // 2)
        self.maximum = min(self.maximum, self.limit)  # Do not repeatedly rediscover an OOM.


def cuda_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def adaptive_backward(
    items,
    controller,
    optimizer,
    attempt,
    *,
    padded_documents=False,
    cleanup=cuda_cleanup,
    max_retries=10,
):
    """Retry the entire logical batch, discarding even partially accumulated gradients."""
    cpu_rng = torch.get_rng_state()
    uses_cuda = any(p.is_cuda for group in optimizer.param_groups for p in group["params"])
    cuda_rng = torch.cuda.get_rng_state_all() if uses_cuda else None
    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        groups = pack_groups(items, controller.limit, padded_documents)
        failed = False
        try:
            value = attempt(groups)
        except torch.cuda.OutOfMemoryError:
            failed = True
        if not failed:
            return dict(
                loss=value,
                microbatches=len(groups),
                oom_retries=retries,
                token_budget=controller.limit,
                max_work_tokens=max(work_tokens(g, padded_documents) for g in groups),
                microbatch_sizes=[len(g) for g in groups],
            )
        # Clear traceback and all gradients before allocator cleanup / replay.
        optimizer.zero_grad(set_to_none=True)
        cleanup()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        retries += 1
        if retries > max_retries or controller.limit == 1:
            raise torch.cuda.OutOfMemoryError("A single complete decision cannot fit on this GPU")
        controller.backoff()
