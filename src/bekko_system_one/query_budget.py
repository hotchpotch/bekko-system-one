"""Structured query rendering and order-independent token budgets."""

from dataclasses import dataclass


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
