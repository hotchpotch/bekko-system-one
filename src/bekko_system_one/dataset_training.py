"""Project structured rows into the existing training renderer's small internal view."""

from __future__ import annotations

import json
from typing import Any

from .dataset_schema import VERSION, to_legacy, validate_row


def _text(value: Any) -> str:
    """Keep strings verbatim and render structured JSON values deterministically."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def training_view(row: dict[str, Any]) -> dict[str, Any]:
    """Return the renderer-compatible projection, without source or teacher metadata.

    Converted rows can use the reversible legacy bridge. Native structured rows do not need
    legacy auxiliary data: this projection reads only inference input and targets.
    """
    if "schema_version" in row and row["schema_version"] != VERSION:
        raise ValueError(f"expected schema_version={VERSION}")
    # Native producers may serialize absent bridge metadata as an empty JSON object.
    # The string "{}" is truthy, but it contains no reversible legacy representation.
    if row.get("legacy_aux_json") and json.loads(row["legacy_aux_json"]):
        return to_legacy(row)
    validate_row(row)
    targets = {target["decision_id"]: target for target in row["targets"]}
    decisions, prompts = [], []
    for source in row["input"]["decisions"]:
        did = source["id"]
        ranking = source["kind"] == "ranking"
        candidates = source["documents"] if ranking else source["criteria"]
        options = []
        for candidate in candidates:
            description_key = "content_json" if ranking else "description_json"
            option = {
                "id": candidate["id"],
                "description": _text(json.loads(candidate[description_key])),
                "value": candidate.get("value"),
                "metadata_json": "{}",
            }
            options.append(option)
        target = targets.get(did)
        old_target = None
        if target is not None:
            old_target = {
                "kind": target["annotation_kind"],
                "option_ids": list(target["ids"]),
                "probabilities": list(target["probabilities"]),
                "metadata_json": target["metadata_json"],
            }
        instruction = _text(json.loads(source["instructions_json"]))
        decisions.append(
            {
                "decision_id": did,
                "type": "score" if ranking else source["type"],
                "instructions": instruction,
                "options": options,
                "target": old_target,
                "metadata_json": "{}",
            }
        )
        prompt = {
            "decision_id": did,
            "system_prompt": source["system_prompt"] or "",
            "instruction": instruction,
        }
        if ranking:
            prompt["input_format"] = "reranking"
        prompts.append(prompt)
    return {
        "schema_version": VERSION,
        "case_id": row["case_id"],
        "group_id": row["group_id"],
        "input_hash": row["input_hash"],
        "split": row["split"],
        "language": row["language"],
        "state_json": row["input"]["state_json"],
        "decisions": decisions,
        "decision_prompts": prompts,
    }
