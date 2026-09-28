"""Model-independent structured case format and lossless adapters for the v1 release rows.

Judgments and document ranking share a case/input envelope, but keep different
target semantics. In particular, a ranking distribution is relative supervision.
"""

from __future__ import annotations

import json
import math
from typing import Any

from datasets import Features, List, Value

VERSION = "system_one.v1"


def dataset_features() -> Features:
    """Arrow features for structured cases. JSON payload strings preserve arbitrary state values."""
    return Features(
        {
            "schema_version": Value("string"),
            "case_id": Value("string"),
            "group_id": Value("string"),
            "input_hash": Value("string"),
            "split": Value("string"),
            "language": Value("string"),
            "input": {
                "state_json": Value("large_string"),
                "decisions": List(
                    {
                        "id": Value("string"),
                        "kind": Value("string"),
                        "type": Value("string"),
                        "instructions_json": Value("large_string"),
                        "system_prompt": Value("large_string"),
                        "criteria": List(
                            {
                                "id": Value("string"),
                                "description_json": Value("large_string"),
                                "value": Value("float64"),
                            }
                        ),
                        "documents": List(
                            {"id": Value("string"), "content_json": Value("large_string")}
                        ),
                        "scoring": Value("string"),
                    }
                ),
            },
            "targets": List(
                {
                    "decision_id": Value("string"),
                    "kind": Value("string"),
                    "annotation_kind": Value("string"),
                    "ids": List(Value("string")),
                    "probabilities": List(Value("float64")),
                    "metadata_json": Value("large_string"),
                }
            ),
            "provenance_json": Value("large_string"),
            "legacy_aux_json": Value("large_string"),
        }
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str) -> Any:
    return json.loads(value)


def _parse_json_string(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return fallback
    return fallback


def _validate_distribution(ids: list[str], probabilities: list[float]) -> None:
    if len(ids) != len(set(ids)) or len(ids) != len(probabilities) or not ids:
        raise ValueError("target IDs must be unique and align with probabilities")
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("target probabilities must be finite and nonnegative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError("target probabilities must sum to one")


def validate_row(row: dict[str, Any]) -> None:
    """Validate structured structure, decision/target alignment, and target normalization."""
    if "schema_version" in row and row["schema_version"] != VERSION:
        raise ValueError(f"expected schema_version={VERSION}")
    inp = row["input"]
    decisions = inp["decisions"]
    if not isinstance(_loads(inp["state_json"]), (dict, list, str, int, float, bool, type(None))):
        raise ValueError("invalid state_json")
    dids = [d["id"] for d in decisions]
    if len(dids) != len(set(dids)):
        raise ValueError("decision IDs must be unique")
    by_id = {d["id"]: d for d in decisions}
    seen_targets = set()
    for target in row["targets"]:
        did = target["decision_id"]
        if did not in by_id or did in seen_targets:
            raise ValueError("target decision_id must name exactly one input decision")
        seen_targets.add(did)
        if target["kind"] not in {"judgment_distribution", "ranking_distribution"}:
            raise ValueError("unknown target kind")
        decision = by_id[did]
        expected_kind = (
            "ranking_distribution" if decision["kind"] == "ranking" else "judgment_distribution"
        )
        if target["kind"] != expected_kind:
            raise ValueError("target kind does not match decision kind")
        ids = (
            [c["id"] for c in decision["documents"]]
            if decision["kind"] == "ranking"
            else [c["id"] for c in decision["criteria"]]
        )
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        if set(target["ids"]) != set(ids):
            raise ValueError("target IDs must match input candidate IDs")
        _validate_distribution(target["ids"], target["probabilities"])
    for decision in decisions:
        candidates = (
            decision["documents"] if decision["kind"] == "ranking" else decision["criteria"]
        )
        candidate_ids = [c["id"] for c in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        if decision["kind"] not in {"judgment", "ranking"}:
            raise ValueError("decision kind must be judgment or ranking")
        if decision["kind"] == "ranking":
            if decision["type"] is not None or decision["scoring"] != "relative":
                raise ValueError("ranking requires null type and relative scoring")
            if decision["criteria"] or len(decision["documents"]) < 1:
                raise ValueError("ranking requires documents and no criteria")
        else:
            if decision["type"] not in {"noul", "choice", "score"} or decision["documents"]:
                raise ValueError("judgment requires a supported type and no documents")
            if len(decision["criteria"]) < 1:
                raise ValueError("judgment requires criteria")
            if decision["type"] == "score":
                values = [c["value"] for c in decision["criteria"]]
                if any(v is None or not math.isfinite(v) for v in values) or len(values) != len(
                    set(values)
                ):
                    raise ValueError("Score criteria require finite, unique numeric values")


def from_legacy(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat v1 case row; ``legacy_aux_json`` makes conversion reversible."""
    prompts = row.get("decision_prompts") or []
    decisions, targets = [], []
    legacy_decisions = row.get("decisions") or []
    if prompts and len(prompts) != len(legacy_decisions):
        raise ValueError("decision/prompt alignment mismatch")
    for i, old in enumerate(legacy_decisions):
        prompt = prompts[i] if prompts else {}
        if prompt and prompt.get("decision_id") != old["decision_id"]:
            raise ValueError("decision/prompt ID mismatch")
        is_ranking = prompt.get("input_format") == "reranking"
        did = old["decision_id"]
        if is_ranking:
            documents = [
                {"id": o["id"], "content_json": _json(o["description"])} for o in old["options"]
            ]
            criteria = []
            kind, typ, scoring = "ranking", None, "relative"
        else:
            documents = []
            criteria = [
                {
                    "id": o["id"],
                    "description_json": _json(o["description"]),
                    "value": None if o.get("value") is None else float(o["value"]),
                }
                for o in old["options"]
            ]
            kind, typ, scoring = "judgment", old["type"], None
            if typ == "score":
                # Expose ordinal criteria in ascending numeric order; targets remain ID-aligned.
                criteria.sort(key=lambda c: (c["value"] is None, c["value"] or 0.0))
        decisions.append(
            {
                "id": did,
                "kind": kind,
                "type": typ,
                "instructions_json": _json(
                    prompt["instruction"] if "instruction" in prompt else old.get("instructions")
                ),
                "system_prompt": prompt.get("system_prompt", "") or "",
                "criteria": criteria,
                "documents": documents,
                "scoring": scoring,
            }
        )
        target = old.get("target")
        if target is not None:
            targets.append(
                {
                    "decision_id": did,
                    "kind": "ranking_distribution" if is_ranking else "judgment_distribution",
                    "annotation_kind": target.get("kind", "unknown"),
                    "ids": list(target["option_ids"]),
                    "probabilities": [float(x) for x in target["probabilities"]],
                    "metadata_json": target.get("metadata_json"),
                }
            )
    aux = {
        "schema_version": row.get("schema_version"),
        "field_presence": {
            k: k in row for k in ("source_json", "metadata_json", "decisions", "decision_prompts")
        },
        "legacy_columns": {
            k: v
            for k, v in row.items()
            if k
            not in {
                "schema_version",
                "case_id",
                "group_id",
                "input_hash",
                "split",
                "language",
                "state_json",
                "source_json",
                "metadata_json",
                "decisions",
                "decision_prompts",
            }
        },
        "metadata_json": row.get("metadata_json"),
        "decision_details": [],
        "option_orders": {},
        "prompts_present": "decision_prompts" in row,
        "prompts_were_null": row.get("decision_prompts") is None,
        "prompts": [],
    }
    for i, old in enumerate(legacy_decisions):
        aux["option_orders"][old["decision_id"]] = [o["id"] for o in old["options"]]
        prompt = prompts[i] if prompts else {}
        effective = prompt.get("instruction", old.get("instructions"))
        aux["decision_details"].append(
            {
                "instructions": old.get("instructions")
                if old.get("instructions") != effective
                else None,
                "instructions_present": "instructions" in old,
                "instructions_match": old.get("instructions") == effective,
                "metadata_json": old.get("metadata_json"),
                "metadata_present": "metadata_json" in old,
                "extra": {
                    k: v
                    for k, v in old.items()
                    if k
                    not in {
                        "decision_id",
                        "type",
                        "instructions",
                        "options",
                        "target",
                        "metadata_json",
                    }
                },
                "target_present": "target" in old,
                "target_was_null": old.get("target") is None,
                "option_details": [
                    {
                        "value_present": "value" in o,
                        "value": o.get("value"),
                        "metadata_json": o.get("metadata_json"),
                        "metadata_present": "metadata_json" in o,
                        "extra": {
                            k: v
                            for k, v in o.items()
                            if k not in {"id", "description", "value", "metadata_json"}
                        },
                    }
                    for o in old["options"]
                ],
                "target_extra": {
                    k: v
                    for k, v in (old.get("target") or {}).items()
                    if k not in {"kind", "option_ids", "probabilities", "metadata_json"}
                },
                "target_kind_present": "kind" in (old.get("target") or {}),
                "target_metadata_present": "metadata_json" in (old.get("target") or {}),
            }
        )
        if prompts:
            p = prompts[i]
            aux["prompts"].append(
                {
                    "extra": {
                        k: v for k, v in p.items() if k not in {"instruction", "system_prompt"}
                    },
                    "instruction_present": "instruction" in p,
                    "instruction_was_null": p.get("instruction") is None,
                    "system_prompt_present": "system_prompt" in p,
                    "system_prompt_was_null": p.get("system_prompt") is None,
                }
            )
        else:
            aux["prompts"].append(
                {"extra": {}, "instruction_present": False, "system_prompt_present": False}
            )
    result = {
        "schema_version": VERSION,
        "case_id": row["case_id"],
        "group_id": row["group_id"],
        "input_hash": row["input_hash"],
        "split": row["split"],
        "language": row["language"],
        "input": {"state_json": row["state_json"], "decisions": decisions},
        "targets": targets,
        "provenance_json": row.get("source_json", "{}"),
        "legacy_aux_json": _json(aux),
    }
    validate_row(result)
    return result


def to_legacy(row: dict[str, Any]) -> dict[str, Any]:
    """Restore the exact v1 row fields and values from a row returned by ``from_legacy``."""
    validate_row(row)
    aux = _loads(row["legacy_aux_json"])
    targets = {t["decision_id"]: t for t in row["targets"]}
    decisions, prompts = [], []
    details = aux.get("decision_details", [])
    for i, d in enumerate(row["input"]["decisions"]):
        detail = details[i]
        criteria = d["criteria"]
        if d["kind"] == "ranking":
            options = []
            by_id = {c["id"]: c for c in d["documents"]}
            for j, oid in enumerate(aux["option_orders"][d["id"]]):
                c = by_id[oid]
                od = detail["option_details"][j]
                opt = {"id": c["id"], "description": _loads(c["content_json"])}
                if od["value_present"]:
                    opt["value"] = od["value"]
                if od["metadata_present"]:
                    opt["metadata_json"] = od["metadata_json"]
                opt.update(od.get("extra", {}))
                options.append(opt)
            typ = "score"
        else:
            # Restore original legacy ordering from its prompt's aligned IDs.
            originals = aux.get("option_orders", {}).get(d["id"])
            if originals is None:
                originals = [c["id"] for c in criteria]
            cmap = {c["id"]: c for c in criteria}
            options = []
            for j, oid in enumerate(originals):
                c = cmap[oid]
                opt = {"id": c["id"], "description": _loads(c["description_json"])}
                od = detail["option_details"][j]
                if od["value_present"]:
                    opt["value"] = od["value"] if c["value"] == od["value"] else c["value"]
                if od["metadata_present"]:
                    opt["metadata_json"] = od["metadata_json"]
                opt.update(od.get("extra", {}))
                options.append(opt)
            typ = d["type"]
        target_structured = targets.get(d["id"])
        target = (
            None
            if target_structured is None
            else (
                (
                    {"kind": target_structured["annotation_kind"]}
                    if detail.get("target_kind_present")
                    else {}
                )
                | {
                    "option_ids": target_structured["ids"],
                    "probabilities": target_structured["probabilities"],
                }
                | (
                    {"metadata_json": target_structured["metadata_json"]}
                    if detail.get("target_metadata_present")
                    else {}
                )
                | detail.get("target_extra", {})
            )
        )
        decision = {
            "decision_id": d["id"],
            "type": typ,
            "instructions": detail["instructions"]
            if detail["instructions"] is not None
            else _loads(d["instructions_json"]),
            "options": options,
        }
        if detail["instructions_present"]:
            decision["instructions"] = (
                _loads(d["instructions_json"])
                if detail["instructions_match"]
                else detail["instructions"]
            )
        else:
            decision.pop("instructions", None)
        if detail["target_present"]:
            decision["target"] = None if detail["target_was_null"] else target
        if detail["metadata_present"]:
            decision["metadata_json"] = detail["metadata_json"]
        decision.update(detail.get("extra", {}))
        decisions.append(decision)
        if i < len(aux.get("prompts", [])):
            pd = aux["prompts"][i]
            p = dict(pd.get("extra", {}))
            if pd.get("instruction_present"):
                p["instruction"] = (
                    None if pd.get("instruction_was_null") else _loads(d["instructions_json"])
                )
            if pd.get("system_prompt_present"):
                p["system_prompt"] = (
                    None if pd.get("system_prompt_was_null") else d["system_prompt"]
                )
            prompts.append(p)
    legacy = {
        "schema_version": aux.get("schema_version"),
        "case_id": row["case_id"],
        "group_id": row["group_id"],
        "input_hash": row["input_hash"],
        "split": row["split"],
        "language": row["language"],
        "state_json": row["input"]["state_json"],
    }
    if aux["field_presence"].get("decisions"):
        legacy["decisions"] = decisions
    if aux["field_presence"].get("source_json"):
        legacy["source_json"] = row["provenance_json"]
    if aux["field_presence"].get("metadata_json"):
        legacy["metadata_json"] = aux["metadata_json"]
    if aux["prompts_present"]:
        legacy["decision_prompts"] = None if aux["prompts_were_null"] else prompts
    legacy.update(aux.get("legacy_columns", {}))
    return legacy


def inference_input(row: dict[str, Any]) -> dict[str, Any]:
    """Return parsed model-available state/instructions/candidates, excluding all labels/source."""
    inp = row["input"]
    result = {
        "state": _loads(inp["state_json"]),
        "decisions": [],
    }
    for d in inp["decisions"]:
        item = {
            "id": d["id"],
            "kind": d["kind"],
            "type": d["type"],
            "instructions": _loads(d["instructions_json"]),
            "system_prompt": d["system_prompt"],
        }
        if d["kind"] == "ranking":
            item["documents"] = [
                {"id": c["id"], "content": _loads(c["content_json"])} for c in d["documents"]
            ]
            item["scoring"] = d["scoring"]
        else:
            item["criteria"] = [
                {"id": c["id"], "description": _loads(c["description_json"]), "value": c["value"]}
                for c in d["criteria"]
            ]
        result["decisions"].append(item)
    return result


def to_jev_request(
    row: dict[str, Any], *, ranking_question: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build Jev's ``state`` plus ID-keyed ``questions`` API request.

    Ranking is list-relative supervision and cannot be represented as a Jev typed
    judgment without an explicit pointwise question supplied by the caller.
    """
    parsed = inference_input(row)
    questions: dict[str, dict[str, Any]] = {}
    for decision in parsed["decisions"]:
        if decision["kind"] == "ranking":
            if ranking_question is None:
                raise ValueError("ranking requires an explicit pointwise ranking_question mapping")
            typ = ranking_question.get("type")
            instructions = ranking_question.get("instructions")
            criteria = ranking_question.get("criteria")
            if typ not in {"noul", "score"} or instructions is None:
                raise ValueError("ranking_question requires type (noul or score) and instructions")
            if typ == "score" and not isinstance(criteria, list):
                raise ValueError("Score ranking_question requires an ordered criteria list")
            for document in decision["documents"]:
                qid = _json([decision["id"], document["id"]])
                if qid in questions:
                    raise ValueError(f"question ID collision: {qid}")
                questions[qid] = {
                    "type": typ,
                    "instructions": {
                        "document": document["content"],
                        "instruction": instructions,
                        "ranking_instruction": decision["instructions"],
                        "ranking_system_prompt": decision["system_prompt"],
                    },
                }
                if criteria is not None:
                    questions[qid]["criteria"] = criteria
            continue
        instruction = decision["instructions"]
        system = decision["system_prompt"]
        if system:
            instruction = {"system": system, "instruction": instruction}
        criteria = decision["criteria"]
        if decision["type"] == "score":
            rendered_criteria: Any = [c["description"] for c in criteria]
        else:
            rendered_criteria = {c["id"]: c["description"] for c in criteria}
        qid = decision["id"]
        if qid in questions:
            raise ValueError(f"question ID collision: {qid}")
        questions[qid] = {
            "type": decision["type"],
            "instructions": instruction,
            "criteria": rendered_criteria,
        }
    return {"state": parsed["state"], "questions": questions}
