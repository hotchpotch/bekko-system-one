"""Decision Index engine using the pinned standalone public Bekko v0 release.

Requires the official apolinario/decision-index kit on PYTHONPATH or installed.
Model downloads are public; rebuilding the full suite requires HLE access.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from decision_index.engines import Engine, Unsupported
from huggingface_hub import snapshot_download

MODEL = "hotchpotch/bekko-system-one-v0-17m"
REVISION = "b886a1f9b91f4e8368d7830080d52d955e2c8dfa"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class BekkoEngine(Engine):
    name = "bekko-v0"

    def __init__(
        self, model=MODEL, revision=REVISION, device="cpu", allow_truncation=False, **options
    ):
        if not isinstance(allow_truncation, bool):
            raise ValueError("allow_truncation must be a boolean")
        self.allow_truncation = allow_truncation
        super().__init__(
            model=model,
            revision=revision,
            device=device,
            allow_truncation=allow_truncation,
            **options,
        )
        path = Path(snapshot_download(model, revision=revision))
        source = path / "inference_v0.py"
        module_name = "bekko_decision_index_release"
        spec = importlib.util.spec_from_file_location(module_name, source)
        self.runtime_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.runtime_module
        spec.loader.exec_module(self.runtime_module)
        self.model = self.runtime_module.BekkoSentenceTransformer(
            str(path), device=device, trust_remote_code=True, attn_implementation="sdpa"
        )
        self.provenance = {
            "repo": model,
            "revision": revision,
            "runtime_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy": "Native rendering and heads; permit native truncation (not LB compliant)."
            if allow_truncation
            else "Native rendering and heads; refuse any request that would truncate.",
            "allow_truncation": allow_truncation,
            "implicit_noul_criteria": {
                "true": "The statement in the instructions is true.",
                "false": "The statement in the instructions is false.",
            },
        }

    def __call__(self, state, questions):
        decisions = []
        for key, question in questions.items():
            task = question["type"]
            if task not in {"choice", "noul"}:
                raise Unsupported(f"Unsupported question type {task}")
            criteria = question["criteria"]
            if task == "noul" and not criteria:
                criteria = {
                    "true": "The statement in the instructions is true.",
                    "false": "The statement in the instructions is false.",
                }
            if task == "noul" and set(criteria) not in ({"yes", "no"}, {"true", "false"}):
                raise Unsupported("Noul requires explicit yes/no or true/false criteria")
            decisions.append(
                {
                    "id": key,
                    "kind": "judgment",
                    "type": task,
                    "instructions_json": encoded(question["instructions"]),
                    "criteria": [
                        {"id": cid, "description_json": encoded(description), "value": None}
                        for cid, description in criteria.items()
                    ],
                    "documents": [],
                    "scoring": None,
                    "system_prompt": "",
                }
            )
        request = {"state_json": encoded(state), "decisions": decisions}
        encoder = self.model[0]
        groups = self.runtime_module.input_groups(request, prefix_layout=encoder.prefix_layout)
        prepared = encoder.prepare_groups(groups)
        # Ask the release tokenizer for uncapped branches and compare token IDs,
        # including independent balanced-query fields and task-marker overhead.
        # A UTF-8 byte count bounds byte-level BPE token counts conservatively.
        limit = 100 + sum(
            len(g.query.encode("utf-8")) + sum(len(c.encode("utf-8")) for c in g.candidates)
            for g in groups
        )
        truncated = []
        for group, item in zip(groups, prepared, strict=True):
            full_query = encoder._queries([group.query], [group.query_parts], [limit])[0]
            full_documents = encoder._documents(
                group.candidates, [group.task] * len(group.candidates), limit
            )
            if item.query != full_query or item.documents != full_documents:
                if not self.allow_truncation:
                    raise Unsupported("Native v0 branch/context limits would truncate this request")
                truncated.append(group.metadata.decision_id)
        predictions = self.model.predict(request, show_progress_bar=False)
        answers = {}
        for key, question in questions.items():
            prediction = predictions[key]
            if question["type"] == "choice":
                answers[key] = {
                    "type": "choice",
                    "choice": prediction["selected_id"],
                    "probabilities": prediction["probabilities"],
                }
            else:
                answers[key] = {"type": "noul", "noul": prediction["probability_yes"]}
        raw = {"truncated_decision_ids": truncated} if self.allow_truncation else None
        return {"model": self.options["model"], "answers": answers}, raw

    def synchronize(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)

    def runtime(self):
        encoder = self.model[0]
        return {
            "torch": torch.__version__,
            "device": str(self.model.device),
            "gpu": torch.cuda.get_device_name(self.model.device)
            if self.model.device.type == "cuda"
            else None,
            "context_length": encoder.context_length,
            "query_length": encoder.query_length,
            "document_length": encoder.document_length,
            "attn_implementation": encoder.attn_implementation,
        }
