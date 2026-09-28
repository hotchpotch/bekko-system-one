"""Resolve typed Hugging Face configurations through the standard datasets API."""

from __future__ import annotations

from pathlib import Path

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi


def validate_hub_reference(reference):
    allowed = {"repo_id", "revision", "configs", "cache_dir", "token"}
    if set(reference) - allowed:
        raise ValueError(f"Unknown Hub source options: {sorted(set(reference) - allowed)}")
    for key in ("repo_id", "revision", "cache_dir"):
        if key in reference and (not isinstance(reference[key], str) or not reference[key].strip()):
            raise ValueError(f"Hub {key} must be a nonempty string")
    if "repo_id" not in reference:
        raise ValueError("Hub source requires repo_id")
    if "token" in reference and not isinstance(reference["token"], bool):
        raise ValueError(
            "Hub token must be a boolean; use cached login or HF_TOKEN, not a secret in config"
        )
    if "configs" in reference:
        configs = reference["configs"]
        if (
            not isinstance(configs, list)
            or not configs
            or any(not isinstance(name, str) or not name.strip() for name in configs)
        ):
            raise ValueError("Hub configs must be a nonempty list of configuration names")
        if len(configs) != len(set(configs)):
            raise ValueError("Duplicate Hub configuration name")


def pin_hub_reference(reference, cache=None):
    """Resolve a branch/tag once per training run; never write authentication secrets."""
    if not isinstance(reference, dict):
        return reference
    validate_hub_reference(reference)
    cache = {} if cache is None else cache
    key = (reference["repo_id"], reference.get("revision"))
    if key not in cache:
        info = HfApi(token=reference.get("token")).dataset_info(
            reference["repo_id"], revision=reference.get("revision")
        )
        cache[key] = info.sha
    if not cache[key]:
        raise ValueError("Hub did not return a dataset commit")
    return {**reference, "revision": cache[key]}


class HubRelease:
    """A pinned config/split inventory; datasets are fetched only when selected."""

    def __init__(self, reference):
        self.reference = pin_hub_reference(reference)
        self.repo_id = self.reference["repo_id"]
        self.kwargs = {
            k: self.reference[k] for k in ("revision", "token", "cache_dir") if k in self.reference
        }
        available = get_dataset_config_names(self.repo_id, **self.kwargs)
        selected = self.reference.get("configs", available)
        missing = set(selected) - set(available)
        if missing:
            raise ValueError(f"Unknown Hub configurations: {sorted(missing)}")
        self.manifest = {"train": [], "evaluation": [], "unlabeled": []}
        for name in sorted(selected):
            for split in sorted(get_dataset_split_names(self.repo_id, name, **self.kwargs)):
                role = "train" if split == "train" or split.startswith("train_") else "evaluation"
                if "unlabeled" in split:
                    role = "unlabeled"
                self.manifest[role].append({"dataset": name, "split": split})
        self.audit = dict(
            backend="huggingface",
            repo_id=self.repo_id,
            revision=self.reference["revision"],
            configs=sorted(selected),
        )

    def load(self, name, split):
        return load_dataset(self.repo_id, name, split=split, **self.kwargs)


def pin_model_reference(model_config):
    """Resolve remote backbone main/tag once so config, tokenizer and weights agree."""
    result = dict(model_config)
    name = result.get("model_name_or_path")
    if name is not None and not Path(name).is_dir():
        info = HfApi().model_info(name, revision=result.get("revision"))
        if not info.sha:
            raise ValueError("Hub did not return a model commit")
        result["revision"] = info.sha
    return result
