"""Load namespaced supplemental typed releases without flattening their inputs."""

import re

from .hub_release import pin_hub_reference
from .release import load_release


def load_additional_releases(
    roots,
    *,
    existing_names,
    prefix_layout,
    validation_samples,
    validation_sampling,
    validation_seed,
    hub_pins,
):
    """Retain native rendering, explicit splits and audit information per release."""
    if not isinstance(roots, dict):
        raise ValueError("additional_roots must be a namespace-to-release mapping")
    for namespace in roots:
        if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", namespace):
            raise ValueError(
                "additional_roots namespaces require letters, digits, underscores or hyphens"
            )
    sources, validation, audits, resolved = {}, {}, {"train": {}, "validation": {}}, {}
    for namespace, reference in sorted(roots.items()):
        resolved[namespace] = pin_hub_reference(reference, hub_pins)
        train, train_audit = load_release(resolved[namespace], "train", prefix_layout=prefix_layout)
        if not train:
            raise ValueError(f"No training sources in additional release: {namespace}")
        valid, valid_audit = load_release(
            resolved[namespace],
            "validation",
            prefix_layout=prefix_layout,
            sample_cases=validation_samples,
            sampling=validation_sampling,
            seed=validation_seed,
        )
        for target, incoming in [(sources, train), (validation, valid)]:
            for name, source in incoming.items():
                key = f"{namespace}/{name}"
                if key in existing_names:
                    raise ValueError(f"Additional release source name collision: {key}")
                target[key] = source
        audits["train"][namespace] = train_audit
        audits["validation"][namespace] = valid_audit
    return sources, validation, audits, resolved
