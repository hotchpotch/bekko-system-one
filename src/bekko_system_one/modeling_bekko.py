"""Standalone Transformers AutoModel interface for exported typed decisions."""

import json
from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .inference_v0 import BekkoInference


class BekkoConfig(PretrainedConfig):
    """Locate the portable inference module within a model repository."""

    model_type = "bekko_system_one"

    def __init__(self, runtime_subfolder="0_BekkoInference", **kwargs):
        super().__init__(**kwargs)
        if Path(runtime_subfolder).is_absolute() or ".." in Path(runtime_subfolder).parts:
            raise ValueError("runtime_subfolder must stay within the model repository")
        self.runtime_subfolder = runtime_subfolder


class BekkoModel(PreTrainedModel):
    """AutoModel wrapper exposing native typed predict(), without the trainer."""

    config_class = BekkoConfig
    _supports_sdpa = True

    def __init__(self, config, runtime):
        super().__init__(config)
        self.runtime = runtime

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, config=None,
        revision=None, token=None, cache_dir=None, local_files_only=False,
        subfolder="", device=None, device_map=None, dtype=None,
        attn_implementation="auto", **kwargs,
    ):
        """Load local or Hub weights; supports a single CPU or CUDA device."""
        for key in ("trust_remote_code", "_from_auto", "_commit_hash", "code_revision", "adapter_kwargs"):
            kwargs.pop(key, None)
        legacy_dtype = kwargs.pop("torch_dtype", None)
        if dtype is None:
            dtype = legacy_dtype
        if model_args or kwargs:
            raise TypeError(f"Unsupported loading arguments: {sorted(kwargs)}")
        if device is not None and device_map is not None:
            raise ValueError("Specify device or device_map, not both")
        if device_map is not None:
            if isinstance(device_map, dict) and set(device_map) == {""}:
                device = device_map[""]
            elif isinstance(device_map, (str, torch.device)) and str(device_map) != "auto":
                device = device_map
            else:
                raise ValueError("Use device or a single-device map; automatic sharding is unsupported")
        hub = dict(revision=revision, token=token, cache_dir=cache_dir,
                   local_files_only=local_files_only, subfolder=subfolder)
        if config is None:
            config = BekkoConfig.from_pretrained(pretrained_model_name_or_path, **hub)
        # AutoModel may consume this argument while resolving the config. The
        # portable runtime, rather than Transformers, selects its attention.
        configured_attention = getattr(config, "_attn_implementation_internal", None)
        if attn_implementation == "auto" and configured_attention is not None:
            attn_implementation = configured_attention
        config._attn_implementation = "eager"
        revision = getattr(config, "_commit_hash", None) or revision
        runtime = BekkoInference.load(
            pretrained_model_name_or_path,
            subfolder="/".join(filter(None, (subfolder, config.runtime_subfolder))),
            revision=revision, token=token, cache_folder=cache_dir,
            local_files_only=local_files_only,
        )
        model = cls(config, runtime).eval().requires_grad_(False)
        if dtype not in (None, "auto"):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            model.to(dtype=dtype)
        model.to(device=device or "cpu")
        runtime.set_attention_implementation(attn_implementation, device=model.device)
        return model

    def predict(self, inputs, **kwargs):
        """Return typed Noul, Choice, Score or ranking results."""
        return self.runtime.predict(inputs, **kwargs)

    def forward(self, features):
        """Run already-tokenized portable runtime features."""
        return self.runtime(features)

    def compile_inference(self, **kwargs):
        self.runtime.compile_inference(**kwargs)
        return self

    def save_pretrained(self, save_directory, **kwargs):
        """Save an independently reloadable AutoModel and ST-compatible runtime."""
        if kwargs:
            raise TypeError(f"Unsupported saving arguments: {sorted(kwargs)}")
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.runtime.save(path / self.config.runtime_subfolder)
        write_automodel_files(path, self.config)


def write_automodel_files(path, config=None):
    """Add AutoClass metadata/code to an existing standalone ST export."""
    path = Path(path)
    config = config or BekkoConfig()
    config.auto_map = {
        "AutoConfig": "modeling_bekko.BekkoConfig",
        "AutoModel": "modeling_bekko.BekkoModel",
    }
    config.architectures = ["BekkoModel"]
    config.save_pretrained(path)
    (path / "modeling_bekko.py").write_text(Path(__file__).read_text())
    (path / "inference_v0.py").write_text(Path(__file__).with_name("inference_v0.py").read_text())
    (path / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": config.runtime_subfolder,
         "type": "inference_v0.BekkoInference"}
    ], indent=2) + "\n")
