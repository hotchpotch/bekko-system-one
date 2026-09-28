import torch
from sentence_transformers import SentenceTransformer

from bekko_system_one import DecisionHeads, Group, SharedPrefix, prepare_batch
from bekko_system_one.checkpoint import merge_lora_for_full_training


def test_merge_nonzero_adapters_and_roundtrip(tmp_path, tiny_encoder):
    with torch.no_grad():
        for name, p in tiny_encoder.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.01)
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    ).eval()
    groups = [Group("query", ["good", "bad"], "choice")]
    with torch.no_grad():
        expected = model(prepare_batch(model, groups))["scores"]
    merged = merge_lora_for_full_training(model)
    assert isinstance(model[0], SharedPrefix)
    assert model[0].settings["lora"] is not None
    assert merged[0].settings["lora"] is None
    assert all(p.requires_grad and p.dtype == torch.float32 for p in merged.parameters())
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    with torch.no_grad():
        torch.testing.assert_close(
            merged(prepare_batch(merged, groups))["scores"], expected, rtol=1e-4, atol=1e-6
        )
    merged.save_pretrained(str(tmp_path / "model"), create_model_card=False)
    loaded = SentenceTransformer(
        str(tmp_path / "model"), device="cpu", local_files_only=True, trust_remote_code=True
    )
    assert isinstance(loaded[0], SharedPrefix)
    assert loaded[0].settings["lora"] is None
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(prepare_batch(loaded, groups))["scores"], expected, rtol=1e-4, atol=1e-6
        )
