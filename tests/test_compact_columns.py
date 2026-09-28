import pytest
from datasets import Dataset
from test_dataset_training import native_case

from bekko_system_one.dataset_schema import validate_row
from bekko_system_one.dataset_training import training_view
from bekko_system_one.release import DecisionSource, render_group
from bekko_system_one.stratification import select_balanced_cases


def test_rows_without_schema_and_legacy_have_identical_training_semantics():
    original = native_case()
    original["legacy_aux_json"] = "{}"
    compact = {k: v for k, v in original.items() if k not in {"schema_version", "legacy_aux_json"}}
    validate_row(compact)
    assert training_view(compact) == training_view(original)
    for i in range(len(original["input"]["decisions"])):
        assert render_group(compact, i) == render_group(original, i)
    dataset = Dataset.from_list([compact])
    assert len(DecisionSource(dataset)) == len(original["targets"])
    assert select_balanced_cases(dataset, cap=1, seed=42, name="compact")[0] == [0]


@pytest.mark.parametrize("version", [None, "unknown"])
def test_explicit_incompatible_schema_still_rejected(version):
    row = native_case()
    row["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        validate_row(row)
    with pytest.raises(ValueError, match="schema_version"):
        training_view(row)
    with pytest.raises(ValueError, match="Expected"):
        DecisionSource(Dataset.from_list([row]))
