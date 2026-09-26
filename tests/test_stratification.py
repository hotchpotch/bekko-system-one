import copy
from collections import Counter

from datasets import Dataset

from bekko_system_one.stratification import select_balanced_cases


def row(label, *, labels=("false", "true"), task="noul", probabilities=None):
    return dict(
        decisions=[
            dict(
                type=task,
                decision_id="axis",
                options=[dict(id=k, description=k) for k in labels],
                target=dict(
                    option_ids=list(labels),
                    probabilities=probabilities or [float(k == label) for k in labels],
                ),
            )
        ]
    )


def sample(rows, cap=5):
    return select_balanced_cases(Dataset.from_list(rows), cap=cap, seed=42, name="test")


def test_odd_binary_cap_determinism_and_shortage():
    rows = [row("false") for _ in range(100)] + [row("true") for _ in range(10)]
    ids, report = sample(rows)
    assert sorted(
        Counter(rows[i]["decisions"][0]["target"]["probabilities"][1] for i in ids).values()
    ) == [2, 3]
    assert sample(rows) == (ids, report)
    ids, report = sample(rows[:101])
    assert 100 in ids and len(set(ids)) == 5
    assert sample(rows[:2])[0] == [0, 1]
    assert any(x["unavailable"] for x in sample(rows[:100])[1]["labels"])


def test_ties_soft_bands_and_targets_unchanged():
    rows = [
        row("false", probabilities=p)
        for p in [[1.0, 0.0], [0.9, 0.1], [0.6, 0.4], [0.5, 0.5], [0.0, 1.0]]
        for _ in range(10)
    ]
    before = copy.deepcopy(rows)
    ids, report = sample(rows, 15)
    assert rows == before and len(ids) == 15
    assert {x["label"]: x["selected"] for x in report["labels"]} == {
        "false": 5,
        "true": 5,
        "__tie__": 5,
    }
    negative = next(x for x in report["labels"] if x["label"] == "false")
    assert sorted(x["selected"] for x in negative["confidence_bands"].values()) == [1, 2, 2]


def test_many_classes_do_not_expand_cap_and_report_missing():
    labels = tuple(map(str, range(8)))
    rows = [row(k, labels=labels, task="choice") for k in labels for _ in range(3)]
    ids, report = sample(rows)
    assert len(ids) == 5 and not report["expanded_for_class_coverage"]
    assert sum(x["missing"] for x in report["labels"]) == 3


def test_multiaxis_coverage_keeps_cases_and_reports_conflicts():
    rows = [row("false") for _ in range(100)]
    for i, r in enumerate(rows):
        d = row("true" if i == 99 else "false")["decisions"][0]
        d["decision_id"] = "rare"
        r["decisions"].append(d)
    ids, report = sample(rows)
    assert 99 in ids and len(ids) == len(set(ids)) == 5
    assert all(len(rows[i]["decisions"]) == 2 for i in ids)
    assert report["policy"] == "multilabel_case_preserving"


def test_variable_qa_and_score_remain_uniform():
    for task in ["choice", "score"]:
        rows = [row("a", labels=("a", "b"), task=task) for _ in range(10)]
        for i, r in enumerate(rows):
            r["decisions"][0]["options"][0]["description"] = f"answer {i}"
        ids, report = sample(rows)
        assert len(ids) == 5 and report["policy"] == "uniform" and not report["labels"]
