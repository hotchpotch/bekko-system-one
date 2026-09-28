"""Observed sampling volume; token counts describe accepted, truncated inputs."""

from collections import Counter


def usage_kind(group):
    return (
        "ranking" if group.metadata is not None and group.metadata.kind == "ranking" else group.task
    )


def batch_usage(prepared):
    """Count each decision once, independent of packing, shared prefixes and OOM retries."""
    totals = {}
    for group in prepared:
        item = totals.setdefault(usage_kind(group), Counter())
        item.update(
            decisions=1,
            candidates=len(group.documents),
            query_tokens=len(group.query),
            candidate_tokens=sum(map(len, group.documents)),
            attention_work_tokens=group.cost,
        )
    return {kind: dict(counts) for kind, counts in totals.items()}


class TrainingUsage:
    """Exact unique case counts per source and kind, only when IDs are available."""

    def __init__(self):
        self.sources = {}
        self.cases = {}
        self.kind_cases = {}

    def update(self, source, groups, usage):
        tasks = self.sources.setdefault(source, {})
        self.cases.setdefault(source, set())
        for kind, counts in usage.items():
            tasks.setdefault(kind, Counter()).update(counts)
        for group in groups:
            kind = usage_kind(group)
            seen = self.kind_cases.setdefault((source, kind), set())
            case_id = group.metadata.case_id if group.metadata is not None else None
            if case_id:
                self.cases[source].add(case_id)
                seen.add(case_id)
                tasks[kind]["identified_decisions"] += 1

    def result(self):
        sources = {}
        for source, tasks in self.sources.items():
            summaries = {}
            for kind, counts in tasks.items():
                identified = counts["identified_decisions"]
                summaries[kind] = dict(
                    counts,
                    identified_decisions=identified,
                    unique_cases=len(self.kind_cases[source, kind]) if identified else None,
                    case_identity_complete=identified == counts["decisions"],
                )
            sources[source] = dict(
                tasks=summaries,
                unique_cases=len(self.cases[source]) if self.cases[source] else None,
                case_identity_complete=all(v["case_identity_complete"] for v in summaries.values()),
            )
        return dict(
            sources=sources,
            token_counting="post-truncation per decision; excludes padding and OOM replay; "
            "query_tokens counts each prefix once per decision; attention_work_tokens "
            "counts it once per candidate",
        )
