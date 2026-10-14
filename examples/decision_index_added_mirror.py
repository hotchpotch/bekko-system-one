"""Rebuild added rows with the small HoVer document mirror; require official hash.

Use after the official kit has acquired the base sources and produced its v1
rows. Requires the kit on PYTHONPATH and tiktoken as an optional dependency.
"""

import argparse
import json
import sqlite3
import unicodedata

import pyarrow.parquet as pq
from decision_index import editions
from decision_index.suite.build import adapters_added
from decision_index.suite.build.layout import Layout
from decision_index.suite.build.release_v2 import cut, gz
from decision_index.suite.download import hub_file
from huggingface_hub import hf_hub_download

MIRROR = "Seungjuhan/hover_original"
REVISION = "e04edde79f89d5664589f02240057409065a9a7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    layout = Layout(args.work)
    expected = editions.get("0.2.1")
    out = layout.suite / "release-v2-rebuilt"
    out.mkdir(exist_ok=True)
    exclusions = json.loads(hub_file("0.2", "excluded-questions.json").read_text())["rows"]
    base = cut(
        layout.suite / "release-v1-rebuilt/selected-rows.jsonl",
        out / "selected-rows.jsonl",
        set(exclusions),
    )
    if base["sha256"] != expected["rows_sha256"]:
        raise ValueError("Base corpus differs from the official hash")
    gz(out / "selected-rows.jsonl")
    mirror = hf_hub_download(
        MIRROR, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
    )
    documents = pq.read_table(mirror).to_pylist()
    database = layout.raw / "hover/needed-documents.sqlite"
    database.parent.mkdir(exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, text TEXT)")
        connection.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?,?)",
            [(unicodedata.normalize("NFD", row["title"]), row["text"]) for row in documents],
        )
    adapters_added.acquire(layout, [n for n in adapters_added.ORDER if n != 61])
    # Only replace the local document lookup. The unmodified official builder
    # must produce the exact same bytes for every request, including HoVer.
    adapters_added.HOVER_DB = ("hover/needed-documents.sqlite", None, None)
    result = adapters_added.build(layout, out / "added-rows.jsonl")
    if result["sha256"] != expected["added_sha256"]:
        raise ValueError("Mirror-derived added corpus differs from the official hash")
    gz(out / "added-rows.jsonl")
    result["hover_document_mirror"] = {"repo": MIRROR, "revision": REVISION}
    (out / "mirror-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
