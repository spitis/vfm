#!/usr/bin/env python3
"""Export CriteriaBank (Hugging Face dataset `spitis/CriteriaBank`) to VFM training records.

CriteriaBank is the pretraining corpus of the paper: 393,531 (context, target, criterion,
label) rows from five streams. The dataset ships the text of 316,974 rows (configs
stream_a, stream_b, stream_c and stream_d), which this script exports. The other 76,439 rows
(configs stream_b_keyed and stream_c_plus_keyed) carry keys into WildChat-1M, WildJailbreak
and LMSYS-Chat-1M instead of text; the dataset's `rehydrate.py` rebuilds them once you have
accepted those datasets' terms, and `--keyed` adds the files it writes.

Each row becomes one single-criterion record. Fields stored as JSON are decoded: message
contents in `content_structured`, tool calls, and targets in `target_structured`. A tool-call
field that holds a single object rather than a list is wrapped in a list.

Output: data/criteriabank/train.jsonl

Usage:
    python scripts/prepare/prepare_criteriabank.py
    python scripts/prepare/prepare_criteriabank.py --keyed stream_b_keyed_text.parquet stream_c_plus_text.parquet
"""
import argparse
import json
import os

from _common import REPO_ROOT, write_split

STREAMS = ("stream_a", "stream_b", "stream_c", "stream_d")


def to_record(row):
    context = []
    for m in row["context"]:
        msg = {k: v for k, v in m.items() if v is not None and k != "content_structured"}
        if m.get("content_structured") is not None:
            msg["content"] = json.loads(m["content_structured"])
        if m.get("tool_calls") is not None:
            calls = json.loads(m["tool_calls"])
            msg["tool_calls"] = calls if isinstance(calls, list) else [calls]
        context.append(msg)
    structured = row.get("target_structured")
    target = json.loads(structured) if structured is not None else row["target"]
    return {"id": row["id"], "context": context, "target": target, "criteria": [row["criterion"]],
            "labels": [row["label"]], "meta": {"stream": row["stream"], "source_dataset": row["source_dataset"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="spitis/CriteriaBank",
                    help="Hub id, or a local copy of the dataset repository")
    ap.add_argument("--keyed", nargs="+", default=[],
                    help="Files written by the dataset's rehydrate.py (.parquet or .jsonl)")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "criteriabank"))
    args = ap.parse_args()
    from datasets import load_dataset

    keyed = [row for p in args.keyed
             for row in load_dataset("parquet" if p.endswith(".parquet") else "json", data_files=p, split="train")]
    train = [to_record(row) for name in STREAMS for row in load_dataset(args.dataset, name, split="train")]
    train += [to_record(row) for row in keyed]
    write_split(args.out_dir, "train", train)


if __name__ == "__main__":
    main()
