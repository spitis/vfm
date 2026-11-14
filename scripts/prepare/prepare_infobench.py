#!/usr/bin/env python3
"""Prepare InFoBench (expert split) for zero-shot evaluation.

Source: `ContextualAI/InfoBenchExpertSplit` on the Hugging Face Hub (927 expert-annotated
(instruction, response, decomposed requirement) triples from InFoBench, He et al. 2024,
as used by LMUnit). InFoBench is released under the MIT license; the Hub copy declares no
license of its own. This script downloads the data; the repository does not redistribute it.

Conversion: context = the instruction's input and query (one user turn), target = the
response, criterion = the decomposed requirement (`natural_unit_test`), label YES iff
the expert label is 1.

Output: data/infobench/test.jsonl (927 single-criterion records; 802 YES / 125 NO).
"""
import argparse

from _common import add_common_args, load_manifest, select_by_manifest, write_split

from vfm.data import make_record

NAME = "infobench"


def convert(row):
    inp = (row.get("input") or "").strip()
    query = (row.get("query") or "").strip()
    context = f"{inp}\n\n{query}".strip() if inp else query
    return make_record([{"role": "user", "content": context}], row["response"], row["natural_unit_test"],
                       "YES" if int(row["label"]) == 1 else "NO",
                       meta={"id": row["id"], "difficulty": row.get("difficulty")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, NAME)
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("ContextualAI/InfoBenchExpertSplit", split="test")
    records = [convert(r) for r in ds]
    manifest = None if args.no_manifest else load_manifest(NAME)
    if manifest is not None:
        records = select_by_manifest(records, manifest, "test", strict=args.strict)
    write_split(args.out_dir, "test", records)


if __name__ == "__main__":
    main()
