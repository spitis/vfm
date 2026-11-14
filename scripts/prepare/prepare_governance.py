"""Synthetic governance benchmark: copy and validate the shipped files.

Source: ``benchmarks/governance/`` in this repository (our own synthetic data, released
under CC BY 4.0, so it is redistributed as-is; nothing is downloaded).

Splits written to ``data/governance/`` (records, label counts in benchmarks/governance/README.md):
    train          2,247   finetuning set (130 interactions, 3 regulations)
    dev              302   in-distribution dev set (20 interactions)
    dev300_s1..s3    299   per-seed dev subsets for early stopping and checkpoint selection
    test             920   in-distribution test set (60 interactions)
    test_transfer    951   transfer test set (40 interactions; adds HIPAA and PCI-DSS)

Each file is checked for the record format, the label vocabulary, and unique record ids.
"""
from __future__ import annotations

import argparse
import os
import shutil

from _common import REPO_ROOT

from vfm.configuration_vfm import LABEL_NAMES
from vfm.data import read_jsonl, record_id

SPLITS = {"train": 2247, "dev": 302, "dev300_s1": 299, "dev300_s2": 299, "dev300_s3": 299,
          "test": 920, "test_transfer": 951}
SRC_DIR = os.path.join(REPO_ROOT, "benchmarks", "governance")


def validate(path: str, expected: int) -> list:
    records = read_jsonl(path)
    seen = set()
    for r in records:
        assert len(r["criteria"]) == 1 and r["labels"][0] in LABEL_NAMES, f"bad record in {path}"
        rid = record_id(r["context"], r["target"], r["criteria"][0])
        assert r["id"] == rid, f"id mismatch in {path}"
        assert rid not in seen, f"duplicate record in {path}"
        seen.add(rid)
    assert len(records) == expected, f"{path}: expected {expected} records, found {len(records)}"
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "governance"),
                    help="Output directory (default: data/governance)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split, expected in SPLITS.items():
        src = os.path.join(SRC_DIR, f"{split}.jsonl")
        records = validate(src, expected)
        shutil.copyfile(src, os.path.join(args.out_dir, f"{split}.jsonl"))
        print(f"  {split}: {len(records)} records ok")
    for name in ("criteria.json", "regulations.json"):
        shutil.copyfile(os.path.join(SRC_DIR, name), os.path.join(args.out_dir, name))
    print(f"copied to {args.out_dir}")


if __name__ == "__main__":
    main()
