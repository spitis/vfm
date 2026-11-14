#!/usr/bin/env python3
"""Prepare the PRM800K step-verification benchmark (one criterion per reasoning step).

Source: Hugging Face dataset ``Birchlabs/openai-prm800k-stepwise-critic`` (a stepwise
re-packaging of OpenAI's PRM800K), https://huggingface.co/datasets/Birchlabs/openai-prm800k-stepwise-critic.
License: MIT (dataset card; the original PRM800K release, https://github.com/openai/prm800k,
is also MIT). This script downloads the data from the
Hugging Face Hub; this repository does not redistribute it.

Conversion (as in the paper): 35000 source rows are sampled; steps rated +1 (YES) or
-1 (NO) are kept (neutral steps are dropped); problems are split 66/17/17 into
train/dev/test so no problem appears in two splits; YES/NO is balanced within each split.
The context is "Problem: ..." followed by the prior steps ("Step i: ..."), the target is the
step under verification, and the criterion asks whether the step is a valid and correct step
toward solving the problem.

Output splits (sizes as in the paper): train 11144, dev 2700, test 2932 (identical
context/step pairs occurring more than once are kept), plus, from the manifest, the
300-record dev subsets dev300_s1, dev300_s2, dev300_s3 (not written with --no-manifest).

Usage:
    python scripts/prepare/prepare_prm800k.py [--out-dir DIR] [--no-manifest] [--strict]
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from datasets import load_dataset

from _common import add_common_args, load_manifest, select_by_manifest, write_split
from vfm.data import make_record

CRITERION = ("Is the reasoning in the text a valid and correct step toward solving "
             "the given math problem?")


def extract_step(row):
    """One HF row -> (problem, record), or None for neutral/unlabeled rows."""
    instruction = row.get("instruction", "")
    step = row.get("next_response", "")
    prior = row.get("responses", []) or []
    rating = row.get("rating", None)
    if not instruction or not step or rating is None or rating not in (1, -1):
        return None
    parts = [f"Problem: {instruction}"] + [f"Step {i + 1}: {p}" for i, p in enumerate(prior)]
    rec = make_record([{"role": "user", "content": "\n".join(parts)}], step, CRITERION,
                      "YES" if rating == 1 else "NO", meta={"step_index": len(prior)})
    return instruction, rec


def balance_records(records, rng):
    yes = [r for r in records if r["labels"][0] == "YES"]
    no = [r for r in records if r["labels"][0] == "NO"]
    if not yes or not no:
        return records
    n = min(len(yes), len(no))
    rng.shuffle(yes)
    rng.shuffle(no)
    balanced = yes[:n] + no[:n]
    rng.shuffle(balanced)
    return balanced


def sample_splits(seed=42, max_samples=35000, train_ratio=0.66, dev_ratio=0.17, balance=True):
    ds = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    if max_samples and max_samples < len(indices):
        indices = indices[:max_samples]
    groups = defaultdict(list)
    for idx in indices:
        step = extract_step(ds[int(idx)])
        if step is not None:
            groups[step[0]].append(step[1])
    problems = list(groups)
    rng.shuffle(problems)
    n_train, n_dev = int(len(problems) * train_ratio), int(len(problems) * dev_ratio)
    keys = {"train": problems[:n_train], "dev": problems[n_train:n_train + n_dev],
            "test": problems[n_train + n_dev:]}
    splits = {}
    for name, ps in keys.items():
        recs = [r for p in ps for r in groups[p]]
        splits[name] = balance_records(recs, rng) if balance else recs
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "prm800k")
    args = ap.parse_args()
    splits = sample_splits(seed=args.seed)
    manifest = None if args.no_manifest else load_manifest("prm800k")
    if manifest is None:
        for name, recs in splits.items():
            write_split(args.out_dir, name, recs)
        return
    pool = [r for name in ("dev", "test", "train") for r in splits[name]]
    for name in manifest["splits"]:
        write_split(args.out_dir, name, select_by_manifest(pool, manifest, name, args.strict))


if __name__ == "__main__":
    main()
