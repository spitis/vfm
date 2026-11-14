#!/usr/bin/env python3
"""Prepare the IFEval-style instruction-following benchmark (one criterion per constraint).

Source: Hugging Face dataset ``argilla/ifeval-like-data`` (default config, with both
passing and failing responses), https://huggingface.co/datasets/argilla/ifeval-like-data.
License: "other" on the dataset card (the Qwen license, as the responses were generated with
Qwen2.5-72B-Instruct; https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE). This script downloads the data from
the Hugging Face Hub; this repository does not redistribute it.

Conversion (as in the paper): 15000 source rows are sampled and split 66/17/17 into
train/dev/test at the row level; each (prompt, response) row is exploded into one record per
instruction constraint. The criterion is a natural-language rendering of the constraint
(e.g. "Does the response contain at least 400 words?") and the label is the result of
programmatic IFEval verification (YES/NO); YES/NO is then balanced per constraint type.

The language-detection step (langdetect) is not fully deterministic, so a handful of
language-constraint labels can differ between runs; with the manifest, labels are taken
from it.

The manifest labels are the ones evaluated in the paper. The paper's pipeline checked
keywords:forbidden_words against the quoted list in the criterion text, which never matches,
so those criteria are all labelled YES there; 11 of the 118 in test and 71 of the 457 in
train should be NO. This script verifies against the original word list; that only changes
labels with --no-manifest.

The keywords:frequency and keywords:letter_frequency criteria always say "at least", while
the label checks the constraint's own relation, which is sometimes "less than". This is as
in the paper and is kept so that the record ids match the manifest. In the paper's records
the criterion says "at least" for an upper-bound constraint in 74 test, 70 dev and 336 train
records, and the label disagrees with the question as worded in 71, 63 and 326 of them.

Output splits (sizes as in the paper): train 13635, dev 3403, test 3418 criteria, plus, from
the manifest, the 300-record dev subsets dev300_s1, dev300_s2, dev300_s3 (not written with
--no-manifest).

Extra dependencies: nltk (punkt tokenizer, downloaded on first use) and langdetect.

Usage:
    python scripts/prepare/prepare_ifeval.py [--out-dir DIR] [--no-manifest] [--strict]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset

from _common import add_common_args, load_manifest, rec_ids, select_by_manifest, write_split
from ifeval_checks import constraint_to_criterion, verify_constraint
from vfm.data import make_record

# kwargs keys used by each constraint type
KWARGS_KEYS = {
    "keywords:existence": ["keyword"],
    "keywords:frequency": ["keyword", "frequency", "relation"],
    "keywords:forbidden_words": ["forbidden_words"],
    "keywords:letter_frequency": ["letter", "let_frequency", "relation"],
    "length_constraints:number_words": ["num_words", "relation"],
    "length_constraints:number_sentences": ["num_sentences", "relation"],
    "length_constraints:number_paragraphs": ["num_paragraphs", "relation"],
    "length_constraints:nth_paragraph_first_word": ["num_paragraphs", "first_word"],
    "detectable_format:number_highlighted_sections": ["num_highlights"],
    "detectable_format:number_bullet_lists": ["num_bullets"],
    "detectable_format:json_format": [],
    "detectable_format:title": [],
    "detectable_format:multiple_sections": ["section_spliter", "num_sections"],
    "detectable_format:constrained_response": [],
    "detectable_content:number_placeholders": ["num_placeholders"],
    "detectable_content:postscript": [],
    "punctuation:no_comma": [],
    "change_case:capital_word_frequency": ["capital_frequency", "relation"],
    "change_case:english_capital": [],
    "change_case:english_lowercase": [],
    "startend:end_checker": ["end_phrase"],
    "startend:quotation": [],
    "combination:two_responses": [],
    "combination:repeat_prompt": [],
    "language:response_language": ["language"],
}


def parse_kwargs(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return parsed[0] if parsed else {}
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def convert_row(row):
    """One source row -> one record per constraint, labelled by programmatic verification."""
    prompt, response = row["instruction"], row["response"]
    all_kwargs = parse_kwargs(row["kwargs"])
    out = []
    for cid in row["instruction_id_list"] or []:
        kw = {k: all_kwargs[k] for k in KWARGS_KEYS.get(cid, []) if k in all_kwargs}
        # constraint_to_criterion rewrites list-valued kwargs in place, so pass it a copy and
        # verify against the original values.
        criterion = constraint_to_criterion(cid, dict(kw))
        satisfied = verify_constraint(cid, response, prompt=prompt, **kw)
        if satisfied is None:
            continue
        out.append(make_record([{"role": "user", "content": prompt}], response,
                               criterion, "YES" if satisfied else "NO",
                               meta={"constraint_type": cid}))
    return out


def balance_per_type(records, rng):
    """Downsample the majority label within each constraint type."""
    groups = defaultdict(lambda: {"YES": [], "NO": []})
    for r in records:
        groups[r["meta"]["constraint_type"]][r["labels"][0]].append(r)
    balanced = []
    for ctype in sorted(groups):
        yes, no = groups[ctype]["YES"], groups[ctype]["NO"]
        if not yes or not no:
            balanced.extend(yes)
            balanced.extend(no)
            continue
        n = min(len(yes), len(no))
        rng.shuffle(yes)
        rng.shuffle(no)
        balanced.extend(yes[:n])
        balanced.extend(no[:n])
    rng.shuffle(balanced)
    return balanced


def sample_splits(seed=42, max_samples=15000, train_ratio=0.66, dev_ratio=0.17, balance=True):
    ds = load_dataset("argilla/ifeval-like-data", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    if max_samples and max_samples < len(ds):
        rng.shuffle(indices)
        indices = sorted(indices[:max_samples])
    rng.shuffle(indices)
    n_train, n_dev = int(len(indices) * train_ratio), int(len(indices) * dev_ratio)
    parts = {"train": sorted(indices[:n_train]),
             "dev": sorted(indices[n_train:n_train + n_dev]),
             "test": sorted(indices[n_train + n_dev:])}
    splits = {}
    for name, idx in parts.items():
        records = [r for row in ds.select(idx) for r in convert_row(row)]
        splits[name] = balance_per_type(records, rng) if balance else records
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "ifeval")
    args = ap.parse_args()
    manifest = None if args.no_manifest else load_manifest("ifeval")
    if manifest is None:
        for name, recs in sample_splits(seed=args.seed).items():
            write_split(args.out_dir, name, recs)
        return
    # The language-detection labels are not fully deterministic, which would change the
    # per-type balancing, so select the paper records from the unbalanced pool and take
    # their labels from the manifest.
    splits = sample_splits(seed=args.seed, balance=False)
    pool = [r for recs in splits.values() for r in recs]
    for name, wanted in manifest["splits"].items():
        recs = select_by_manifest(pool, manifest, name)
        expected = sum(sum(manifest.get("counts", {}).get(name, {}).get(i, {l: 1}).values())
                       for i, l in wanted.items())
        if args.strict and len(recs) != expected:
            raise SystemExit(f"[ifeval/{name}] only {len(recs)}/{expected} paper records found")
        for r in recs:
            r["labels"] = [wanted[rec_ids(r)[0]]]
        write_split(args.out_dir, name, recs)


if __name__ == "__main__":
    main()
