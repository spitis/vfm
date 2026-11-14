#!/usr/bin/env python3
"""Prepare the ContractNLI benchmark (does a contract stipulate a given provision?).

Source: ContractNLI (Koreeda and Manning, 2021), official release at
https://stanfordnlp.github.io/contract-nli/ (607 non-disclosure agreements, 17 fixed
hypotheses). License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). This script
downloads the official archive; this repository does not redistribute it. If the download is
not possible, download and extract
https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip yourself and pass the
extracted directory (containing train.json, dev.json, test.json) as ``--source-dir``.

Conversion (as in the paper): one record per (agreement, hypothesis) pair. The context is the
agreement text as a single user message, the target is empty, and the criterion is
"Does this agreement stipulate that: <hypothesis>?" (the hypothesis without its final
period). Labels: Entailment -> YES, Contradiction -> NO, NotMentioned -> N/A. The official
train/dev/test split is used.

Output splits (sizes as in the paper): train 7191, dev 1037, test 2091, plus, from the
manifest, the 300-record dev subsets dev300_s1, dev300_s2, dev300_s3 (not written with
--no-manifest).

Usage:
    python scripts/prepare/prepare_contractnli.py [--out-dir DIR] [--source-dir DIR]
                                                  [--no-manifest] [--strict]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
import zipfile

from _common import add_common_args, load_manifest, select_by_manifest, write_split
from vfm.data import make_record

URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
LABELS = {"Entailment": "YES", "Contradiction": "NO", "NotMentioned": "N/A"}
SPLITS = ("train", "dev", "test")


def criterion(hypothesis: str) -> str:
    return f"Does this agreement stipulate that: {hypothesis.rstrip().rstrip('.')}?"


def load_official(source_dir=None):
    if source_dir:
        out = {}
        for s in SPLITS:
            with open(os.path.join(source_dir, f"{s}.json"), encoding="utf-8") as f:
                out[s] = json.load(f)
        return out
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    names = {os.path.basename(n): n for n in zf.namelist()}
    return {s: json.loads(zf.read(names[f"{s}.json"]).decode("utf-8")) for s in SPLITS}


def convert(data):
    hyps = data["labels"]
    records = []
    for doc in data["documents"]:
        for hyp_id, ann in doc["annotation_sets"][0]["annotations"].items():
            records.append(make_record(
                [{"role": "user", "content": doc["text"]}], "",
                criterion(hyps[hyp_id]["hypothesis"]), LABELS[ann["choice"]],
                meta={"doc_id": doc["id"], "hypothesis_id": hyp_id,
                      "original_label": ann["choice"]}))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "contractnli")
    ap.add_argument("--source-dir", default=None,
                    help="Extracted official release (train.json, dev.json, test.json); "
                         "default: download it.")
    args = ap.parse_args()
    try:
        official = load_official(args.source_dir)
    except (OSError, KeyError, zipfile.BadZipFile) as e:
        raise SystemExit(f"Could not load ContractNLI ({e}).\nDownload {URL}, extract it, and "
                         "rerun with --source-dir <extracted contract-nli directory>.")
    splits = {name: convert(data) for name, data in official.items()}
    manifest = None if args.no_manifest else load_manifest("contractnli")
    if manifest is None:
        for name, recs in splits.items():
            write_split(args.out_dir, name, recs)
        return
    pool = [r for recs in splits.values() for r in recs]
    for name in manifest["splits"]:
        write_split(args.out_dir, name, select_by_manifest(pool, manifest, name, args.strict))


if __name__ == "__main__":
    main()
