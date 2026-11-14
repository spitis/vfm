"""Shared helpers for the benchmark prepare scripts.

Each prepare script downloads a benchmark from its original source, converts it to VFM
records (see ``vfm.data``), and writes ``<out_dir>/<split>.jsonl``. When a manifest exists
in ``benchmarks/manifests/<name>.json``, the script keeps exactly the records used in the
paper (matched by ``vfm.data.record_id``) and checks their labels; the manifest stores
record ids and labels only, never benchmark text.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vfm.data import record_id, write_jsonl  # noqa: E402

MANIFEST_DIR = os.path.join(REPO_ROOT, "benchmarks", "manifests")


def load_manifest(name: str) -> Optional[Dict]:
    path = os.path.join(MANIFEST_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def rec_ids(rec: Dict) -> List[str]:
    return [record_id(rec["context"], rec["target"], c) for c in rec["criteria"]]


def select_by_manifest(records: Iterable[Dict], manifest: Dict, split: str,
                       strict: bool = False) -> List[Dict]:
    """Keep the single-criterion records whose id is listed for `split`.

    A few benchmarks contain the same (context, target, criterion) triple more than once;
    ``manifest["counts"][split][id]`` then gives the multiset of its labels, and the record
    is emitted once per occurrence. Prints coverage and label agreement; with strict=True,
    missing records or label mismatches raise an error.
    """
    wanted: Dict[str, str] = manifest["splits"][split]
    counts: Dict[str, Dict[str, int]] = manifest.get("counts", {}).get(split, {})
    pool: Dict[str, List[Dict]] = {}
    for r in records:
        for i in rec_ids(r):
            pool.setdefault(i, []).append(r)
    out, missing, mismatched, total = [], 0, 0, 0
    for i, label in wanted.items():
        targets = [lab for lab, k in counts[i].items() for _ in range(k)] if i in counts else [label]
        total += len(targets)
        cands = pool.get(i)
        if not cands:
            missing += len(targets)
            continue
        unused = list(cands)
        for lab in targets:
            pick = next((r for r in unused if r["labels"][0] == lab), None)
            if pick is None:
                pick = next((r for r in cands if r["labels"][0] == lab), None)
            if pick is None:
                mismatched += 1
                pick = cands[0]
            else:
                if pick in unused:
                    unused.remove(pick)
            out.append(pick)
    msg = (f"[{manifest['name']}/{split}] matched {total - missing}/{total} paper records"
           f" ({missing} missing, {mismatched} label mismatches)")
    print(msg)
    if strict and (missing or mismatched):
        raise SystemExit(msg)
    return out


def write_split(out_dir: str, split: str, records: List[Dict]) -> None:
    path = os.path.join(out_dir, f"{split}.jsonl")
    n = write_jsonl(path, records)
    labels = Counter(l for r in records for l in r["labels"])
    print(f"  wrote {n} records -> {os.path.relpath(path, os.getcwd())}  labels={dict(labels)}")


def add_common_args(parser, name: str) -> None:
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", name),
                        help="Output directory (default: data/%s)" % name)
    parser.add_argument("--no-manifest", action="store_true",
                        help="Ignore the paper manifest and write the freshly sampled splits.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any paper record cannot be reconstructed.")
    parser.add_argument("--seed", type=int, default=42)
