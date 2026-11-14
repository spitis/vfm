"""VFM record format: record ids and JSONL input/output.

A VFM record pairs one (context, target) with one or more natural-language criteria:

    {
      "id": "3f1c...",                                   # optional; see record_id()
      "context": [{"role": "user", "content": "..."}],   # chat messages before the target
      "target": "...",                                   # text being verified ("" if none)
      "criteria": ["Does the response ...?", ...],
      "labels": ["YES", "NO", "N/A", ...],               # optional at inference time
      "meta": {...}                                      # optional, ignored by the model
    }

Labels are YES (satisfied), NO (violated) and N/A (inapplicable or undeterminable).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from .configuration_vfm import LABEL_NAMES

Messages = List[Dict[str, Any]]


def record_id(context: Messages, target: str, criterion: str) -> str:
    """Stable 16-hex-digit id of one (context, target, criterion) triple.

    Used by the prepare scripts to select exactly the records evaluated in the paper
    without redistributing their text: benchmarks/manifests/<name>.json lists ids only.
    """
    msgs = [{"role": m["role"], "content": m["content"]} for m in context]
    payload = json.dumps([msgs, target, criterion], ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_record(context: Messages, target: str, criterion: str, label: str,
                meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a single-criterion record with its id (used by the prepare scripts)."""
    if label not in LABEL_NAMES:
        raise ValueError(f"label must be one of {LABEL_NAMES}, got {label!r}")
    rec = {"id": record_id(context, target, criterion), "context": context, "target": target,
           "criteria": [criterion], "labels": [label]}
    if meta:
        rec["meta"] = meta
    return rec


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
