"""DynaBench: multi-rule guardrail benchmark (DynaGuard), converted to VFM records.

Source:   https://huggingface.co/datasets/tomg-group-umd/DynaBench  (Hoover et al., 2025)
License:  MIT (dataset card). Only the ``DynaBench`` (test) and ``DynaBenchTrain`` configs are
          used; the ``DynaBenchSafetyMix`` config, which mixes in BeaverTails, WildGuard,
          ToxicChat and Aegis 2.0 samples, is never read.
Download-only, not redistributed: the dialogues are downloaded by this script. The repository
ships only our rule translations (``benchmarks/dynabench/rule_translations.json.gz``) and the
ids of the dev policies.

Rules are free-text policy statements. A one-time, label-blind LLM pass translated each rule
into one or more atomic questions, worded so that YES means the rule was violated (for
example "Never use emojis." became "Did the agent use emojis in any of its responses?"). The
translations ship with the repository, so this script needs no API access. A criterion takes
its rule's trace-level label: YES if the rule was violated in the trace, NO if it was
followed, and N/A (training only) if the source marks the rule as ambiguous.

One record per trace: context = the full transcript as one user message, target empty,
criteria = the translated questions of every rule. ``meta`` holds ``base_id``, ``trace_label``
(FAIL if any rule is violated, else PASS), ``rule_index`` and ``sub_idx`` (rule and question
index of each criterion, 0-based), ``rules_violated`` (gold violated rule indices, 0-based),
and, for test, ``rules`` (the raw rule texts) and ``failure_mode``.

Splits written to ``<out_dir>/``:
    test    543 traces (267 FAIL), 11,486 criteria, binary labels
    train   13,997 traces, 75,031 criteria from the first 14,000 DynaBenchTrain policies of
            equal PASS/FAIL share (seed 0, at most 10 rules per policy)
    dev     502 held-out DynaBenchTrain policies (ids in benchmarks/dynabench/dev_policy_ids.json),
            9,227 criteria, binary labels (rules marked ambiguous are dropped)
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional

from _common import REPO_ROOT, add_common_args, load_manifest, write_split

from vfm.data import record_id

HF_REPO = "tomg-group-umd/DynaBench"
BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "dynabench")
NUM_POLICIES = 14000
RULES_PER_POLICY = 10


# ---------------------------------------------------------------- parsing helpers

def parse_numbered_string(s: str) -> List[str]:
    rules = []
    for p in re.split(r"\n(?=\d+[.\)]\s)", s):
        p = p.strip()
        if not p:
            continue
        cleaned = re.sub(r"^\d+[.\)]\s*", "", p).strip()
        if cleaned:
            rules.append(cleaned)
    return rules


def parse_policy(policy) -> List[str]:
    if isinstance(policy, str):
        return parse_numbered_string(policy)
    if isinstance(policy, list):
        if len(policy) == 1 and isinstance(policy[0], str) and re.match(r"^\d+[.\)]", policy[0].strip()):
            return parse_numbered_string(policy[0])
        return [r.strip() for r in policy if r and r.strip()]
    return []


def _parse_by_turn(entries) -> List[list]:
    """Parse "1. ['PASS', 'FAIL']\\n"-style strings into lists, one per rule."""
    if not entries:
        return []
    if isinstance(entries, str):
        try:
            entries = ast.literal_eval(entries)
        except (ValueError, SyntaxError):
            return []
    result = []
    for entry in entries:
        m = re.search(r"\[.*\]", entry) if isinstance(entry, str) else None
        if not m:
            result.append([])
            continue
        try:
            result.append(list(ast.literal_eval(m.group(0))))
        except (ValueError, SyntaxError):
            result.append([])
    return result


def aggregate_rule(turn_labels: list, turn_ambiguity: list):
    """A rule fails if any turn fails; it is ambiguous only if every turn is ambiguous."""
    if not turn_labels:
        return None, None
    label = "FAIL" if any(l == "FAIL" for l in turn_labels) else "PASS"
    ambiguity = 2 if turn_ambiguity and all(a == 2 for a in turn_ambiguity) else 1
    return label, ambiguity


def as_list(value) -> list:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return list(value or [])


def vfm_label(compliance: str, ambiguity: int, allow_na: bool) -> str:
    """YES = the rule was violated (FAIL); NO = it was followed; N/A = ambiguous (train only)."""
    if allow_na and ambiguity == 2:
        return "N/A"
    return "YES" if compliance == "FAIL" else "NO"


def load_translations() -> Dict[str, dict]:
    with gzip.open(os.path.join(BENCH_DIR, "rule_translations.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- record building

def trace_record(base_id: str, transcript: str, rules: List[str], rule_flags: Dict[int, tuple],
                 translations: Dict[str, dict], trace_label: str, allow_na: bool,
                 meta_extra: Optional[dict] = None) -> Optional[dict]:
    """One trace-level record. rule_flags maps rule index -> (FAIL/PASS, ambiguity)."""
    criteria, labels, rule_index, sub_idx = [], [], [], []
    for r_idx in sorted(rule_flags):
        t = translations.get(rules[r_idx])
        if not t:
            continue
        compliance, ambiguity = rule_flags[r_idx]
        for s_idx, q in enumerate(t["sub_questions"]):
            criteria.append(q)
            labels.append(vfm_label(compliance, ambiguity, allow_na))
            rule_index.append(r_idx)
            sub_idx.append(s_idx)
    if not criteria:
        return None
    ctx = [{"role": "user", "content": transcript}]
    meta = {"base_id": base_id, "trace_label": trace_label, "rule_index": rule_index, "sub_idx": sub_idx}
    meta.update(meta_extra or {})
    return {"id": record_id(ctx, "", criteria[0]), "context": ctx, "target": "",
            "criteria": criteria, "labels": labels, "meta": meta}


def build_test(translations) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, "DynaBench", split="test")
    out = []
    for ex in ds:
        rules = parse_policy(ex["policy"])
        if not rules:
            continue
        md = ex.get("metadata") or "{}"
        md = json.loads(md) if isinstance(md, str) else md
        violated = md.get("rules_violated", []) or []
        flags = {i: ("FAIL" if (i + 1) in violated else "PASS", 1) for i in range(len(rules))}
        rec = trace_record(ex["base_id"], ex["transcript"], rules, flags, translations, ex["label"],
                           allow_na=False, meta_extra={"rules": rules, "rules_violated": [i - 1 for i in violated],
                                                       "failure_mode": md.get("failure_mode"),
                                                       "num_rules": len(rules)})
        if rec:
            out.append(rec)
    return out


def _train_rows():
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, "DynaBenchTrain", split="train")
    return ds.select_columns(["policy", "transcript", "label", "rules_violated",
                              "label_by_rule_by_turn", "ambiguity_by_rule_by_turn", "base_id"])


def _row_rules(ex):
    rules = parse_policy(ex["policy"])
    lbrt = _parse_by_turn(ex.get("label_by_rule_by_turn"))
    abrt = _parse_by_turn(ex.get("ambiguity_by_rule_by_turn"))
    lbrt += [[]] * (len(rules) - len(lbrt))
    abrt += [[]] * (len(rules) - len(abrt))
    per_rule = {}
    for i in range(len(rules)):
        label, ambiguity = aggregate_rule(lbrt[i], abrt[i])
        if label is not None:
            per_rule[i] = (label, ambiguity)
    return rules, per_rule


def build_train(rows, translations, seed: int) -> List[dict]:
    """Select policies and rules as in the paper, then build trace-level records."""
    rng = random.Random(seed)
    target_pass = NUM_POLICIES // 2
    target_fail = NUM_POLICIES - target_pass
    pass_pool, fail_pool = [], []
    for ex in rows:
        if len(pass_pool) >= target_pass and len(fail_pool) >= target_fail:
            break
        if ex["label"] == "PASS" and len(pass_pool) < target_pass:
            pass_pool.append(ex)
        elif ex["label"] == "FAIL" and len(fail_pool) < target_fail:
            fail_pool.append(ex)

    out = []
    for ex in pass_pool + fail_pool:
        rules, per_rule = _row_rules(ex)
        if not rules or not per_rule:
            continue
        violated = [i for i, (l, _) in per_rule.items() if l == "FAIL"]
        compliant = [i for i, (l, _) in per_rule.items() if l == "PASS"]
        if ex["label"] == "FAIL":
            sampled = list(violated)
            k = max(0, RULES_PER_POLICY - len(violated))
            if compliant:
                sampled.extend(rng.sample(compliant, min(k, len(compliant))))
        else:
            sampled = rng.sample(compliant, min(RULES_PER_POLICY, len(compliant)))
        if not sampled:
            continue
        flags = {i: per_rule[i] for i in sampled}
        rec = trace_record(ex["base_id"], ex["transcript"], rules, flags, translations, ex["label"],
                           allow_na=True, meta_extra={"rules_violated": [i - 1 for i in as_list(ex["rules_violated"])]})
        if rec:
            out.append(rec)
    return out


def build_dev(rows, translations) -> List[dict]:
    with open(os.path.join(BENCH_DIR, "dev_policy_ids.json")) as f:
        wanted = set(json.load(f)["base_ids"])
    out = []
    for ex in rows:
        if ex["base_id"] not in wanted:
            continue
        rules, per_rule = _row_rules(ex)
        flags = {i: v for i, v in per_rule.items() if v[1] != 2}
        rec = trace_record(ex["base_id"], ex["transcript"], rules, flags, translations, ex["label"],
                           allow_na=False, meta_extra={"rules_violated": [i - 1 for i in as_list(ex["rules_violated"])]})
        if rec:
            out.append(rec)
    return out


# ---------------------------------------------------------------- manifest selection

def select(records: List[dict], manifest: dict, split: str, strict: bool) -> List[dict]:
    """Keep the criteria listed for `split`, trace by trace.

    A question text can recur within a trace (for different rules, sometimes with different
    labels); ``manifest["counts"][split][id]`` gives the label multiset of such ids.
    """
    wanted = manifest["splits"][split]
    counts = manifest.get("counts", {}).get(split, {})
    left: Dict[str, Counter] = {i: Counter(counts[i]) if i in counts else Counter({lab: 1})
                                for i, lab in wanted.items()}
    expected = {i: set(c) for i, c in left.items()}
    out, n_kept, mismatched = [], 0, 0
    for rec in records:
        ctx, tgt = rec["context"], rec["target"]
        keep = []
        for j, c in enumerate(rec["criteria"]):
            rid = record_id(ctx, tgt, c)
            if rid not in wanted:
                continue
            label = rec["labels"][j]
            if left[rid][label] <= 0:
                mismatched += label not in expected[rid]
                continue
            left[rid][label] -= 1
            keep.append(j)
        if not keep:
            continue
        new = dict(rec)
        new["criteria"] = [rec["criteria"][j] for j in keep]
        new["labels"] = [rec["labels"][j] for j in keep]
        meta = dict(rec["meta"])
        for key in ("rule_index", "sub_idx"):
            meta[key] = [rec["meta"][key][j] for j in keep]
        new["meta"] = meta
        new["id"] = record_id(ctx, tgt, new["criteria"][0])
        out.append(new)
        n_kept += len(keep)
    total = sum(sum(counts[i].values()) if i in counts else 1 for i in wanted)
    missing = total - n_kept
    msg = (f"[dynabench/{split}] matched {n_kept}/{total} paper criteria in {len(out)} traces"
           f" ({missing} missing, {mismatched} label mismatches)")
    print(msg)
    if strict and (missing or mismatched):
        raise SystemExit(msg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "dynabench")
    ap.set_defaults(seed=0)
    ap.add_argument("--splits", default="test,train,dev", help="Comma-separated subset of test,train,dev.")
    args = ap.parse_args()
    wanted = [s for s in args.splits.split(",") if s]

    translations = load_translations()
    manifest = None if args.no_manifest else load_manifest("dynabench")

    rows = _train_rows() if any(s in wanted for s in ("train", "dev")) else None
    for split in wanted:
        if split == "test":
            records = build_test(translations)
        elif split == "train":
            records = build_train(rows, translations, args.seed)
        elif split == "dev":
            records = build_dev(rows, translations)
        else:
            raise SystemExit(f"unknown split {split!r}")
        if manifest and split in manifest["splits"]:
            records = select(records, manifest, split, args.strict)
        write_split(args.out_dir, split, records)
        print(f"  {split}: {len(records)} traces, {sum(len(r['criteria']) for r in records)} criteria,"
              f" {Counter(r['meta']['trace_label'] for r in records)}")


if __name__ == "__main__":
    main()
