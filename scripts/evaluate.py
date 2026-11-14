#!/usr/bin/env python3
"""Evaluate a VFM checkpoint on a prepared benchmark, with the paper's metrics.

Examples (run from the repository root):
  # A released model, zero-shot (ROC-AUC as in the Appendix table "Zero-shot ROC-AUC")
  python scripts/evaluate.py --model spitis/VFM-0.6B --benchmark infobench

  # Finetuned checkpoint(s): dev-select the checkpoint and threshold, report test + 95% CI
  python scripts/evaluate.py --model "runs/prm800k_s1/checkpoint-*" --benchmark prm800k \
      --dev data/prm800k/dev300_s1.jsonl

  # INSURE-Dial: checkpoint selected by 5-fold cross-validation over the real calls
  python scripts/evaluate.py --model "runs/insure_s1/checkpoint-*" --benchmark insure_dial --cv-folds

Protocols (see BENCHMARKS below):
  binary   P(YES) renormalized over YES/NO; YES iff P >= threshold (dev-tuned if --dev,
           else 0.5); primary metric balanced accuracy or F1 (positive class YES).
  ternary  argmax over YES / NO / N/A; accuracy or macro-F1 over the three classes.
  trace    DynaBench: a dialogue FAILs iff any criterion's violation probability is
           >= threshold (dev-tuned); F1 with FAIL as the positive class.
Binary and ternary results also report threshold-free ROC-AUC: the renormalized P(YES) on
binary benchmarks and the raw three-way P(YES) on the YES/NO subset of ternary benchmarks.
Governance selects checkpoints by dev macro-F1 and reports accuracy. INSURE-Dial confidence
intervals resample whole calls.

Writes a JSON summary (and per-criterion predictions with --save-predictions).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vfm import VFM  # noqa: E402
from vfm import metrics as M  # noqa: E402
from vfm.data import read_jsonl  # noqa: E402

BENCHMARKS: Dict[str, Dict] = {
    "beavertails": {"protocol": "binary", "metric": "f1"},
    "ifeval": {"protocol": "binary", "metric": "balanced_accuracy"},
    "prm800k": {"protocol": "binary", "metric": "balanced_accuracy"},
    "aggrefact": {"protocol": "binary", "metric": "balanced_accuracy"},
    "infobench": {"protocol": "binary", "metric": "balanced_accuracy"},
    "contractnli": {"protocol": "ternary", "metric": "macro_f1", "max_length": 8192},
    "governance": {"protocol": "ternary", "metric": "accuracy", "select_metric": "macro_f1"},
    "insure_dial": {"protocol": "ternary", "metric": "accuracy", "cluster": "call_id"},
    "dynabench": {"protocol": "trace", "metric": "f1"},
}


# ----------------------------------------------------------------------------- scoring
def score_records(vfm: VFM, records: List[Dict], batch_size: int, max_length: int,
                  chunk_size: int) -> List[Dict]:
    """Per-criterion scores for every (record, criterion), in record order."""
    empty = [r.get("id", i) for i, r in enumerate(records) if not r["criteria"]]
    if empty:
        raise SystemExit(f"{len(empty)} records have an empty 'criteria' list (first: {empty[0]}); "
                         "remove them before evaluating")
    single = [i for i, r in enumerate(records) if len(r["criteria"]) == 1]
    multi = [i for i, r in enumerate(records) if len(r["criteria"]) > 1]
    out: Dict[int, List[Dict]] = {}
    rows = [{"context": records[i]["context"], "target": records[i]["target"],
             "criterion": records[i]["criteria"][0]} for i in single]
    for s in tqdm(range(0, len(rows), batch_size), desc="score", leave=False):
        res = vfm.verify_batch(rows[s:s + batch_size], batch_size=batch_size, max_length=max_length)
        for i, r in zip(single[s:s + batch_size], res):
            out[i] = [r]
    for i in tqdm(multi, desc="score (multi-criterion)", leave=False):
        r = records[i]
        out[i] = vfm.verify(r["context"], r["target"], r["criteria"], chunk_size=chunk_size,
                            max_length=max_length)
    scored = []
    for i, r in enumerate(records):
        for j, (s, label) in enumerate(zip(out[i], r["labels"])):
            scored.append({"record": i, "criterion_index": j, "label": label,
                           "p_yes": s["p_yes"], "p_no": s["p_no"], "p_na": s["p_na"],
                           "p_yes_binary": s["p_yes_binary"], "verdict": s["verdict"]})
    return scored


# ----------------------------------------------------------------------------- metrics
def binary_eval(scored, metric, threshold, n_boot):
    s = np.array([x["p_yes_binary"] for x in scored])
    y = [x["label"] for x in scored]
    m = M.binary_metrics(M.threshold_preds(s, threshold), y)
    ylab = np.array(y, dtype=object)

    def fn(idx):
        return M.binary_metrics(M.threshold_preds(s[idx], threshold), ylab[idx])[metric]
    lo, hi = M.bootstrap_ci(fn, len(y), n_boot=n_boot)
    return {"metric": metric, "value": m[metric], "ci95": [lo, hi], "threshold": threshold, **m}


def ternary_eval(scored, metric, n_boot, clusters=None):
    preds = np.array([x["verdict"] for x in scored], dtype=object)
    y = np.array([x["label"] for x in scored], dtype=object)
    m = M.ternary_metrics(list(preds), list(y))

    def fn(idx):
        return M.ternary_metrics(list(preds[idx]), list(y[idx]))[metric]
    lo, hi = M.bootstrap_ci(fn, len(y), n_boot=n_boot, clusters=clusters)
    return {"metric": metric, "value": m[metric], "ci95": [lo, hi], **m}


def trace_scores(records, scored):
    """Per-record max violation probability and per-rule scores (DynaBench)."""
    by_rec: Dict[int, List[Dict]] = {}
    for x in scored:
        by_rec.setdefault(x["record"], []).append(x)
    traces = []
    for i, r in enumerate(records):
        meta = r.get("meta", {})
        rules = meta.get("rule_index") or list(range(len(r["criteria"])))
        per_rule: Dict[int, float] = {}
        for x in by_rec.get(i, []):
            k = rules[x["criterion_index"]]  # criteria ask whether a rule was violated: YES = violation
            per_rule[k] = max(per_rule.get(k, 0.0), x["p_yes_binary"])
        traces.append({"fail": meta.get("trace_label") == "FAIL", "per_rule": per_rule,
                       "violated": set(meta.get("rules_violated") or []),
                       "score": max(per_rule.values(), default=0.0)})
    return traces


def trace_eval(traces, threshold, n_boot):
    score = np.array([t["score"] for t in traces])
    gold = np.array(["YES" if t["fail"] else "NO" for t in traces], dtype=object)
    m = M.binary_metrics(M.threshold_preds(score, threshold), gold)
    lo, hi = M.bootstrap_ci(lambda idx: M.binary_metrics(M.threshold_preds(score[idx], threshold),
                                                          gold[idx])["f1"], len(traces), n_boot=n_boot)
    out = {"metric": "f1", "value": m["f1"], "ci95": [lo, hi], "threshold": threshold, **m}
    fails = [t for t in traces if t["fail"] and t["violated"]]
    for k in (1, 3):
        hit = [bool(set(sorted(t["per_rule"], key=t["per_rule"].get, reverse=True)[:k]) & t["violated"])
               for t in fails]
        out[f"top{k}_localization"] = float(np.mean(hit)) if hit else None
    return out


def yes_no_auc(scored, protocol):
    rows = [x for x in scored if x["label"] in ("YES", "NO")]
    key = "p_yes" if protocol == "ternary" else "p_yes_binary"
    return M.roc_auc([x["label"] == "YES" for x in rows], [x[key] for x in rows]), len(rows)


def tune(scored_or_traces, protocol, metric):
    if protocol == "binary":
        return M.tune_threshold([x["p_yes_binary"] for x in scored_or_traces],
                                [x["label"] for x in scored_or_traces], metric)
    if protocol == "trace":
        return M.tune_threshold([t["score"] for t in scored_or_traces],
                                ["YES" if t["fail"] else "NO" for t in scored_or_traces], "f1",
                                grid=M.TRACE_THRESHOLD_GRID)
    raise ValueError(protocol)


def dev_value(scored, records, protocol, metric):
    """(selection value, threshold) of one checkpoint on the dev set."""
    if protocol == "ternary":
        preds = [x["verdict"] for x in scored]
        return M.ternary_metrics(preds, [x["label"] for x in scored])[metric], None
    if protocol == "trace":
        t, v = tune(trace_scores(records, scored), protocol, metric)
        return v, t
    t, v = tune(scored, protocol, metric)
    return v, t


# ----------------------------------------------------------------------------- main
def expand_models(specs: List[str]) -> List[str]:
    out = []
    for s in specs:
        hits = sorted(glob.glob(s), key=lambda p: (len(p), p)) if any(ch in s for ch in "*?[") else [s]
        if not hits:
            raise SystemExit(f"no checkpoints match {s}")
        out.extend(hits)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True,
                    help="Hub id or local dir; several (or a glob) to dev-select a checkpoint.")
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--test", default=None, help="Test file (default: data/<benchmark>/test.jsonl)")
    ap.add_argument("--dev", default=None, help="Dev file for checkpoint and threshold selection")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold when no --dev is given")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=64, help="Criteria per pass for multi-criterion records")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate the first N test records only")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cv-folds", action="store_true",
                    help="Select the checkpoint by cross-validation over meta['fold'] of the test "
                         "records (INSURE-Dial protocol): for each fold, pick the checkpoint with the "
                         "best primary metric on the other folds and predict the held-out fold.")
    ap.add_argument("--save-predictions", default=None, help="Write per-criterion predictions (JSONL)")
    ap.add_argument("--out", default=None, help="Write the JSON summary here")
    args = ap.parse_args()

    spec = BENCHMARKS[args.benchmark]
    protocol, metric = spec["protocol"], spec["metric"]
    select_metric = spec.get("select_metric", metric)
    max_length = args.max_length or spec.get("max_length", 4096)
    test_path = args.test or os.path.join(REPO_ROOT, "data", args.benchmark, "test.jsonl")
    test = read_jsonl(test_path)[: args.limit]
    dev = read_jsonl(args.dev) if args.dev else None
    models = expand_models(args.model)

    def run(model_id, records):
        vfm = VFM.from_pretrained(model_id)
        try:
            return score_records(vfm, records, args.batch_size, max_length, args.chunk_size)
        finally:
            del vfm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    selection = []
    chosen, threshold = models[0], args.threshold
    if args.cv_folds:
        if protocol != "ternary":
            raise SystemExit("--cv-folds is implemented for ternary benchmarks")
        per_model = [run(m, test) for m in models]
        folds = sorted({r.get("meta", {}).get("fold") for r in test}, key=str)
        if len(folds) < 2 or None in folds:
            raise SystemExit("--cv-folds needs records with meta.fold and at least two folds")
        rec_fold = [test[x["record"]].get("meta", {}).get("fold") for x in per_model[0]]
        labels = [x["label"] for x in per_model[0]]
        scored = [None] * len(labels)
        for f in folds:
            held = [i for i, rf in enumerate(rec_fold) if rf == f]
            rest = [i for i, rf in enumerate(rec_fold) if rf != f]
            vals = [M.ternary_metrics([sc[i]["verdict"] for i in rest], [labels[i] for i in rest])[metric]
                    for sc in per_model]
            best = int(np.argmax(vals))  # first maximum wins
            selection.append({"fold": f, "model": models[best], "other_folds_value": vals[best]})
            print(f"[cv] fold {f}: {models[best]} ({metric} on the other folds {vals[best]:.4f})")
            for i in held:
                scored[i] = per_model[best][i]
        chosen = models
    else:
        if dev is not None:
            for m in models:
                v, t = dev_value(run(m, dev), dev, protocol, select_metric)
                selection.append({"model": m, "dev_value": v, "threshold": t})
                print(f"[dev] {m}: {select_metric}={v:.4f}" + (f" @ t={t:.2f}" if t is not None else ""))
            best = max(selection, key=lambda r: r["dev_value"])  # first maximum wins
            chosen = best["model"]
            threshold = best["threshold"] if best["threshold"] is not None else args.threshold
        elif len(models) > 1:
            raise SystemExit("Several models need --dev (or --cv-folds) for selection")
        scored = run(chosen, test)

    if protocol == "binary":
        result = binary_eval(scored, metric, threshold, args.n_boot)
    elif protocol == "ternary":
        clusters = [test[x["record"]]["meta"][spec["cluster"]] for x in scored] if "cluster" in spec else None
        result = ternary_eval(scored, metric, args.n_boot, clusters)
    else:
        result = trace_eval(trace_scores(test, scored), threshold, args.n_boot)
    if protocol != "trace":
        auc, n_auc = yes_no_auc(scored, protocol)
        result["roc_auc"], result["roc_auc_n"] = auc, n_auc

    summary = {"benchmark": args.benchmark,
               "protocol": protocol + (" (cross-validated checkpoint selection)" if args.cv_folds else ""),
               "model": chosen, "test_file": os.path.relpath(test_path, os.getcwd()),
               "n_records": len(test), "n_criteria": len(scored), "max_length": max_length,
               "selection": selection, "result": result}
    print(json.dumps({k: v for k, v in summary.items() if k != "selection"}, indent=1, default=str))
    for path in (args.out, args.save_predictions):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1, default=str)
    if args.save_predictions:
        with open(args.save_predictions, "w") as f:
            for x in scored:
                f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
