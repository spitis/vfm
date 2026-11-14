"""Evaluation metrics used in the paper, with percentile-bootstrap confidence intervals.

Binary benchmarks score P(YES) renormalized over YES and NO, predict YES iff the score
is >= a threshold, and report balanced accuracy or F1 (positive class YES). Thresholds
are tuned on a dev set over the grid 0.05, 0.06, ..., 0.95 (first maximum wins); DynaBench
uses the grid 0.05, 0.10, ..., 0.70.
Ternary benchmarks take the argmax over YES / NO / N/A and report accuracy or macro-F1
over all three classes.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .configuration_vfm import LABEL_NAMES

YES, NO, NA = LABEL_NAMES
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
TRACE_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def binary_metrics(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    """Confusion-based metrics with positive class YES; any non-YES label counts as negative."""
    p = np.asarray([x == YES for x in preds])
    y = np.asarray([x == YES for x in labels])
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    fp, tn = int((p & ~y).sum()), int((~p & ~y).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    n = tp + tn + fp + fn
    return {"balanced_accuracy": (tpr + tnr) / 2, "f1": f1, "precision": prec, "recall": tpr,
            "accuracy": (tp + tn) / n if n else 0.0, "tpr": tpr, "tnr": tnr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}


def threshold_preds(scores: Sequence[float], t: float) -> List[str]:
    return [YES if s >= t else NO for s in scores]


def tune_threshold(scores: Sequence[float], labels: Sequence[str], metric: str,
                   grid: Sequence[float] = THRESHOLD_GRID) -> Tuple[float, float]:
    """Threshold on `grid` maximizing `metric` (first maximum wins)."""
    best_t, best_v = 0.5, -1.0
    for t in grid:
        v = binary_metrics(threshold_preds(scores, t), labels)[metric]
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t, best_v


def ternary_metrics(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    """Accuracy and macro-F1 over YES / NO / N/A (all three classes, present or not)."""
    per_class = {}
    for c in (YES, NO, NA):
        tp = sum(p == c and l == c for p, l in zip(preds, labels))
        fp = sum(p == c and l != c for p, l in zip(preds, labels))
        fn = sum(p != c and l == c for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {"precision": prec, "recall": rec,
                        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    n = len(labels)
    return {"accuracy": sum(p == l for p, l in zip(preds, labels)) / n if n else 0.0,
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class, "n": n}


def roc_auc(labels01: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels01, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(metric_fn: Callable[[np.ndarray], float], n: int, n_boot: int = 10000,
                 seed: int = 42, alpha: float = 0.05,
                 clusters: Optional[Sequence] = None) -> Tuple[float, float]:
    """Percentile bootstrap CI of `metric_fn(indices)` over `n` resampled units.

    With `clusters` (one cluster id per unit), whole clusters are resampled and the indices
    of their units concatenated (cluster bootstrap).
    """
    rng = np.random.RandomState(seed)
    groups = None
    if clusters is not None:
        c = np.asarray(clusters)
        groups = [np.flatnonzero(c == k) for k in sorted(set(c.tolist()))]
    vals = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.randint(0, n, n)
        else:
            idx = np.concatenate([groups[k] for k in rng.randint(0, len(groups), len(groups))])
        vals.append(metric_fn(idx))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and population standard deviation (ddof=0), as reported over seeds in the paper."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=0))
