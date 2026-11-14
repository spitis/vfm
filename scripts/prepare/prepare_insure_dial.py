"""INSURE-Dial: insurance-benefit verification calls, converted to VFM records.

Source:   https://huggingface.co/datasets/dataframer/insure-dial  (Kulkarni et al., EACL 2026)
License:  the dataset card declares no license; the authors' code repository is MIT.
          Check the terms on the dataset page before use.
Download-only, not redistributed: the call transcripts are never stored in this
repository. This script downloads them and builds the records locally. Only record ids
and labels (``benchmarks/manifests/insure_dial.json``) are shipped.

Each call is scored against 6 or 9 atomic criteria (greeting, coverage-check, patient
details, per-drug formulary / restrictions / copay, agent name, reference number). The last
turn of the call is the target and the preceding turns are the context. In the dataset the
calling AI agent has role ``assistant``.

Splits written to ``<out_dir>/``:
    test    417 criteria over the 50 real calls (47.5% YES, 19.0% NO, 33.6% N/A)
    train  2,000 criteria over synthetic calls, rebalanced (785 YES, 430 NO, 785 N/A)
    dev      903 criteria over 100 held-out synthetic calls (natural label mix)

The paper's 2,000 training records were sampled at random and are recovered exactly from the
manifest; with ``--no-manifest`` the script draws a fresh sample of the same composition
(about 40% overlap with the paper's).

Checkpoints in the paper are selected by 5-fold cross-validation over the real calls
(split by call). ``meta.fold`` of each test record holds its fold: the calls are sorted by
id and assigned round-robin, so ``call_NNN`` (the NNN-th row of the ``real`` split) is in
fold NNN mod 5.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Dict, List

from _common import add_common_args, load_manifest, write_split

from vfm.data import make_record

HF_REPO = "dataframer/insure-dial"
MAX_CHARS = 24000
NFOLD = 5
DEV_RATIO = 0.1
TARGET_SIZE = 2000


def turn_range_str(tr) -> str:
    if tr is None:
        return ""
    if isinstance(tr, (list, tuple)) and len(tr) == 2:
        return f"{tr[0]}-{tr[1]}"
    return str(tr)


def yes_no(b) -> str:
    return "YES" if bool(b) else "NO"


def call_criteria(ann: dict) -> List[dict]:
    """The atomic criteria of one call: {phase, kind, criterion, label, turn_range}."""
    out = []

    def add(phase, kind, criterion, label, tr):
        out.append({"phase": phase, "kind": kind, "criterion": criterion, "label": label,
                    "turn_range": turn_range_str(tr)})

    g = ann.get("greeting") or {}
    if g:
        add("greeting", "phase_present", "Did the assistant provide a greeting at the start of the call?",
            yes_no(g.get("present", False)), g.get("turn_range"))

    cc = ann.get("coverage_check") or {}
    if cc:
        required = cc.get("required", False)
        add("coverage_check", "assistant_asked",
            "Did the assistant ask the user to confirm their current insurance plan or coverage status?",
            "N/A" if not required else yes_no(cc.get("assistant_asked_status", False)), cc.get("turn_range"))
        add("coverage_check", "user_answered",
            "Did the user provide an answer when the assistant asked about their coverage status?",
            "N/A" if not required else yes_no(cc.get("user_answered_status", False)), cc.get("turn_range"))

    pi = ann.get("patient_info") or {}
    if pi:
        required = pi.get("required", False)
        add("patient_info", "assistant_provided_details",
            "Did the assistant provide details about the patient's record (such as confirming the "
            "patient was found in the system or confirming insurance/eligibility details)?",
            "N/A" if not required else yes_no(pi.get("assistant_provided_details", False)),
            pi.get("turn_range"))

    for drug in ann.get("drugs") or []:
        if not isinstance(drug, dict):
            continue
        name = (drug.get("name") or "").strip()
        if not name or name.upper() in {"NA", "N/A"}:
            continue
        for slot, question in [
            ("formulary", "Did the assistant ask whether {name} is on the patient's formulary?"),
            ("restrictions", "Did the assistant ask whether there are any prior-authorization or other "
                             "restrictions on {name}?"),
            ("copay", "Did the assistant ask about the copay or out-of-pocket cost for {name}?"),
        ]:
            slot_data = drug.get(slot) or {}
            label = "N/A" if not slot_data.get("required", False) else yes_no(
                slot_data.get("assistant_asked", False))
            add(f"drugs[{drug.get('index', '?')}].{slot}", "assistant_asked", question.format(name=name),
                label, slot_data.get("turn_range"))

    ai = ann.get("agent_interaction") or {}
    if ai:
        add("agent_interaction", "asked_user_name", "Did the assistant ask the user for their name?",
            yes_no(ai.get("assistant_asked_user_name", False)), ai.get("turn_range"))
        add("agent_interaction", "asked_reference",
            "Did the assistant ask the user for a reference or case number?",
            yes_no(ai.get("assistant_asked_reference", False)), ai.get("turn_range"))
    return out


def split_context_target(chat: List[dict]):
    """Last turn is the target; drop the oldest context turns beyond MAX_CHARS."""
    if not chat:
        return [], "", None, False
    target = str(chat[-1].get("content", ""))
    last_role = chat[-1].get("role", "user")
    context = [{"role": t.get("role", "user"), "content": str(t.get("content", ""))} for t in chat[:-1]]

    def total(ctx, tgt):
        return sum(len(c["content"]) + len(c["role"]) + 4 for c in ctx) + len(tgt)

    truncated = False
    while context and total(context, target) > MAX_CHARS:
        context.pop(0)
        truncated = True
    return context, target, last_role, truncated


def call_records(call_id: str, script: list, ann: dict) -> List[dict]:
    chat = [{"role": t.get("role", "user"), "content": str(t.get("content", ""))}
            for t in script if isinstance(t, dict)]
    ctx, tgt, last_role, truncated = split_context_target(chat)
    recs = []
    for c in call_criteria(ann):
        rec = make_record(ctx, tgt, c["criterion"], c["label"], {
            "call_id": call_id, "phase": c["phase"], "criterion_kind": c["kind"],
            "turn_range": c["turn_range"], "last_turn_role": last_role,
            "context_truncated": truncated})
        recs.append(rec)
    return recs


def stratified_sample(records, target_n, seed):
    """Phase-stratified subsample, proportional to each phase's frequency."""
    if target_n >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_phase = defaultdict(list)
    for r in records:
        by_phase[r["meta"]["phase"]].append(r)
    total = len(records)
    sampled = []
    for phase in sorted(by_phase):
        recs = by_phase[phase]
        keep = min(max(1, int(round(target_n * len(recs) / total))), len(recs))
        rng.shuffle(recs)
        sampled.extend(recs[:keep])
    if len(sampled) > target_n:
        rng.shuffle(sampled)
        sampled = sampled[:target_n]
    elif len(sampled) < target_n:
        chosen = {id(r) for r in sampled}
        remaining = [r for r in records if id(r) not in chosen]
        rng.shuffle(remaining)
        sampled.extend(remaining[:target_n - len(sampled)])
    return sampled


def build_train(train_pool: List[dict], seed: int) -> List[dict]:
    """Keep every NO record; fill the rest equally with phase-stratified YES and N/A."""
    by_label = defaultdict(list)
    for r in train_pool:
        by_label[r["labels"][0]].append(r)
    no = list(by_label["NO"])
    yes_target = (TARGET_SIZE - len(no)) // 2
    na_target = TARGET_SIZE - len(no) - yes_target
    out = (no + stratified_sample(by_label["YES"], yes_target, seed)
           + stratified_sample(by_label["N/A"], na_target, seed + 1))
    random.Random(seed + 2).shuffle(out)
    return out


def call_level_split(records: List[dict], seed: int):
    """Hold out 10% of the synthetic calls (shuffled by sorted call id) as dev."""
    rng = random.Random(seed)
    by_call = defaultdict(list)
    for r in records:
        by_call[r["meta"]["call_id"]].append(r)
    call_ids = sorted(by_call)
    rng.shuffle(call_ids)
    n_dev = max(1, int(round(len(call_ids) * DEV_RATIO)))
    dev_calls, train_calls = call_ids[:n_dev], sorted(call_ids[n_dev:])
    train = [r for c in train_calls for r in by_call[c]]
    dev = [r for c in dev_calls for r in by_call[c]]
    return train, dev


def select(pool: List[dict], manifest: dict, split: str, strict: bool) -> List[dict]:
    """Keep the pool records listed for `split`, repeating a record as often as the manifest says.

    A few (context, target, criterion) triples occur more than once (a drug named twice in a
    call); ``manifest["counts"][split][id]`` gives the label multiset of such ids.
    """
    wanted = manifest["splits"][split]
    counts = manifest.get("counts", {}).get(split, {})
    left: Dict[str, Counter] = {i: Counter(counts[i]) if i in counts else Counter({lab: 1})
                                for i, lab in wanted.items()}
    expected = {i: set(c) for i, c in left.items()}
    out, mismatched = [], 0
    for r in pool:
        rid, label = r["id"], r["labels"][0]
        if rid not in wanted:
            continue
        if left[rid][label] <= 0:
            mismatched += label not in expected[rid]
            continue
        left[rid][label] -= 1
        out.append(r)
    total = sum(sum(counts[i].values()) if i in counts else 1 for i in wanted)
    missing = total - len(out)
    msg = (f"[insure_dial/{split}] matched {len(out)}/{total} paper records"
           f" ({missing} missing, {mismatched} label mismatches)")
    print(msg)
    if strict and (missing or mismatched):
        raise SystemExit(msg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, "insure_dial")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(HF_REPO)

    real: List[dict] = []
    for i, row in enumerate(ds["real"]):
        real.extend(call_records(f"call_{i:03d}", row["script"], row["annotation"]))
    calls = sorted({r["meta"]["call_id"] for r in real})
    fold = {c: i % NFOLD for i, c in enumerate(calls)}
    for r in real:
        r["meta"]["fold"] = fold[r["meta"]["call_id"]]

    synth: List[dict] = []
    for i, row in enumerate(ds["synthetic"]):
        synth.extend(call_records(f"synth_{i:04d}", row["script"], row["annotation"]))
    train_pool, dev = call_level_split(synth, args.seed)
    train = build_train(train_pool, args.seed)

    # The paper's training set was drawn from the pool below; its exact records are recovered
    # from the manifest. Without the manifest, the sampling above gives a fresh ~2K set.
    splits: Dict[str, List[dict]] = {"test": real, "train": train, "dev": dev}
    pools: Dict[str, List[dict]] = {"test": real, "train": train_pool, "dev": dev}
    manifest = None if args.no_manifest else load_manifest("insure_dial")
    for name, records in splits.items():
        if manifest and name in manifest["splits"]:
            records = select(pools[name], manifest, name, args.strict)
        write_split(args.out_dir, name, records)
    print("test labels:", dict(Counter(r["labels"][0] for r in real)), "calls:", len(calls))


if __name__ == "__main__":
    main()
