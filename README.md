# VFM: Runtime Verification of Multiple Natural Language Criteria

Code and models for *Runtime Verification of Multiple Natural Language Criteria for Agent
Governance* (NeurIPS 2026).

The VFM checks an agent's output against many natural-language criteria at once. Given a
context (system prompt, conversation, tool calls), a target text (for example the agent's
response), and N criteria phrased as yes/no questions, it returns an independent verdict for
each criterion: **YES** (satisfied), **NO** (violated), or **N/A** (inapplicable, or not
determinable from the trace), with probabilities.

All N criteria are scored in one forward pass. The shared prefix is encoded once; each
criterion is a separate branch of a tree attention mask that sees the full prefix but not the
other criteria, and every branch starts at the same position. Verdicts are therefore
independent of how many criteria are scored and of their order. A linear head on a Qwen3
backbone reads the last token of each branch. The models are pretrained on CriteriaBank, a
corpus of 394K (context, target, criterion, verdict) tuples with 321K distinct criteria.

| Resource | Link |
|---|---|
| VFM-0.6B (Qwen3-0.6B backbone) | https://huggingface.co/spitis/VFM-0.6B |
| VFM-4B (Qwen3-4B backbone) | https://huggingface.co/spitis/VFM-4B |
| CriteriaBank pretraining data | https://huggingface.co/datasets/spitis/CriteriaBank |

## Install

```bash
pip install git+https://github.com/spitis/vfm            # inference
pip install "vfm[train,benchmarks] @ git+https://github.com/spitis/vfm"   # + training and benchmark preparation
```

Or from a clone: `pip install -e ".[train,benchmarks,dev]"`. Tested with Python 3.10,
PyTorch 2.10, and Transformers 4.57.1, 4.57.6 and 5.20.2. The scripts below are run from the
root of a clone.

## Quickstart

```python
from vfm import VFM

vfm = VFM.from_pretrained("spitis/VFM-4B")   # bf16, on GPU if available
verdicts = vfm.verify(
    context=[{"role": "user", "content": "I have $50,000 to invest. Any advice?"}],
    target="Put it all in one stock; it is guaranteed to double within a year.",
    criteria=[
        "Does the response guarantee investment returns?",
        "Does the response ask about the user's risk tolerance?",
        "If the response cites a regulation, does it name the regulation?",
    ],
)
for v in verdicts:
    print(v["verdict"], f'{v["p_yes"]:.3f} {v["p_no"]:.3f} {v["p_na"]:.3f}', v["criterion"])
```

The same model loads without this package:

```python
import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("spitis/VFM-4B", trust_remote_code=True, dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("spitis/VFM-4B", trust_remote_code=True)
verdicts = model.verify(
    tokenizer,
    context=[{"role": "user", "content": "I have $50,000 to invest. Any advice?"}],
    target="Put it all in one stock; it is guaranteed to double within a year.",
    criteria=["Does the response guarantee investment returns?"],
)
```

Usage notes:
- `context` is a string (one user turn) or a list of `{"role", "content"}` messages. The target
  is appended as the next turn. Prefixes longer than 4,096 tokens are truncated from the left
  (`max_length=` to change).
- `verify(context, target, criteria)` scores many criteria against one trace, reusing the
  prefix's key-value cache across chunks of `chunk_size` criteria (default 64), so memory stays
  flat in N. `verify_batch(rows)` scores many `{context, target, criterion}` rows with padding.
- Each result has `verdict`, `p_yes`, `p_no`, `p_na`, `p_yes_binary` (P(YES) renormalized over
  YES and NO, the score used for binary decisions in the paper), and the raw `logits`.
- Criteria work best as short, positively phrased, atomic yes/no questions about something
  checkable from the trace. Each criterion has a 64-token block that includes about 12
  chat-template tokens; longer criteria are truncated from the left, as in training.
- The tree mask is a dense 4D attention mask, so the model runs with PyTorch SDPA (or eager),
  not flash attention.

## Repository layout

```
src/vfm/                 model, config, collator, inference API, metrics, training utilities
scripts/prepare/         download a benchmark from its source and convert it to VFM records
scripts/evaluate.py      evaluation with the paper's metrics, dev selection and bootstrap CIs
scripts/finetune.py      CriteriaBank pretraining and downstream finetuning
scripts/configs/         the paper's training recipes
benchmarks/              shipped benchmark files, record-id manifests and DynaBench rule translations
tests/                   CPU tests with a tiny random model
```

## Data format

Every benchmark is converted to JSONL records of the form

```json
{"id": "...", "context": [{"role": "user", "content": "..."}], "target": "...",
 "criteria": ["Does the response ...?"], "labels": ["YES"], "meta": {}}
```

with labels exactly YES, NO or N/A (see `src/vfm/data.py`). Most records hold one criterion;
DynaBench records hold all translated criteria of one dialogue.

## Benchmarks

Each prepare script downloads the benchmark from its original source, applies the paper's
conversion, and writes `data/<benchmark>/<split>.jsonl`. `benchmarks/manifests/<benchmark>.json`
lists the ids and labels of the exact records used in the paper (ids are hashes of
context, target and criterion; no text), and the scripts select those records by default and
report coverage (`--no-manifest` rebuilds the splits with the original sampling code, and
`--strict` fails on any missing record). We checked that every script reproduces 100% of the
paper's records. The 300-record dev subsets `dev300_s{1,2,3}` used for finetuning come from
the manifests, so `--no-manifest` does not write them.

```bash
python scripts/prepare/prepare_prm800k.py        # likewise for the other benchmarks
```

| Benchmark | Source | License | In this repo |
|---|---|---|---|
| Synthetic governance | ours | CC BY 4.0 | full data (`benchmarks/governance/`) |
| INSURE-Dial | `dataframer/insure-dial` | none declared on the dataset card | download-only |
| DynaBench | `tomg-group-umd/DynaBench` | MIT | dialogues download-only; the rule texts and our translations of them shipped (`benchmarks/dynabench/`) |
| BeaverTails | `PKU-Alignment/BeaverTails` | CC BY-NC 4.0 | download-only |
| IFEval-like | `argilla/ifeval-like-data` | other (Qwen license) | download-only; constraint checkers vendored |
| PRM800K | `Birchlabs/openai-prm800k-stepwise-critic` | MIT | download-only |
| LLM-AggreFact | `lytang/LLM-AggreFact` | CC BY-ND 4.0 | download-only |
| AggreFact finetuning | `facebook/anli`, `tals/vitaminc`, `copenlu/fever_gold_evidence` | CC BY-NC 4.0; CC BY-SA 3.0; CC BY-SA 3.0 | download-only |
| ContractNLI | official Stanford release | CC BY 4.0 | download-only |
| InFoBench (zero-shot only) | `ContextualAI/InfoBenchExpertSplit` | MIT (InFoBench) | download-only |
| CriteriaBank (pretraining) | `spitis/CriteriaBank` | see dataset card | `scripts/prepare/prepare_criteriabank.py` |

The released models were trained only on CriteriaBank; the benchmark licenses above apply to
the benchmark data, not to the models.

## Evaluate

```bash
# A released model, zero-shot (ROC-AUC as in the Appendix table "Zero-shot ROC-AUC")
python scripts/prepare/prepare_infobench.py
python scripts/evaluate.py --model spitis/VFM-0.6B --benchmark infobench

# Synthetic governance (shipped in benchmarks/governance/; the prepare step copies and checks it)
python scripts/prepare/prepare_governance.py
python scripts/evaluate.py --model spitis/VFM-0.6B --benchmark governance

# A finetuned run (train it first, see Train below): pick the checkpoint and decision
# threshold on the dev subset, score the test set
python scripts/evaluate.py --benchmark prm800k --model "runs/prm800k_s1/checkpoint-*" \
    --dev data/prm800k/dev300_s1.jsonl --out results/prm800k_s1.json

# INSURE-Dial (after prepare_insure_dial.py and finetune_insure_dial.yaml): checkpoint
# chosen by 5-fold cross-validation over the real calls
python scripts/evaluate.py --benchmark insure_dial --model "runs/insure_s1/checkpoint-*" --cv-folds
```

Metrics follow the paper: F1 (BeaverTails), balanced accuracy (IFEval, PRM800K,
LLM-AggreFact), macro-F1 over YES/NO/N/A (ContractNLI), accuracy (governance, INSURE-Dial),
and trace-level FAIL F1 (DynaBench; a dialogue fails if any criterion's violation probability
reaches the threshold, with top-1 and top-3 rule localization). Binary benchmarks threshold
P(YES) renormalized over YES and NO, with the threshold tuned on the dev set over
0.05, 0.06, ..., 0.95 (DynaBench: 0.05, 0.10, ..., 0.70). Checkpoints are selected by the
reported metric on the dev subset, except governance, which selects by dev macro-F1. Binary
and ternary results also report threshold-free ROC-AUC. The primary metric carries a 95%
percentile-bootstrap interval (10,000 resamples; INSURE-Dial resamples whole calls). For
finetuned models the paper reports the mean and population standard deviation over three
seeds: seeds 1-3, each with its own dev subset (`dev300_s{1,2,3}`), for the benchmarks that
have them, and seeds 42-44 for INSURE-Dial (cross-validation) and DynaBench (the held-out dev
policies); `vfm.metrics.mean_std` computes this from the per-seed results.

Scores computed here agree with the paper's up to bf16 rounding: re-running zero-shot
evaluation of VFM-0.6B on an RTX 3090 gave ROC-AUC 0.888 on InFoBench (paper 0.887), 0.723 on
BeaverTails (paper 0.722), 0.880 on governance (paper 0.880) and 0.712 on INSURE-Dial (paper
0.713).

## Train

`scripts/finetune.py` runs both stages; `scripts/configs/` holds the paper's recipes. All
training updates all parameters (no LoRA) with AdamW, with weights and optimizer states in
bf16 as in the paper's runs, weight decay 0.01 on the head weight only (none on the backbone
or the head bias), gradient clipping 1.0, and an effective batch of 128 (per-device batch ×
accumulation × GPUs).

Downstream finetuning (head LR 1e-4, backbone LR 5e-5, cosine schedule, 10% warmup, up to 30
epochs, early stopping on a 300-example dev subset with patience 5):

```bash
for s in 1 2 3; do
  python scripts/finetune.py --config scripts/configs/finetune.yaml --model spitis/VFM-4B \
      --train data/prm800k/train.jsonl --dev data/prm800k/dev300_s$s.jsonl --seed $s \
      --output-dir runs/prm800k_s$s
done
```

Use `finetune_contractnli.yaml` (8,192-token prefixes), `finetune_insure_dial.yaml` (20 epochs,
cross-validated selection) and `finetune_dynabench.yaml` (400 steps) for those benchmarks. To
reproduce the Direct-FT control, replace `--model spitis/VFM-4B` with
`--init-from-base Qwen/Qwen3-4B`. This mean-centers the head bias over the first 512
downstream training rows; the paper's Direct-FT start calibrated it on CriteriaBank rows
instead.

CriteriaBank pretraining (one epoch; head LR 5e-4, backbone LR 5e-5, 3% warmup; KL penalty
toward the frozen base model with λ = 1; head initialized from the unembedding rows of
" Yes", "No", " Unknown" with mean-centered bias):

```bash
python scripts/prepare/prepare_criteriabank.py
torchrun --nproc_per_node 4 scripts/finetune.py --config scripts/configs/pretrain_kl_4b.yaml
```

`pretrain_nokl_*.yaml` trains the "VFM (w/o KL)" ablation. The released checkpoints are the
step-2,000 pretraining checkpoints (`checkpoint-2000`) for both sizes. Training was scheduled for
3,075 steps (one epoch's worth of samples at batch 128). `prepare_criteriabank.py` exports the 316,974
rows that CriteriaBank ships with text, and adds the keyed rows if you rebuild them with the
dataset's `rehydrate.py` (`--keyed`).

## Tests

```bash
pytest tests
```

The tests run on CPU with a tiny random backbone and check shapes, independence of criteria,
order invariance, padding invariance, the cached-prefix path, and save/load round trips
(including loading with `trust_remote_code`).

## License

Code and model weights: Apache-2.0 (see `LICENSE` and `NOTICE`). The models are finetuned
from Qwen3 (Apache-2.0). Benchmark data keeps its original license (table above).

## Citation

```bibtex
@inproceedings{pitis2026vfm,
  title     = {Runtime Verification of Multiple Natural Language Criteria for Agent Governance},
  author    = {Pitis, Silviu and Alamdari, Parand A. and Tang, Jessica and Klassen, Toryn Q. and McIlraith, Sheila A.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
