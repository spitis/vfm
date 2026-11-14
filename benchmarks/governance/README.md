# Synthetic governance benchmark

Synthetic compliance interactions for 52 atomic criteria drawn from five regulations
(EU AI Act Art. 50, GDPR Art. 13, SEC Regulation Best Interest, HIPAA Privacy Rule
Sec. 164.520, PCI-DSS v4.0). Each record is one (context, target, criterion) triple with
a YES / NO / N/A label, in the VFM record format (`vfm.data`). Released under CC BY 4.0.

| File | Records | YES | NO | N/A | Content |
|---|---|---|---|---|---|
| `train.jsonl` | 2,247 | 801 | 781 | 665 | 130 interactions, 32 criteria from EU AI Act, GDPR, SEC Reg BI (finetuning) |
| `dev.jsonl` | 302 | 94 | 111 | 97 | 20 interactions, same 32 criteria (checkpoint selection) |
| `dev300_s1.jsonl`, `dev300_s2.jsonl`, `dev300_s3.jsonl` | 299 each | 93 | 110 | 96 | stratified subsets of `dev.jsonl`, one per seed |
| `test.jsonl` | 920 | 276 | 340 | 304 | 60 interactions, same 32 criteria |
| `test_transfer.jsonl` | 951 | 281 | 359 | 311 | 40 interactions over all 52 criteria, including the HIPAA and PCI-DSS criteria never seen in training |

Other files: `criteria.json` (the 52 criteria with regulation, article, difficulty tag,
and applicability domains; `text` is the wording used in the records) and
`regulations.json` (short regulation summaries and the paragraph excerpts the criteria
were written from, with source URLs; PCI-DSS excerpts are omitted for copyright, so
follow its URL).

Record fields: `id`, `context` (chat messages), `target` (the assistant response being
checked), `criteria` and `labels` (one each), and `meta` with `example_id`,
`interaction_id`, `criterion_id`, `regulation`, `domain`, `difficulty`, and `scenario`.
Several records share one interaction (the same context and target scored against
different criteria). N/A marks a criterion that does not apply to the interaction or
whose precondition does not hold. `dev.jsonl` words the criteria as "the response" where
the other files say "the text".

## How the labels were made

Claude Sonnet 4.6 drafted the interactions together with their per-criterion compliance
flags, and the labels are derived from those flags. Claude Opus 4.6 and Gemini 3.1 Pro
audited the data. A three-annotator check of 75 items found 87--92% agreement with the
labels on the binary YES vs. not-YES projection. The labels are model-generated and can
contain errors.

The interactions are fictional; company names, addresses, and contact details are invented.

## Use

`python scripts/prepare/prepare_governance.py` validates these files and copies them to
`data/governance/`.
