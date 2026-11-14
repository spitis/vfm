# DynaBench rule translations

DynaBench (Hoover et al., 2025; https://huggingface.co/datasets/tomg-group-umd/DynaBench, MIT)
pairs multi-turn dialogues with policies of free-text rules. The dialogues are downloaded by
`scripts/prepare/prepare_dynabench.py` and are not redistributed here.

Files:

- `rule_translations.json.gz`: our translation of 23,483 rules (keyed by the rule text as the
  prepare script parses it) into atomic questions. `sub_questions` lists the questions of the
  rule (1.5 per rule on the test set) and `compound` marks rules that were split into several
  branches. A question is worded so that YES means the rule was violated, for example
  "Never use emojis." becomes "Did the agent use emojis in any of its responses?". The
  translation was done once by an LLM from the rule text alone, without seeing dialogues or labels.
  The file covers the rules of the test, train, and dev splits described below.
- `dev_policy_ids.json`: `base_id` of the 502 held-out DynaBenchTrain policies used as the dev set.

Splits built by the script (one record per dialogue, with all of its translated criteria):

| Split | Traces | Criteria | Labels | Source |
|---|---|---|---|---|
| test | 543 (267 FAIL) | 11,486 | YES 1,859 / NO 9,627 | DynaBench test, all rules |
| train | 13,997 | 75,031 | YES 12,088 / NO 54,079 / N/A 8,864 | DynaBenchTrain, first 14,000 policies with equal PASS/FAIL share (seed 0), violated rules plus compliant rules up to 10 per policy |
| dev | 502 | 9,227 | YES 896 / NO 8,331 | 502 other DynaBenchTrain policies, rules marked ambiguous dropped |

A criterion takes its rule's label: YES if the rule was violated in the dialogue, NO if followed,
N/A (train only) if the source marks the rule ambiguous. A dialogue is FAIL if any criterion is YES.
`meta.rule_index` and `meta.sub_idx` map each criterion back to its rule and question index (0-based); every criterion is worded so that YES means a violation; `meta.rules_violated` lists the gold violated rule indices (0-based).
The `DynaBenchSafetyMix` config of the dataset (which includes BeaverTails and other external
safety data) is not used.

## License

`rule_translations.json.gz` contains the original DynaBench rule texts (as its keys), which
are redistributed under DynaBench's MIT license:

    MIT License

    Copyright (c) 2025 Monte Hoover

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The questions and the dev policy ids are ours and are released under the repository's license.
