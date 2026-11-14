"""CPU tests with a tiny random backbone: shapes, tree-attention invariances, the
prefix-cached path, save/load round trips, and metric helpers."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast  # noqa: E402

from vfm import VFMCollator, VFMConfig, VFMForCriteriaVerification  # noqa: E402
from vfm import metrics as M  # noqa: E402
from vfm.data import make_record, record_id  # noqa: E402

T = 64  # criterion block length (as in the released models)

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["Does the response mention the weather? It is sunny today.",
              "Is the answer correct? The capital of France is Paris.",
              "user assistant system Write a haiku about rain."] * 20
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                   eos_token="<|im_end|>")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_model(vocab_size, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    bc = AutoConfig.for_model("qwen3", vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=8, max_position_embeddings=512, tie_word_embeddings=True)
    cfg = VFMConfig(backbone_config=bc, max_criterion_length=T, max_prefix_length=128)
    return VFMForCriteriaVerification(cfg).to(dtype).eval()


CONTEXT = [{"role": "user", "content": "Is the capital of France Paris?"}]
TARGET = "The capital of France is Paris."
CRITERIA = ["Is the answer correct?", "Does the response mention the weather?", "Is it sunny today?"]


def test_forward_shapes(tokenizer):
    model = tiny_model(len(tokenizer))
    col = VFMCollator(tokenizer, max_length=128, max_criterion_length=T)
    batch = col([{"context": CONTEXT, "target": TARGET, "criteria": CRITERIA, "labels": ["YES", "NO", "N/A"]},
                 {"context": "Hello", "target": "Hi there, it is sunny.", "criteria": CRITERIA,
                  "labels": ["NO", "YES", "YES"]}])
    out = model(**batch)
    assert out.logits.shape == (2, 3, 3)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_criteria_are_independent_and_order_invariant(tokenizer):
    model = tiny_model(len(tokenizer))
    full = model.verify_batch(tokenizer, [{"context": CONTEXT, "target": TARGET, "criterion": c} for c in CRITERIA])
    together = model.verify(tokenizer, CONTEXT, TARGET, CRITERIA)
    reversed_ = model.verify(tokenizer, CONTEXT, TARGET, CRITERIA[::-1])[::-1]
    one_by_one = [model.verify(tokenizer, CONTEXT, TARGET, [c])[0] for c in CRITERIA]
    for a, b, c, d in zip(full, together, reversed_, one_by_one):
        for k in ("p_yes", "p_no", "p_na"):
            assert a[k] == pytest.approx(b[k], abs=1e-5)
            assert b[k] == pytest.approx(c[k], abs=1e-5)
            assert b[k] == pytest.approx(d[k], abs=1e-5)


def test_prefix_cache_matches_full_forward(tokenizer):
    model = tiny_model(len(tokenizer))
    col = VFMCollator(tokenizer, max_length=128, max_criterion_length=T)
    batch = col([{"context": CONTEXT, "target": TARGET, "criteria": CRITERIA}])
    with torch.no_grad():
        full = model(**batch).logits[0]
    chunked = torch.tensor([r["logits"] for r in model.verify(tokenizer, CONTEXT, TARGET, CRITERIA, chunk_size=2)])
    assert torch.allclose(full, chunked, atol=1e-5)


def test_left_padding_does_not_change_scores(tokenizer):
    model = tiny_model(len(tokenizer))
    rows = [{"context": CONTEXT, "target": TARGET, "criterion": CRITERIA[0]},
            {"context": "Hi", "target": "Hello.", "criterion": CRITERIA[0]}]
    alone = model.verify_batch(tokenizer, rows[1:])[0]
    padded = model.verify_batch(tokenizer, rows)[1]
    assert alone["p_yes"] == pytest.approx(padded["p_yes"], abs=1e-5)


def test_save_and_load_roundtrip(tokenizer, tmp_path):
    model = tiny_model(len(tokenizer))
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    for f in ("configuration_vfm.py", "modeling_vfm.py", "collator_vfm.py"):
        assert (tmp_path / f).exists()
    reloaded = VFMForCriteriaVerification.from_pretrained(tmp_path).eval()
    remote = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    a = model.verify(tokenizer, CONTEXT, TARGET, CRITERIA)
    for other in (reloaded, remote):
        b = other.verify(tokenizer, CONTEXT, TARGET, CRITERIA)
        assert [x["p_yes"] for x in a] == pytest.approx([x["p_yes"] for x in b], abs=1e-6)


def test_bf16_forward_runs(tokenizer):
    model = tiny_model(len(tokenizer), dtype=torch.bfloat16)
    res = model.verify(tokenizer, CONTEXT, TARGET, CRITERIA)
    assert {r["verdict"] for r in res} <= {"YES", "NO", "N/A"}
    assert all(abs(r["p_yes"] + r["p_no"] + r["p_na"] - 1) < 1e-3 for r in res)


def test_metrics():
    assert M.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == 1.0
    assert M.roc_auc([1, 0], [0.5, 0.5]) == 0.5
    m = M.binary_metrics(["YES", "NO", "YES", "NO"], ["YES", "NO", "NO", "NO"])
    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 2 and m["balanced_accuracy"] == pytest.approx(5 / 6)
    t, v = M.tune_threshold([0.9, 0.2, 0.6, 0.4], ["YES", "NO", "YES", "NO"], "balanced_accuracy")
    assert v == 1.0 and 0.4 <= t <= 0.6
    tm = M.ternary_metrics(["YES", "NO", "N/A"], ["YES", "NO", "NO"])
    assert tm["accuracy"] == pytest.approx(2 / 3)
    assert M.mean_std([1.0, 3.0]) == (2.0, 1.0)


def test_record_format(tokenizer):
    ctx = [{"role": "user", "content": "Hi"}]
    r = make_record(ctx, "Hello", "Is it a greeting?", "N/A")
    assert r["id"] == record_id(ctx, "Hello", "Is it a greeting?") != record_id(ctx, "Hello", "Q!")
    assert VFMCollator(tokenizer)([r])["labels"].tolist() == [[2]]
    with pytest.raises(ValueError):
        make_record(ctx, "Hello", "Is it a greeting?", "INSUFFICIENT_INFO")
