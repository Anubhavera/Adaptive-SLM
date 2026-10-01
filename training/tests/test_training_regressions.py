"""Numerical/data regressions; tiny CPU tensors, no network or model download."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import DistillationLoss, PAKDLoss, RMSNorm
from finetune_mobile import encode_response, prepare_splits, validate_messages


def tokenizer():
    raw = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2,
                               "user": 3, "assistant": 4, "hello": 5, "yes": 6}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    return tok


def test_fp16_rmsnorm_does_not_overflow_at_moderate_activation():
    out = RMSNorm(64)(torch.full((2, 64), 300., dtype=torch.float16))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out), atol=0.002, rtol=0.002)


@pytest.mark.parametrize("profile", [False, True])
def test_kd_ignores_masked_target_positions_and_backpropagates(profile):
    torch.manual_seed(7)
    s = torch.randn(2, 5, 9, requires_grad=True)
    t = torch.randn(2, 5, 9)
    y = torch.tensor([[1, 2, 3, -100, -100], [1, 4, 2, -100, -100]])
    changed = t.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 100
    loss_fn = PAKDLoss() if profile else DistillationLoss()
    extra = [torch.tensor([0, 2])] if profile else []
    loss = loss_fn(s, t, y, *extra)
    torch.testing.assert_close(loss, loss_fn(s, changed, y, *extra))
    loss.backward()
    assert torch.isfinite(s.grad).all()
    assert s.grad[:, 2:].abs().max() == 0


def test_kd_normalization_is_independent_of_repeated_valid_tokens():
    torch.manual_seed(8)
    s = torch.randn(1, 1, 11).expand(1, 5, 11)
    t = torch.randn(1, 1, 11).expand(1, 5, 11)
    loss = DistillationLoss(alpha=1)
    torch.testing.assert_close(loss(s, t, torch.ones(1, 5, dtype=torch.long)),
                               loss(s[:, :2], t[:, :2], torch.ones(1, 2, dtype=torch.long)))


def test_empty_targets_have_zero_loss_and_finite_zero_gradients():
    s = torch.randn(1, 4, 7, requires_grad=True)
    loss = PAKDLoss()(s, torch.randn_like(s), torch.full((1, 4), -100), torch.tensor([0]))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(s.grad).all() and s.grad.abs().sum() == 0


def test_response_labels_exclude_context_and_padding_preserves_eos():
    tok = tokenizer()
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "yes"}]
    row = encode_response(messages, tok, 64)
    assert row["labels"][:-2] == [-100] * (len(row["labels"]) - 2)
    assert row["labels"][-2:] == [6, 2]
    longer = encode_response([{"role": "user", "content": "hello hello"}, messages[-1]], tok, 64)
    batch = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100)([row, longer])
    assert batch["labels"][0, -1].item() == -100
    assert batch["labels"][0, len(row["labels"]) - 1].item() == tok.eos_token_id
    assert encode_response(messages, tok, 2) is None


def test_prompt_groups_do_not_leak_between_splits():
    rows = []
    for i in range(40):
        for persona in ["beginner", "expert"]:
            rows.append({"messages": [{"role": "system", "content": persona},
                                      {"role": "user", "content": f"hello {i}"},
                                      {"role": "assistant", "content": "yes"}]})
    # Unique raw texts, but group by user question. Instrument via encoded IDs
    # with a tokenizer retaining the numerical questions.
    class RecordingTokenizer:
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            return [100 + int(messages[-1]["content"].split()[-1])]
        def __call__(self, text, **kwargs):
            return {"input_ids": [6]}
    train, evaluation, stats = prepare_splits(rows + [rows[0]], RecordingTokenizer(), 20, 0.2, 42)
    assert {r["input_ids"][0] for r in train}.isdisjoint({r["input_ids"][0] for r in evaluation})
    assert stats["duplicates"] == 1 and len(train) + len(evaluation) == 80


def test_invalid_records_fail_instead_of_becoming_empty_training():
    with pytest.raises(ValueError):
        validate_messages({"messages": [{"role": "user", "content": "hello"}]})


def test_lora_checkpoint_resume_and_merged_export(tmp_path):
    """Resume an interrupted planned run, compare weights with uninterrupted
    training, then verify merged HF reload numerically. No download required.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM,
        Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed,
    )
    torch.set_num_threads(1)
    cfg = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    data = Dataset.from_list([{"input_ids": [3 + i, 4, 5, 6, 2], "attention_mask": [1] * 5,
                              "labels": [-100, -100, 5, 6, 2]} for i in range(8)])

    def model():
        set_seed(42)
        return get_peft_model(LlamaForCausalLM(cfg), LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.1, target_modules="all-linear"))

    def trainer(m, directory, callbacks=None):
        args = TrainingArguments(output_dir=str(directory), max_steps=4,
                                 per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                 learning_rate=1e-3, optim="adamw_torch", use_cpu=True,
                                 save_steps=2, save_total_limit=2, report_to="none", disable_tqdm=True,
                                 gradient_checkpointing=True,
                                 gradient_checkpointing_kwargs={"use_reentrant": False})
        return Trainer(model=m, args=args, train_dataset=data,
                       data_collator=default_data_collator, callbacks=callbacks)

    class StopAtTwo(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True
            return control

    uninterrupted = model()
    trainer(uninterrupted, tmp_path / "full").train()
    trainer(model(), tmp_path / "resume", [StopAtTwo()]).train()
    checkpoint = tmp_path / "resume" / "checkpoint-2"
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "scheduler.pt").exists()
    assert (checkpoint / "rng_state.pth").exists()
    resumed = model()
    trainer(resumed, tmp_path / "resume").train(resume_from_checkpoint=str(checkpoint))
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=1e-6, rtol=1e-5)
    resumed.eval()
    inputs = torch.tensor([[3, 4, 5]])
    before = resumed(input_ids=inputs).logits.detach()
    merged = resumed.merge_and_unload()
    merged.save_pretrained(tmp_path / "merged", safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged").eval()
    torch.testing.assert_close(before, reloaded(input_ids=inputs).logits, atol=1e-5, rtol=1e-4)
