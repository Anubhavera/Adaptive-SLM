#!/usr/bin/env python3
"""Resumable pretrained-model LoRA baseline for Colab/Kaggle.

This establishes training/export reliability; it does not establish agent skill
or the novelty of the separate custom-architecture experiment in train.py.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_messages(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each record needs a messages list with context and a final assistant response")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("This baseline accepts system/user/assistant messages; serialize tool targets as assistant JSON")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Message content must be a nonempty string")
    if messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages[:-1]):
        raise ValueError("Record must contain a user prompt and end with an assistant target")
    return messages


def encode_response(messages, tokenizer, max_length):
    # Use the actual inference prefix, including Qwen's non-thinking suffix.
    # Keep earlier conversation as context; supervise only the final response.
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    answer = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    answer = answer + [tokenizer.eos_token_id]
    ids = prefix + answer
    if len(ids) > max_length:
        return None  # Never train on a truncated tool call or empty response.
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + answer}


def prepare_splits(rows, tokenizer, max_length, eval_fraction, seed):
    train, evaluation, seen = [], [], set()
    stats = {"read": 0, "duplicates": 0, "overlength": 0}
    for row in rows:
        stats["read"] += 1
        messages = validate_messages(row)
        key = digest(messages)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        encoded = encode_response(messages, tokenizer, max_length)
        if encoded is None:
            stats["overlength"] += 1
            continue
        # Keep the same question together across personas/alternative answers.
        questions = [" ".join(m["content"].split()) for m in messages if m["role"] == "user"]
        bucket = int(digest([seed, questions])[:16], 16) / 2**64
        (evaluation if bucket < eval_fraction else train).append(encoded)
    if not train or not evaluation:
        raise ValueError(f"Need nonempty train and validation sets after filtering: {stats}; use more samples or a larger max-length")
    stats.update(train=len(train), validation=len(evaluation),
                 tokenized_sha256=digest([train, evaluation]))
    return Dataset.from_list(train), Dataset.from_list(evaluation), stats


def load_rows(args, dataset_revision):
    if args.data:
        with Path(args.data).open(encoding="utf-8") as handle:
            count = 0
            for line in handle:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if count >= args.max_samples:
                        break
    else:
        # Access/schema errors propagate with a full traceback instead of silently
        # replacing the research data or exporting a model trained on zero rows.
        # Materialize the pinned snapshot in Arrow's disk cache before sampling.
        # Early-stopped remote Parquet streams can leave background readers alive
        # at interpreter shutdown; a local snapshot is also restartable offline.
        source = load_dataset(args.dataset, args.dataset_config, split=args.split,
                              revision=dataset_revision, streaming=False)
        yield from source.shuffle(seed=args.seed).select(range(min(args.max_samples, len(source))))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data", help="Local JSONL messages file instead of a remote dataset")
    parser.add_argument("--output-dir", default="./runs/mobile-baseline")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU smoke tests; slow for real training")
    args = parser.parse_args()
    if min(args.max_samples, args.max_length, args.max_steps, args.batch_size,
           args.grad_accum, args.rank, args.save_steps) <= 0 or not 0 < args.eval_fraction < 1:
        parser.error("Counts must be positive and eval-fraction must be between zero and one")
    return args


def main():
    args = parse_args()
    if not args.cpu and not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab/Kaggle GPU runtime before training (or --cpu for an explicit smoke test)")
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model_info = api.model_info(args.model, revision=args.model_revision)
    model_revision = model_info.sha
    dataset_info = None if args.data else api.dataset_info(args.dataset, revision=args.dataset_revision)
    dataset_revision = None if dataset_info is None else dataset_info.sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision)
    if not tokenizer.chat_template:
        raise ValueError("Selected model has no chat template; define and test one before training")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, evaluation, stats = prepare_splits(
        load_rows(args, dataset_revision), tokenizer, args.max_length, args.eval_fraction, args.seed,
    )
    manifest = {
        "model": args.model, "model_revision": model_revision,
        "dataset": args.data or args.dataset, "dataset_revision": dataset_revision,
        "dataset_license": None if dataset_info is None else getattr(dataset_info.card_data, "license", None),
        "max_length": args.max_length, "rank": args.rank, "seed": args.seed,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "max_steps": args.max_steps, "data": stats,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "datasets", "accelerate", "peft", "safetensors"]},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.preflight_only:
        (out / "preflight.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Preflight passed; no model weights loaded and no training performed.")
        return
    manifest_path = out / "run_manifest.json"
    checkpoint = get_last_checkpoint(str(out))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ["model", "model_revision", "dataset_revision", "max_length", "rank", "seed", "data",
                    "batch_size", "grad_accum", "lr", "max_steps", "packages"]:
            if old.get(key) != manifest[key]:
                raise ValueError(f"Run changed at {key}; use a new output directory for a new experiment")
        if not checkpoint:
            raise ValueError("Existing run has no resumable checkpoint; use a new output directory")
    elif checkpoint:
        raise ValueError("Checkpoint has no run manifest; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bf16 = not args.cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.float32 if args.cpu else (torch.bfloat16 if bf16 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=model_revision, dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    completed = False
    if checkpoint:
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        completed = state["global_step"] >= args.max_steps
    if completed:
        # Trainer.train can take another step when resuming an already-finished
        # run. Reload the saved adapter directly and only evaluate/export it.
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank,
            lora_dropout=0.05, target_modules="all-linear",
        ))
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(out), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_steps=min(10, args.max_steps // 10), lr_scheduler_type="cosine",
        optim="adamw_torch", weight_decay=0.01, max_grad_norm=1.0,
        bf16=bf16, fp16=not args.cpu and not bf16, use_cpu=args.cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, seed=args.seed, data_seed=args.seed,
        logging_steps=1, logging_nan_inf_filter=False, report_to="none",
        eval_strategy="steps", eval_steps=args.save_steps, prediction_loss_only=True,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        save_only_model=False, remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=evaluation,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    if completed:
        print("Checkpoint already reached max-steps; evaluating and exporting without another training step.", flush=True)
    else:
        trainer.train(resume_from_checkpoint=checkpoint)
    metrics = trainer.evaluate()
    if torch.cuda.is_available() and not args.cpu:
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    trainer.save_metrics("eval", metrics)
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out / "merged")
    print(f"Finished. Adapter: {out / 'adapter'}; HF model for GGUF conversion: {out / 'merged'}")


if __name__ == "__main__":
    main()
