#!/usr/bin/env python3
"""
Data preparation for AdaptiveSLM training on cloud (Colab/Kaggle).
Streams public datasets and writes JSONL files for the pretrain, distill, and PAKD phases.
"""

import os
import json
import argparse

from datasets import load_dataset
from tqdm import tqdm

PAKD_SYSTEM_PROMPTS = {
    "beginner": "You are a helpful assistant. Explain things in simple, beginner-friendly language with short sentences and everyday examples.",
    "intermediate": "You are a helpful assistant. Explain things with moderate technical detail, assuming a working knowledge of the topic.",
    "expert": "You are a helpful assistant. Give dense, expert-level explanations using precise domain terminology.",
    "general": "You are a helpful assistant.",
}


def _alpaca_user_content(item):
    instruction = item.get("instruction") or ""
    text_input = item.get("input") or ""
    if text_input:
        return f"{instruction}\n\n{text_input}"
    return instruction


def _alpaca_messages(item):
    instruction = item.get("instruction") or ""
    output = item.get("output") or ""
    if not instruction or not output:
        return None
    return [
        {"role": "user", "content": _alpaca_user_content(item)},
        {"role": "assistant", "content": output},
    ]


def _gsm8k_messages(item):
    question = item.get("question") or ""
    answer = item.get("answer") or ""
    if not question or not answer:
        return None
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _sciq_messages(item):
    support = item.get("support") or ""
    question = item.get("question") or ""
    correct_answer = item.get("correct_answer") or ""
    if not question or not correct_answer:
        return None
    messages = []
    if support:
        messages.append({"role": "system", "content": support})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": correct_answer})
    return messages


def _ultrachat_messages(item):
    raw_messages = item.get("messages") or []
    user_idx = None
    for idx, msg in enumerate(raw_messages):
        if msg.get("role") == "user":
            user_idx = idx
            break
    if user_idx is None:
        return None
    user_content = raw_messages[user_idx].get("content") or ""
    assistant_content = None
    for msg in raw_messages[user_idx + 1:]:
        if msg.get("role") == "assistant":
            assistant_content = msg.get("content") or ""
            break
    if not user_content or not assistant_content:
        return None
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def prepare_pretrain_data(output_path, num_samples=100000):
    print(f"Preparing pre-training data: {num_samples} samples")
    sources = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.8, "text"),
        ("codeparrot/codeparrot-clean", None, "train", 0.2, "code"),
    ]
    all_texts = []
    for ds_name, config, split, ratio, field in sources:
        n = int(num_samples * ratio)
        print(f"  Loading {n} samples from {ds_name}...")
        try:
            ds = load_dataset(ds_name, config, split=split, streaming=True)
            for item in tqdm(ds.take(n), total=n):
                text = item.get(field) or ""
                if len(text) > 200:
                    all_texts.append(text)
        except Exception as e:
            print(f"  Warning: failed to load {ds_name}: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_texts)} samples to {output_path}")


def prepare_distill_data(output_path, num_samples=50000):
    print(f"Preparing distillation data: {num_samples} samples")
    sources = [
        ("tatsu-lab/alpaca", None, "train", 0.30, _alpaca_messages),
        ("openai/gsm8k", "main", "train", 0.15, _gsm8k_messages),
        ("allenai/sciq", None, "train", 0.15, _sciq_messages),
        ("HuggingFaceH4/ultrachat_200k", None, "train_sft", 0.40, _ultrachat_messages),
    ]
    all_conversations = []
    for ds_name, config, split, ratio, to_messages in sources:
        n = int(num_samples * ratio)
        print(f"  Loading {n} samples from {ds_name}...")
        try:
            ds = load_dataset(ds_name, config, split=split, streaming=True)
            for item in tqdm(ds.take(n), total=n):
                messages = to_messages(item)
                if messages:
                    all_conversations.append(messages)
        except Exception as e:
            print(f"  Warning: failed to load {ds_name}: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        for messages in all_conversations:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_conversations)} samples to {output_path}")


def prepare_pakd_data(output_path, num_samples=20000):
    print(f"Preparing PAKD data: {num_samples} samples")
    profiles = ["beginner", "intermediate", "expert", "general"]
    all_data = []
    print(f"  Loading {num_samples} samples from tatsu-lab/alpaca...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        for item in tqdm(ds.take(num_samples), total=num_samples):
            instruction = item.get("instruction") or ""
            output = item.get("output") or ""
            if not instruction or not output:
                continue
            profile = profiles[len(all_data) % len(profiles)]
            all_data.append({
                "messages": [
                    {"role": "system", "content": PAKD_SYSTEM_PROMPTS[profile]},
                    {"role": "user", "content": _alpaca_user_content(item)},
                    {"role": "assistant", "content": output},
                ],
                "profile": profile,
            })
    except Exception as e:
        print(f"  Warning: failed to load tatsu-lab/alpaca: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--pretrain-samples", type=int, default=100000)
    parser.add_argument("--distill-samples", type=int, default=50000)
    parser.add_argument("--pakd-samples", type=int, default=20000)
    parser.add_argument("--phase", choices=["pretrain", "distill", "pakd", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase in ("pretrain", "all"):
        prepare_pretrain_data(
            os.path.join(args.output_dir, "pretrain_corpus.jsonl"),
            args.pretrain_samples,
        )

    if args.phase in ("distill", "all"):
        prepare_distill_data(
            os.path.join(args.output_dir, "distill_data.jsonl"),
            args.distill_samples,
        )

    if args.phase in ("pakd", "all"):
        prepare_pakd_data(
            os.path.join(args.output_dir, "pakd_data.jsonl"),
            args.pakd_samples,
        )

    print("\nData preparation complete!")
    print(f"Files in {args.output_dir}:")
    for filename in os.listdir(args.output_dir):
        size = os.path.getsize(os.path.join(args.output_dir, filename)) / 1024 / 1024
        print(f"  {filename}: {size:.1f} MB")


if __name__ == "__main__":
    main()
