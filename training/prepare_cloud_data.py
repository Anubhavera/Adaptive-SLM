#!/usr/bin/env python3
"""
Data preparation for AdaptiveSLM training on cloud (Colab/Kaggle).
Downloads and processes datasets for all training phases.
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def prepare_pretrain_data(output_path, num_samples=100000):
    """Prepare pre-training corpus from high-quality web data."""
    print(f"Preparing pre-training data: {num_samples} samples")
    
    # Use FineWeb-Edu (high quality educational content) + Code
    datasets_to_load = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.8),
        ("codeparrot/codeparrot-clean", None, "train", 0.2),
    ]
    
    all_texts = []
    for ds_name, config, split, ratio in datasets_to_load:
        n = int(num_samples * ratio)
        print(f"  Loading {n} samples from {ds_name}...")
        ds = load_dataset(ds_name, config, split=split, streaming=True)
        for i, item in enumerate(tqdm(ds.take(n), total=n)):
            text = item.get("text", item.get("content", ""))
            if len(text) > 100:
                all_texts.append(text)
    
    # Save as JSONL
    with open(output_path, 'w') as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}) + "\n")
    
    print(f"Saved {len(all_texts)} samples to {output_path}")

def prepare_distill_data(output_path, num_samples=50000):
    """Prepare distillation data - diverse prompts for teacher generation."""
    print(f"Preparing distillation data: {num_samples} samples")
    
    # Mix of instruction following, coding, reasoning, creative
    datasets_to_load = [
        ("tatsu-lab/alpaca", None, "train", 0.25),
        ("codeparrot/codeparrot-clean", None, "train", 0.25),
        ("openai/gsm8k", "main", "train", 0.15),
        ("allenai/sciq", None, "train", 0.15),
        ("HuggingFaceH4/ultrachat_200k", None, "train_sft", 0.2),
    ]
    
    all_texts = []
    for ds_name, config, split, ratio in datasets_to_load:
        n = int(num_samples * ratio)
        print(f"  Loading {n} samples from {ds_name}...")
        try:
            ds = load_dataset(ds_name, config, split=split, streaming=True)
            for i, item in enumerate(tqdm(ds.take(n), total=n)):
                # Format as instruction-response pairs
                if "instruction" in item and "output" in item:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                elif "question" in item and "answer" in item:
                    text = f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}"
                elif "prompt" in item and "completion" in item:
                    text = f"### Prompt:\n{item['prompt']}\n\n### Completion:\n{item['completion']}"
                elif "text" in item:
                    text = item["text"]
                else:
                    continue
                
                if len(text) > 50:
                    all_texts.append(text)
        except Exception as e:
            print(f"  Warning: Failed to load {ds_name}: {e}")
    
    with open(output_path, 'w') as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}) + "\n")
    
    print(f"Saved {len(all_texts)} samples to {output_path}")

def prepare_pakd_data(output_path, num_samples=20000):
    """Prepare PAKD data with profile annotations."""
    print(f"Preparing PAKD data: {num_samples} samples")
    
    profiles = ["beginner", "intermediate", "expert", "general"]
    profile_prompts = {
        "beginner": [
            "Explain like I'm 5: ",
            "Simple explanation for beginner: ",
            "Easy to understand: ",
        ],
        "intermediate": [
            "Explain with some technical detail: ",
            "Technical overview: ",
            "Detailed explanation: ",
        ],
        "expert": [
            "Expert-level analysis: ",
            "Deep technical dive: ",
            "Comprehensive technical explanation: ",
        ],
        "general": [
            "Explain: ",
            "Overview: ",
            "Summary: ",
        ],
    }
    
    # Load base datasets
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    
    all_data = []
    for i, item in enumerate(tqdm(ds.take(num_samples), total=num_samples)):
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output:
            continue
        
        # Assign profile cyclically
        profile = profiles[i % len(profiles)]
        prefix = profile_prompts[profile][i % len(profile_prompts[profile])]
        
        text = f"{prefix}{instruction}\n\n### Response:\n{output}"
        all_data.append({
            "text": text,
            "profile": profile
        })
    
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
    
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
            args.pretrain_samples
        )
    
    if args.phase in ("distill", "all"):
        prepare_distill_data(
            os.path.join(args.output_dir, "distill_data.jsonl"),
            args.distill_samples
        )
    
    if args.phase in ("pakd", "all"):
        prepare_pakd_data(
            os.path.join(args.output_dir, "pakd_data.jsonl"),
            args.pakd_samples
        )
    
    print("\nData preparation complete!")
    print(f"Files in {args.output_dir}:")
    for f in os.listdir(args.output_dir):
        size = os.path.getsize(os.path.join(args.output_dir, f)) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")

if __name__ == "__main__":
    main()