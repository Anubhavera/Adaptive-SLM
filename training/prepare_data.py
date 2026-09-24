"""
Data Preparation for AdaptiveSLM Training

This script prepares high-quality training data with profile annotations
for Profile-Aware Knowledge Distillation (PAKD).

Data Sources:
1. Pre-training: Filtered web corpus (FineWeb-Edu, SlimPajama)
2. Distillation: High-quality instruction data (Alpaca, Dolly)
3. PAKD: Profile-annotated domain data
"""

import json
import os
import random
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Data Quality Filters
# ============================================================================

def quality_filter(text: str, min_length: int = 50, max_length: int = 4096) -> bool:
    """Filter low-quality text"""
    if not text or len(text) < min_length or len(text) > max_length:
        return False
    
    # Remove mostly non-alphabetic content
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False
    
    # Remove repetitive content
    words = text.split()
    if len(words) < 10:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:  # Too repetitive
        return False
    
    return True

def estimate_complexity(text: str) -> str:
    """Estimate text complexity for profile assignment"""
    words = text.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    # Simple heuristics for complexity estimation
    # (In production, use a classifier or LLM)
    if avg_word_length < 4.5:
        return "beginner"
    elif avg_word_length < 5.5:
        return "intermediate"
    else:
        return "expert"

def detect_domain(text: str) -> str:
    """Detect domain category of text"""
    text_lower = text.lower()
    
    # Simple keyword-based detection
    coding_keywords = ["function", "code", "programming", "python", "javascript", "api", "class", "def "]
    science_keywords = ["research", "study", "experiment", "hypothesis", "data", "analysis", "scientific"]
    creative_keywords = ["story", "imagine", "creative", "fiction", "poem", "narrative"]
    
    if any(kw in text_lower for kw in coding_keywords):
        return "coding"
    elif any(kw in text_lower for kw in science_keywords):
        return "science"
    elif any(kw in text_lower for kw in creative_keywords):
        return "creative"
    else:
        return "general"

# ============================================================================
# Dataset Preparation Functions
# ============================================================================

def prepare_pretrain_data(
    output_path: str,
    num_samples: int = 100000,
    max_length: int = 2048
):
    """
    Prepare pre-training data from FineWeb-Edu
    (High-quality educational web content)
    """
    print("Preparing pre-training data...")
    
    # Load FineWeb-Edu subset
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",  # 10B token sample
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Could not load FineWeb-Edu: {e}")
        print("Using placeholder data instead...")
        # Create placeholder data for testing
        create_placeholder_data(output_path, num_samples)
        return
    
    samples = []
    for item in tqdm(dataset, total=num_samples, desc="Filtering"):
        text = item.get("text", "")
        
        if quality_filter(text):
            samples.append({
                "text": text[:max_length],
                "source": "fineweb-edu"
            })
        
        if len(samples) >= num_samples:
            break
    
    # Save to JSONL
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")

def prepare_distillation_data(
    output_path: str,
    num_samples: int = 50000
):
    """
    Prepare high-quality instruction data for knowledge distillation
    """
    print("Preparing distillation data...")
    
    datasets_to_use = [
        ("databricks/databricks-dolly-15k", "train"),
        ("tatsu-lab/alpaca", "train"),
    ]
    
    samples = []
    
    for ds_name, split in datasets_to_use:
        try:
            dataset = load_dataset(ds_name, split=split)
            
            for item in tqdm(dataset, desc=f"Processing {ds_name}"):
                # Format instruction data
                instruction = item.get("instruction", item.get("question", ""))
                response = item.get("response", item.get("output", item.get("answer", "")))
                
                if instruction and response:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                    
                    if quality_filter(text, min_length=30):
                        samples.append({
                            "text": text,
                            "source": ds_name.split("/")[-1]
                        })
        except Exception as e:
            print(f"Could not load {ds_name}: {e}")
    
    # Shuffle and limit
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    # Save
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")

def prepare_pakd_data(
    output_path: str,
    num_samples: int = 30000
):
    """
    Prepare profile-annotated data for PAKD training
    Each sample is annotated with:
    - profile: beginner/intermediate/expert
    - domain: general/coding/science/creative
    """
    print("Preparing PAKD data with profile annotations...")
    
    # Combine multiple sources
    all_samples = []
    
    # Load instruction data
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        for item in tqdm(alpaca, desc="Processing Alpaca"):
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            
            if instruction and output:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                if quality_filter(text, min_length=30):
                    all_samples.append({
                        "text": text,
                        "profile": estimate_complexity(output),
                        "domain": detect_domain(text),
                        "source": "alpaca"
                    })
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    
    # Load more diverse data if available
    try:
        openorca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        count = 0
        for item in tqdm(openorca, desc="Processing OpenOrca", total=20000):
            question = item.get("question", "")
            response = item.get("response", "")
            
            if question and response:
                text = f"### Question:\n{question}\n\n### Answer:\n{response}"
                
                if quality_filter(text, min_length=30):
                    all_samples.append({
                        "text": text,
                        "profile": estimate_complexity(response),
                        "domain": detect_domain(text),
                        "source": "openorca"
                    })
                    count += 1
            
            if count >= 20000:
                break
    except Exception as e:
        print(f"Could not load OpenOrca: {e}")
    
    # Balance profiles if possible
    profile_counts = {"beginner": 0, "intermediate": 0, "expert": 0}
    for sample in all_samples:
        profile_counts[sample["profile"]] = profile_counts.get(sample["profile"], 0) + 1
    
    print(f"Profile distribution: {profile_counts}")
    
    # Shuffle and limit
    random.shuffle(all_samples)
    samples = all_samples[:num_samples]
    
    # Save
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    
    # Print statistics
    final_counts = {"beginner": 0, "intermediate": 0, "expert": 0}
    domain_counts = {"general": 0, "coding": 0, "science": 0, "creative": 0}
    
    for sample in samples:
        final_counts[sample["profile"]] = final_counts.get(sample["profile"], 0) + 1
        domain_counts[sample["domain"]] = domain_counts.get(sample["domain"], 0) + 1
    
    print(f"Final profile distribution: {final_counts}")
    print(f"Domain distribution: {domain_counts}")

def create_placeholder_data(output_path: str, num_samples: int = 1000):
    """Create placeholder data for testing the pipeline"""
    samples = []
    
    topics = [
        "What is machine learning?",
        "Explain how computers work.",
        "Write a simple Python function.",
        "What is the meaning of life?",
        "How does the internet work?",
        "Explain quantum physics simply.",
        "What is climate change?",
        "How to learn programming?",
    ]
    
    profiles = ["beginner", "intermediate", "expert"]
    domains = ["general", "coding", "science", "creative"]
    
    for i in range(num_samples):
        topic = random.choice(topics)
        profile = random.choice(profiles)
        domain = random.choice(domains)
        
        response = f"This is a {profile}-level explanation about {topic.lower()} "
        response += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * random.randint(2, 10)
        
        samples.append({
            "text": f"### Question:\n{topic}\n\n### Answer:\n{response}",
            "profile": profile,
            "domain": domain,
            "source": "placeholder"
        })
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(samples)} placeholder samples at {output_path}")

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare AdaptiveSLM training data")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--pretrain-samples", type=int, default=100000)
    parser.add_argument("--distill-samples", type=int, default=50000)
    parser.add_argument("--pakd-samples", type=int, default=30000)
    parser.add_argument("--placeholder-only", action="store_true", help="Create only placeholder data")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.placeholder_only:
        create_placeholder_data(
            os.path.join(args.output_dir, "pretrain_corpus.jsonl"),
            args.pretrain_samples
        )
        create_placeholder_data(
            os.path.join(args.output_dir, "distill_data.jsonl"),
            args.distill_samples
        )
        create_placeholder_data(
            os.path.join(args.output_dir, "pakd_data.jsonl"),
            args.pakd_samples
        )
    else:
        prepare_pretrain_data(
            os.path.join(args.output_dir, "pretrain_corpus.jsonl"),
            args.pretrain_samples
        )
        prepare_distillation_data(
            os.path.join(args.output_dir, "distill_data.jsonl"),
            args.distill_samples
        )
        prepare_pakd_data(
            os.path.join(args.output_dir, "pakd_data.jsonl"),
            args.pakd_samples
        )
    
    print("\nData preparation complete!")
    print(f"Files saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()
