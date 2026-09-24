#!/usr/bin/env python3
"""
Kaggle-ready training script for AdaptiveSLM.
Optimized for Kaggle's 30h limit and P100/T4 GPUs.
"""

import os
import sys
import argparse

# Add training to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import TrainingConfig, AdaptiveSLMTrainer

def run_kaggle_training():
    """Run training optimized for Kaggle environment."""
    
    config = TrainingConfig()
    
    # Kaggle-specific adjustments
    config.batch_size = 16  # Smaller for P100 16GB
    config.gradient_accumulation_steps = 16  # Effective batch = 256
    config.max_steps = 50000  # Reduced for 30h limit
    config.warmup_steps = 1000
    config.learning_rate = 2e-4
    
    # Output to Kaggle working directory
    config.output_dir = "/kaggle/working/checkpoints"
    config.data_path = "/kaggle/working/data"
    
    # Use smaller teacher for memory
    config.teacher_model = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller than 7B
    
    print("=" * 60)
    print("AdaptiveSLM Training - Kaggle Optimized")
    print("=" * 60)
    print(f"Batch size: {config.batch_size}")
    print(f"Grad accum: {config.gradient_accumulation_steps}")
    print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Max steps: {config.max_steps}")
    print(f"Teacher: {config.teacher_model}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)
    
    trainer = AdaptiveSLMTrainer(config)
    
    # Phase 1: Pre-training
    print("\n>>> Phase 1: Pre-training")
    pretrain_path = os.path.join(config.data_path, "pretrain_corpus.jsonl")
    if os.path.exists(pretrain_path):
        trainer.pretrain(pretrain_path, num_epochs=1)
    else:
        print("  Pre-train data not found, skipping...")
    
    # Phase 2: Distillation
    print("\n>>> Phase 2: Knowledge Distillation")
    distill_path = os.path.join(config.data_path, "distill_data.jsonl")
    if os.path.exists(distill_path):
        trainer.distill(distill_path, num_epochs=1)
    else:
        print("  Distill data not found, skipping...")
    
    # Phase 3: PAKD
    print("\n>>> Phase 3: PAKD Fine-tuning")
    pakd_path = os.path.join(config.data_path, "pakd_data.jsonl")
    if os.path.exists(pakd_path):
        trainer.pakd_finetune(pakd_path, num_epochs=1)
    else:
        print("  PAKD data not found, skipping...")
    
    # Phase 4: Export
    print("\n>>> Phase 4: Export to GGUF")
    trainer.export_gguf("/kaggle/working/adaptive_slm.gguf")
    
    print("\n>>> Training complete! Model at /kaggle/working/adaptive_slm.gguf")
    print(">>> Download from Kaggle output and quantize locally with llama.cpp")

if __name__ == "__main__":
    run_kaggle_training()