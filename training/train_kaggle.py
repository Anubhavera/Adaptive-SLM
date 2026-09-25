#!/usr/bin/env python3
"""
Kaggle-ready training script for AdaptiveSLM.
Optimized for Kaggle's 30h limit and P100/T4 GPUs.
"""

import os
import sys

# Add training to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import TrainingConfig, AdaptiveSLMTrainer

def run_kaggle_training():
    """Run training optimized for Kaggle environment."""

    config = TrainingConfig()

    # Kaggle-specific adjustments
    config.batch_size = 8
    config.distill_batch_size = 2
    config.gradient_accumulation_steps = 16
    config.max_steps = 20000
    config.warmup_steps = 200
    config.learning_rate = 2e-4
    config.max_length = 512
    config.output_dir = "/kaggle/working/checkpoints"
    config.data_path = "/kaggle/working/data"
    config.teacher_model = "Qwen/Qwen3-1.7B-Instruct-2507"  # P100 16GB can't fit 4B teacher

    print("=" * 60)
    print("AdaptiveSLM Training - Kaggle Optimized")
    print("=" * 60)
    print(f"Batch size: {config.batch_size}")
    print(f"Distill batch size: {config.distill_batch_size}")
    print(f"Grad accum: {config.gradient_accumulation_steps}")
    print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Max steps: {config.max_steps}")
    print(f"Max length: {config.max_length}")
    print(f"Teacher: {config.teacher_model}")
    print("Note: free Colab T4 uses the Qwen/Qwen3-4B-Instruct-2507 teacher")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    trainer = AdaptiveSLMTrainer(config, resume=True)

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
    print("\n>>> Phase 4: Export to HF format")
    trainer.export_gguf("/kaggle/working/hf_export", depths=(30, 22, 15))

    print("\n>>> Training complete!")
    print(">>> HF exports are under /kaggle/working/hf_export (depth30/, depth22/, depth15/)")
    print(">>> Convert + quantize locally with llama.cpp:")
    print(">>>   python convert_hf_to_gguf.py hf_export/depth30 --outfile adaptive_slm_depth30.gguf")
    print(">>>   llama-quantize adaptive_slm_depth30.gguf adaptive_slm_depth30-q4_k_m.gguf Q4_K_M")

if __name__ == "__main__":
    run_kaggle_training()
