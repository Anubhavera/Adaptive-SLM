# Training Guide

AdaptiveSLM uses a specialized training pipeline designed to produce a high-performance SLM (~350M params) from scratch or via distillation.

## Hardware Requirements
- **Preferred**: TPU v5e-8 (Kaggle) - Fastest (~4h total)
- **Alternative**: 2x T4 GPUs (Kaggle) - Good (~14h total)
- **Minimum**: 24GB VRAM GPU

## Pipeline Overview

1.  **Data Preparation**
    - Filtered web corpus (FineWeb-Edu)
    - Instruction tuning data (Alpaca/Dolly)
    - Profile-annotated data for PAKD

2.  **Teacher Setup**
    - We use **Qwen2.5-7B-Instruct** as the teacher.
    - Loaded in 4-bit quantization to fit in memory while retaining IQ.

3.  **Training Phases**
    - **Phase 1: Pre-training**: Learning general language structure.
    - **Phase 2: Distillation**: Minimizing KL divergence between Student and Teacher logits.
    - **Phase 3: PAKD**: Fine-tuning with profile-weighted loss.

5.  **Quantization**
    - Export to GGUF format (Q4_K_M) for deployment.

## Running on Kaggle

We provide a complete Jupyter notebook for one-click training on Kaggle.

1.  Upload `kaggle/adaptive_slm_training.ipynb` to Kaggle.
2.  Select **Accelerator**: TPU v5e-8 or GPU T4 x2.
3.  Run all cells.

The notebook handles:
- Dependency installation
- Dataset streaming (no massive downloads)
- Teacher model loading
- Training loop with mixed precision
- Checkpoint saving

## Profile-Aware Knowledge Distillation (PAKD)

Our custom loss function allows the model to "specialize" during training without separate fine-tuning runs.

```python
loss = (1 - alpha) * CE_Loss + alpha * KL_Div(Student, Teacher)
weighted_loss = loss * profile_weight_matrix
```

This biases the model to perform better on topics relevant to the target profiles (e.g., Coding, Science) while maintaining general capability.
