# AdaptiveSLM Kaggle Training Guide

## 📊 GPU/TPU Comparison

| Resource | VRAM | Est. Time (10K steps) | Recommendation |
|----------|------|----------------------|----------------|
| **T4 x2** | 30GB | ~8 hours | ✅ Best for distillation |
| **P100** | 16GB | ~12 hours | ✅ Good for pre-training |
| **TPU v5e-8** | 32GB HBM | ~4 hours | ⭐ Fastest, best overall |

---

## 🚀 Quick Start

### 1. Upload Files to Kaggle

Upload the notebook: `kaggle/adaptive_slm_training.ipynb`

### 2. Configure Accelerator

In Kaggle:
1. Go to **Settings** (right sidebar)
2. Select **Accelerator**: `GPU T4 x2` or `TPU v3-8`
3. Enable **Internet** (required for HuggingFace downloads)

### 3. Run Training

The notebook will:
1. Load the MobileLLM-style model architecture (~350M params)
2. Download Qwen-7B teacher (4-bit quantized)
3. Load FineWeb-Edu dataset (streaming)
4. Train with knowledge distillation
5. Save checkpoints to `/kaggle/working/`

---

## ⏱️ Training Time Estimates

### Full Training (50 hrs quota)

| Phase | Steps | T4 x2 | P100 | TPU |
|-------|-------|-------|------|-----|
| Pre-training | 10K | 8h | 12h | 4h |
| Distillation | 5K | 4h | 6h | 2h |
| PAKD Fine-tune | 2K | 2h | 3h | 1h |
| **Total** | **17K** | **14h** | **21h** | **7h** |

With 50 hours, you can complete:
- **T4 x2**: 3 full training runs (for ablations)
- **P100**: 2 full training runs
- **TPU**: 7 full training runs!

---

## 📁 Output Files

After training, download from `/kaggle/working/`:
- `adaptive_slm_final.pt` - Final model checkpoint
- `checkpoint_*.pt` - Intermediate checkpoints

---

## 🔧 Converting to GGUF

After downloading the checkpoint:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert PyTorch to GGUF
python convert-hf-to-gguf.py /path/to/checkpoint --outtype q4_k_m --outfile adaptive_slm.gguf
```

---

## 📈 Expected Results

After training, you should achieve:

| Metric | Baseline (Qwen 0.5B) | Target |
|--------|----------------------|--------|
| MMLU | 43.7% | **45%+** |
| RAM (inference) | 600MB | **<512MB** |
| Params | 500M | **~350M** |
