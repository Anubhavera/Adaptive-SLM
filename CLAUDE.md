# AdaptiveSLM — Project Intelligence

## What This Is
C++/Rust hybrid on-device SLM inference engine targeting sub-512MB devices.
Three novel research contributions: PAKD, ACC, SCPD.
Model: Qwen2.5-0.5B Q4_K_M — benchmarked at 11.9 tok/s, 463MB RAM on CPU.

## Architecture
- **C++17 core** (`core/`): llama.cpp-based inference, ACC module, SIMD embeddings, user profiles
- **Rust async wrapper** (`rust-wrapper/`): SCPD cache (SQLite), Tavily web search, FFI bridge (tested end-to-end), PAKD profile→depth model-family selector (`family.rs`)
- **Python training** (`training/`): AdaptiveSLM v2 — deep-thin dense arch (30L × 768, GQA 12/4, RoPE θ=1M, tied embeddings, ~305M params), MatFormer-style elastic exits at layers 15/22, KD from Qwen3-4B-Instruct-2507 (4-bit), PAKD loss

## Build (Windows — MSYS2 UCRT64)
```powershell
# C++ core — must use Ninja + explicit GCC paths (MSVC not supported)
cd core
mkdir build; cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="C:\msys64\ucrt64\bin\gcc.exe" `
  -DCMAKE_CXX_COMPILER="C:\msys64\ucrt64\bin\g++.exe" `
  -DCMAKE_MAKE_PROGRAM="C:\msys64\ucrt64\bin\ninja.exe"
cmake --build . --parallel

# Run benchmark
.\adaptive_slm_bench.exe "..\..\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

```bash
# Linux build (simpler)
cd core && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

```bash
# Rust wrapper
cd rust-wrapper && cargo build --release

# Training (requires CUDA GPU + data files in data/)
pip install -r training/requirements.txt
python training/train.py --phase all --data-dir ./data
```

## Model
- File: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (469MB, Q4_K_M)
- Download: `hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/`
- Benchmarked: 11.9 tok/s, 463MB RAM (Windows CPU-only, no GPU)
- Chat format: ChatML (`<|im_start|>system/user/assistant<|im_end|>`)

## Naming Conventions
- C API: `aslm_` prefix (e.g., `aslm_init`, `aslm_generate`, `aslm_generate_stream`)
- C++ internals: `aslm::` namespace
- Rust: module-per-subsystem (`cache`, `profile`, `search`, `ffi`)

## Key Implementation Facts
- `core/src/inference.cpp` — real llama.cpp inference; prompt wrapped in ChatML template
- `core/src/context_adapt.cpp` — ACC algorithm; cross-platform (Windows + Linux)
- `core/src/embeddings.cpp` — FNV-1a feature hashing (unigram+bigram), AVX2 dot product
- `rust-wrapper/src/cache.rs` — SCPD cache; FNV-1a embeddings (unigram+bigram+char trigram)
- `rust-wrapper/src/lib.rs` — orchestrates cache → search → inference pipeline
- `training/train.py` — v2 pipeline; `--phase {pretrain,distill,pakd,export,all}`; vocab_size=151936 (Qwen3 tokenizer); checkpoints auto-chain (pakd_finetuned > distilled > pretrained); export writes `checkpoints/hf_export/depth{30,22,15}/` as HF qwen3 format (safetensors, QK-norm on per Qwen3 convention) for llama.cpp conversion
- `training/prepare_cloud_data.py` — streams fineweb-edu + codeparrot (pretrain), alpaca/gsm8k/sciq/ultrachat as ChatML messages (distill), alpaca + profile system prompts (pakd)
- `training/training_notebook.ipynb` — end-to-end Colab workflow (verified JSON, nbformat 4); teacher = Qwen3-4B-Instruct-2507 on T4, Qwen3-1.7B-Instruct-2507 on Kaggle P100
- Architecture decision (Sep 2026): **no MoE at sub-512MB** — Qwen3 ships dense below 4B, MoE needs ≥1B active params; elasticity via MatFormer-style nested depth (export 30L/22L/15L family from one run), PAKD profile → depth choice

## Benchmark Results (measured, not claimed)
| Model | Tok/s | RAM | File | Sub-512MB |
|-------|-------|-----|------|-----------|
| **AdaptiveSLM v2 15L (ours, beginner)** | **85.5** | 145MB | 151MB | YES ✓ |
| **AdaptiveSLM v2 22L (ours, intermediate)** | **71.8** | 170MB | 176MB | YES ✓ |
| **AdaptiveSLM v2 30L (ours, advanced/expert)** | **54.0** | 199MB | 205MB | YES ✓ |
| Qwen2.5-0.5B Q4_K_M (Linux, AVX2) | 22.4 | 463MB | 469MB | YES ✓ |
| Qwen2.5-0.5B Q4_K_M (Windows CPU) | 11.9 | 463MB | 469MB | YES ✓ |
| Gemma 3 1B Q4_K_M | 6.8 | 762MB | 769MB | NO |
| Gemma 4 (any) | N/A | >2GB | >4.7GB | NO |

v2 numbers: same engine (`adaptive_slm_bench`), Linux 8-core AVX2, 64-token generations; ±20% run-to-run variance; speed/RAM are weight-independent — quality TBD until first Colab training run. Family artifacts: `models/adaptive_slm-{15l,22l,30l}-q4_k_m.gguf`. Full chain validated offline: export → transformers 5.x load (exact) → llama.cpp b5260 convert → Q4_K_M quantize → `adaptive_slm_bench` (loads qwen3 arch, ACC 4/4, real inference), plus Rust FFI e2e (generate/stream/ACC/profile, 19/19 tests).

## Known Issues / Next Steps
- sqlite-vec extension not bundled — cache search is O(n) brute force
- Training v2 never executed on GPU yet (use `training/training_notebook.ipynb` on Colab/Kaggle); offline smoke test (35 checks: RoPE/masking/KD/PAKD/export) passes on CPU; end-to-end export→GGUF→bench chain validated with untrained weights
- FIXED (Sep 2026): "SIMD: scalar fallback" was a CMake bug — `set(SIMD_COMPILE_OPTIONS "-mavx2 -mfma")` passed both flags as ONE quoted argument; now a proper list, bench prints `SIMD: AVX2 enabled`
- FIXED (Sep 2026): 40 tok/s target met on Linux AVX2 (all three elastic depths exceed it)
- FIXED (Sep 2026): Rust wrapper FFI now linked + tested end-to-end (19/19 tests incl. generate/stream/ACC/profile against real GGUF); `build.rs` links the SHARED `libadaptive_slm.so` via rpath (static llama/ggml fallback paths kept for Windows)
- FIXED (Sep 2026): `user_profile.cpp::computeRelevance` scored unrelated topics at exactly 0.5 (same as neutral) due to `(sim+1)/2` affine map on FNV embeddings; now matches Rust twin semantics (0.9/0/0.3 + 70/30 combine)
- Windows MSYS2 build: re-verify SIMD fix on MSYS2 (flags now list-correct, expected to work)

## Dependencies
- **C++**: llama.cpp b5260 (fetched via CMake FetchContent — do NOT use bare GGML; b5260 is the minimum with qwen3 arch support)
- **Rust**: tokio, rusqlite, reqwest, serde, chrono, tracing
- **Python**: torch, transformers (>=4.51; 5.x works — Qwen2 arch in 5.x hardcodes attn bias, so export targets qwen3), datasets, accelerate, bitsandbytes, safetensors
