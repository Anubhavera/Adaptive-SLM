# AdaptiveSLM — Project Intelligence

## What This Is
C++/Rust hybrid on-device SLM inference engine targeting sub-512MB devices.
Three novel research contributions: PAKD, ACC, SCPD.
Model: Qwen2.5-0.5B Q4_K_M — benchmarked at 11.9 tok/s, 463MB RAM on CPU.

## Architecture
- **C++17 core** (`core/`): llama.cpp-based inference, ACC module, SIMD embeddings, user profiles
- **Rust async wrapper** (`rust-wrapper/`): SCPD cache (SQLite), Tavily web search, FFI bridge
- **Python training** (`training/`): MobileLLM-style architecture, KD, PAKD loss

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
- `training/train.py` — PAKD pipeline; run with `--phase` arg; vocab_size=151665

## Benchmark Results (measured, not claimed)
| Model | Tok/s | RAM | File | Sub-512MB |
|-------|-------|-----|------|-----------|
| Qwen2.5-0.5B Q4_K_M | 11.9 | 463MB | 469MB | YES ✓ |
| Gemma 3 1B Q4_K_M | 6.8 | 762MB | 769MB | NO |
| Gemma 4 (any) | N/A | >2GB | >4.7GB | NO |

## Known Issues / Next Steps
- `SIMD: scalar fallback` shown in bench — AVX2 macro not propagating to bench binary
- Rust wrapper not yet linked to built C++ library (FFI untested end-to-end)
- sqlite-vec extension not bundled — cache search is O(n) brute force
- No unit tests exist yet
- Training never executed (no GPU available locally; use Kaggle/Colab)
- 40 tok/s target not met on Windows CPU; achievable on Linux with AVX2 properly enabled

## Dependencies
- **C++**: llama.cpp b5233 (fetched via CMake FetchContent — do NOT use bare GGML)
- **Rust**: tokio, rusqlite, reqwest, serde, chrono, tracing
- **Python**: torch, transformers, datasets, accelerate, bitsandbytes
