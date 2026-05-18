# AdaptiveSLM — Project Intelligence

## What This Is
C++/Rust hybrid on-device SLM inference engine targeting sub-512MB devices at 40+ tokens/sec.
Three novel research contributions: PAKD, ACC, SCPD.

## Architecture
- **C++17 core** (`core/`): GGML-based inference, ACC module, SIMD embeddings, user profiles
- **Rust async wrapper** (`rust-wrapper/`): SCPD cache (SQLite), Tavily web search, FFI bridge
- **Python training** (`training/`): MobileLLM-style architecture, KD, PAKD loss

## Build
```bash
# C++ core (requires CMake 3.16+, GCC/Clang)
cd core && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Rust wrapper
cd rust-wrapper && cargo build --release

# Training (requires CUDA GPU)
pip install -r requirements.txt
python training/train.py
```

## Model
- File: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (469MB, Q4_K_M quantization)
- Architecture: Qwen2.5-0.5B (494M params), 4-bit quantized via llama.cpp

## Naming Conventions
- C API: `aslm_` prefix (e.g., `aslm_init`, `aslm_generate`)
- C++ internals: `aslm::` namespace
- Rust: module-per-subsystem (`cache`, `profile`, `search`, `ffi`)

## Key Implementation Facts
- `core/src/inference.cpp` — main inference engine using llama.cpp
- `core/src/context_adapt.cpp` — ACC algorithm (fully working, best component)
- `rust-wrapper/src/cache.rs` — SCPD cache with SQLite backend
- `rust-wrapper/src/lib.rs` — orchestrates cache → search → inference pipeline
- `training/train.py` — PAKD training pipeline (MobileLLM + KD + profile-aware loss)

## Known Issues
- Device monitoring in `context_adapt.cpp` reads Linux `/proc/` — needs Windows/macOS support
- Cache embeddings in `cache.rs:compute_embedding()` are simplified hash-based, not semantic
- sqlite-vec extension loading may fail; falls back to O(n) brute-force search
- Training phases in `train.py` need to be uncommented to run

## Dependencies
- **C++**: llama.cpp (fetched via CMake FetchContent, includes GGML)
- **Rust**: tokio, rusqlite, reqwest, serde, chrono, tracing
- **Python**: torch, transformers, datasets, accelerate, bitsandbytes
