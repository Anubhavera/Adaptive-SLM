# AdaptiveSLM — Complete Engineering Audit & Repair Plan

## Context

**Problem**: The AdaptiveSLM project claims to be a "High-Throughput On-Device Inference Engine" with three novel research contributions (PAKD, ACC, SCPD), targeting sub-512MB devices at 40+ tokens/sec. A thorough source-code audit reveals that **the core inference engine is entirely fake** — it returns hardcoded demo strings instead of running actual model inference. Multiple other components are placeholders masquerading as implementations.

**Why this matters**: Every claim on the README, docs, and presumably a resume is technically indefensible until real inference works. The project has good architectural scaffolding and some genuinely well-implemented subsystems (ACC, SCPD priority formula, SIMD), but the foundation — actual model inference — is missing.

**Intended outcome**: Transform this from a demo/mockup into a working on-device inference engine with real benchmarks, defensible claims, and research-grade quality.

---

## Phase 0: .claude/ Folder Setup

### 0.1 Create CLAUDE.md (project root)

**File**: `C:\Anubhav\Adaptive-SLM\CLAUDE.md`

Contents:
- Project: AdaptiveSLM — C++/Rust hybrid on-device SLM inference engine
- Architecture: C++17 core (GGML tensor ops, ACC, embeddings, user profiles) + Rust async wrapper (SCPD cache, Tavily search, FFI bridge) + Python training pipeline (PAKD, KD, MobileLLM-style architecture)
- Build: `cmake` for core/, `cargo build --release` for rust-wrapper/, `pip install -r requirements.txt` for training/
- **CRITICAL TRUTH**: `core/src/inference.cpp:183` returns a hardcoded demo string — inference is NOT real. Model weights are never loaded into GGML. The tokenizer in `embeddings.cpp` is hash-based (`hash % 10000`), not BPE. Embeddings use `sin(id*1337+i*7)*0.1` — not real vectors.
- Model file: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (469MB, Q4_K_M)
- Naming: C API uses `aslm_` prefix. Rust modules mirror C++ structure.
- Platform: Device monitoring in `context_adapt.cpp` reads Linux `/proc/` — won't work on Windows
- CMake flags (`-O3 -march=native -ffast-math`) are GCC-only — won't compile with MSVC
- Training phases in `train.py` are ALL commented out (lines 767-776)
- `UserProfile::fromJson()` in `user_profile.cpp` never parses interests array — always returns empty

### 0.2 Create .claude/settings.json

**File**: `C:\Anubhav\Adaptive-SLM\.claude\settings.json`

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Glob(**)",
      "Grep(**)",
      "Bash(cmake *)",
      "Bash(cargo *)",
      "Bash(python *)",
      "Bash(git status*)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(head *)",
      "Bash(wc *)"
    ]
  }
}
```

### 0.3 Create Memory Files

**File**: `C:\Users\User\.claude\projects\C--Anubhav-Adaptive-SLM\memory\MEMORY.md`

```
- [Project Overview](project_overview.md) — AdaptiveSLM audit: inference is fake, ACC works, SCPD has fake embeddings
- [User Profile](user_profile.md) — Capstone/research project owner, needs technically defensible claims
- [Audit Feedback](audit_feedback.md) — Brutally honest audit requested; never assume features work without code proof
```

Plus individual memory files:
- `project_overview.md` — type: project, summarizes repo state
- `user_profile.md` — type: user, student building capstone with research ambitions
- `audit_feedback.md` — type: feedback, user wants precise technical honesty

---

## Phase 1: Repository Discovery (READ-ONLY)

### 1.1 Verified Repository Map

```
C:\Anubhav\Adaptive-SLM\
├── README.md                         # Project overview, benchmark claims
├── core/                             # C++ inference engine
│   ├── CMakeLists.txt                # Build: GGML FetchContent, SIMD detection
│   ├── include/
│   │   ├── adaptive_slm.h            # Public C API (165 lines) — COMPLETE
│   │   ├── context_adapt.h           # ACC interface (78 lines) — COMPLETE
│   │   ├── embeddings.h              # SIMD embedding engine (94 lines) — COMPLETE
│   │   └── user_profile.h            # User profile (74 lines) — COMPLETE
│   ├── src/
│   │   ├── inference.cpp             # FAKE — returns demo string (241 lines)
│   │   ├── context_adapt.cpp         # REAL — ACC algorithm works (186 lines)
│   │   ├── embeddings.cpp            # MIXED — SIMD real, embeddings fake (221 lines)
│   │   └── user_profile.cpp          # MOSTLY REAL — fromJson broken (201 lines)
│   └── benchmark/
│       └── bench.cpp                 # Framework works, inference benchmarks meaningless (223 lines)
├── rust-wrapper/                     # Rust async wrapper
│   ├── Cargo.toml                    # deps: tokio, rusqlite, reqwest, serde
│   └── src/
│       ├── lib.rs                    # Main wrapper — orchestrates cache→search→C++ (385 lines)
│       ├── ffi.rs                    # FFI bindings — COMPLETE (92 lines)
│       ├── cache.rs                  # SCPD cache — logic REAL, embeddings FAKE (363 lines)
│       ├── profile.rs               # User profile — COMPLETE (155 lines)
│       └── search.rs                # Tavily + DDG search — COMPLETE (267 lines)
├── training/
│   ├── train.py                      # Full MobileLLM arch + PAKD — ALL COMMENTED OUT (783 lines)
│   └── prepare_data.py              # Data pipeline — placeholder mode works (351 lines)
├── benchmarks/
│   ├── mmlu_eval.py                  # MMLU evaluator — needs real model (261 lines)
│   └── compare.py                   # Comparisons — HARDCODED results (131 lines)
├── storage/
│   └── schema.sql                   # DB schema — COMPLETE, sqlite-vec commented out (74 lines)
├── docs/
│   ├── index.md, architecture.md, api_reference.md, training.md
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf  # 469MB Q4_K_M model file — EXISTS
```

### 1.2 Dependency Graph

```
Rust Wrapper (lib.rs)
  ├── FFI Bridge (ffi.rs) ──→ C++ Core
  │                              ├── inference.cpp ──→ GGML (allocated, unused)
  │                              ├── context_adapt.cpp (standalone, works)
  │                              ├── embeddings.cpp ──→ SIMD intrinsics
  │                              └── user_profile.cpp ──→ embeddings.cpp
  ├── SCPD Cache (cache.rs) ──→ rusqlite + sqlite-vec (fallback)
  ├── User Profile (profile.rs)
  └── Web Search (search.rs) ──→ Tavily API + DuckDuckGo

Training Pipeline (train.py)
  ├── PyTorch + Transformers
  ├── Qwen2.5-7B teacher model
  └── Custom MobileLLM architecture (GQA, SwiGLU, RMSNorm)
```

### 1.3 Execution Flow (Current — BROKEN)

```
User prompt → Rust generate() → check SCPD cache (fake embeddings)
  → optional Tavily search → build augmented prompt
  → FFI call to aslm_generate() → RETURNS HARDCODED STRING
  → store result in SCPD cache → return to user
```

### 1.4 Execution Flow (Target — WORKING)

```
User prompt → Rust generate() → check SCPD cache (real embeddings)
  → optional Tavily search → build augmented prompt
  → FFI call to aslm_generate()
    → llama_tokenize(prompt)
    → llama_decode(tokens) [forward pass through quantized model]
    → llama_sampler_sample() [autoregressive loop]
    → llama_token_to_piece() [detokenize]
  → store result in SCPD cache → return to user
```

---

## Phase 2: Claim Verification Matrix

### A. "40+ tokens/sec" — VERDICT: UNVERIFIABLE (NO REAL INFERENCE)

| Check | Status | Evidence |
|-------|--------|----------|
| Benchmarking implemented? | Framework only | `bench.cpp` measures demo string copy time |
| Real profiling? | NO | No profiler integration, no flame graphs |
| Batching? | NO | Single-token generation only |
| SIMD? | YES (helper only) | AVX2/NEON dot product in `embeddings.cpp:118-178` |
| Threading? | NO | Single-threaded generation |
| mmap? | DECLARED, UNUSED | `use_mmap` param exists, never used (`inference.cpp:41`) |
| KV cache optimization? | NO | No KV cache exists at all |
| Speculative decoding? | NO | Not implemented |
| Token streaming? | NO | Batch return only |
| Hardware target? | UNCLEAR | `-march=native` implies host CPU |

**Reality**: llama.cpp achieves 30-60 tok/s on Q4 0.5B models on modern x86 CPUs. The 40+ target is achievable IF llama.cpp is properly integrated.

### B. "sub-512MB devices" — VERDICT: PLAUSIBLE BUT UNPROVEN

| Component | Estimated Memory |
|-----------|-----------------|
| Model weights (Q4_K_M, mmap) | ~150-250MB resident |
| Model weights (Q4_K_M, no mmap) | ~469MB |
| KV cache (2048 ctx, 24 layers) | ~24MB |
| GGML compute buffers | ~32MB |
| SQLite + SCPD cache | ~5-20MB |
| Tokenizer vocab | ~2MB |
| Rust runtime | ~5MB |
| **Total with mmap** | **~220-330MB** ✓ |
| **Total without mmap** | **~540-550MB** ✗ |

**Reality**: Achievable WITH mmap on Linux/Android. Fails without mmap. Need Q3_K_S (~380MB file) for non-mmap sub-512MB.

### C. "GGML 4-bit quantized Qwen2.5-0.5B" — VERDICT: FILE EXISTS, NOT LOADED

| Check | Status | Evidence |
|-------|--------|----------|
| GGML integrated? | PARTIALLY | `ggml.h` included, `ggml_init()` called at `inference.cpp:86` |
| llama.cpp used? | NO | Only bare GGML (tensor lib), not llama.cpp (model loader) |
| Model file exists? | YES | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` present |
| Model loaded? | NO | File size read only (`inference.cpp:105-123`) |
| Weights in tensors? | NO | 300MB GGML context allocated but empty |
| Tokenization works? | NO | Hash-based fake tokenizer used |

**Critical insight**: GGML alone CANNOT load GGUF files. GGML is a tensor math library. **llama.cpp** is needed for GGUF loading, tokenization, KV cache, and inference. This is the fundamental architectural gap.

### D. "Profile-Aware Knowledge Distillation (PAKD)" — VERDICT: IMPLEMENTED IN CODE, NEVER EXECUTED

| Check | Status | Evidence |
|-------|--------|----------|
| Training architecture? | YES | MobileLLM-style in `train.py:62-207` |
| KD loss? | YES | `KnowledgeDistillationLoss` at `train.py:209-237` |
| PAKD loss? | YES | `PAKDLoss` at `train.py:239-280` with profile weights |
| Training loop? | YES | `AdaptiveSLMTrainer` with pretrain/distill/pakd_finetune |
| Actually runs? | NO | All training calls commented out at `train.py:767-776` |
| Exported model? | NO | `export_gguf()` is a print statement |

**Reality**: The PAKD implementation is the strongest novel contribution. The loss function is mathematically sound. But it has NEVER been executed. No trained model exists.

### E. "Adaptive Context Compression (ACC)" — VERDICT: FULLY IMPLEMENTED ✓

| Check | Status | Evidence |
|-------|--------|----------|
| Resource scoring? | YES | Weighted RAM/CPU/battery at `context_adapt.cpp:19-45` |
| Context adjustment? | YES | Linear interpolation + power-of-2 snap at `context_adapt.cpp:47-79` |
| EMA smoothing? | YES | Factor 0.3 at `context_adapt.cpp:81-101` |
| Emergency handling? | YES | <20% RAM → min_context at `context_adapt.cpp:56-58` |
| Device state reading? | LINUX ONLY | `/proc/meminfo`, `/proc/stat` at `context_adapt.cpp:121-183` |
| Benchmarked? | YES | 4 scenarios tested in `bench.cpp:41-77` |

**This is the project's best-implemented component.** ACC works correctly and is well-tested. The only issue is Linux-only device monitoring.

### F. "Semantic Cache with Priority Decay (SCPD)" — VERDICT: LOGIC REAL, EMBEDDINGS FAKE

| Check | Status | Evidence |
|-------|--------|----------|
| SQLite backend? | YES | `cache.rs` creates tables, indices |
| Priority formula? | YES | `P = α·Recency + β·Relevance + γ·Profile` at `cache.rs` update_priority |
| Decay logic? | YES | `priority *= 0.95` per day in `decay_and_evict()` |
| Eviction? | YES | Entries below threshold removed |
| Semantic similarity? | FAKE | `compute_embedding()` uses `sin(c as u32)*0.1` at `cache.rs:307-315` |
| sqlite-vec? | FAILS | Extension load fails, falls back to O(n) brute force |
| Access boost? | YES | `1 + 0.1·ln(access_count+1)` multiplier |

**Reality**: The cache data structure and eviction logic are well-designed. But semantic similarity is completely fake because embeddings are hash-based, not learned. With real embeddings, SCPD would work as described.

---

## Phase 2.5: Gemma Migration Feasibility

### Current State Assessment

Since inference is entirely fake (no Qwen-specific code exists in C++), migration cost is about the same as building inference from scratch — which must happen regardless.

### Model Comparison: Qwen2.5-0.5B vs Gemma Candidates

| Property | Qwen2.5-0.5B | Gemma 3 1B | Gemma 3n E2B |
|----------|-------------|------------|--------------|
| Parameters | 494M | 1B | 2B (selective layers) |
| Q4_K_M size | ~469MB | ~700MB | ~1.2GB |
| Context window | 32K | 32K | 32K |
| Tokenizer | tiktoken BPE (151K vocab) | SentencePiece (256K vocab) | SentencePiece (256K vocab) |
| llama.cpp support | Mature | Mature | Newer, growing |
| Mobile optimization | Standard | Standard | **Purpose-built for mobile** |
| ARM NEON perf | Good | Good | **Optimized (selective layers)** |
| MMLU (approx) | 43.7% | ~50% | ~55% |
| KV cache / token | ~0.5MB/1K tokens | ~1MB/1K tokens | Variable (selective) |
| Android deploy | Via llama.cpp | Via llama.cpp | Via MediaPipe / llama.cpp |

### Migration Analysis

**A. Migration Difficulty: LOW** (2/10)
- No Qwen-specific inference code exists (it's all fake)
- llama.cpp handles all architecture differences internally
- Tokenizer is abstracted by llama.cpp
- Only changes needed: model path string, prompt template format

**B. Recommended Candidates:**
1. **Primary: Keep Qwen2.5-0.5B** — Smallest, fits sub-512MB with mmap, proven llama.cpp support
2. **Secondary: Gemma 3 1B Q4** — Better quality but ~700MB, borderline for 512MB target
3. **Stretch: Gemma 3n E2B** — Best mobile story but 1.2GB even quantized, exceeds budget
4. **Watch: Future Gemma sub-1B models** — If Google releases a 500M Gemma, it wins

**C. Expected Memory Impact:**
- Qwen2.5-0.5B Q4: ~469MB file, ~220MB resident with mmap ✓
- Gemma 3 1B Q4: ~700MB file, ~350MB resident with mmap — tight but possible
- Gemma 3n E2B Q4: ~1.2GB file — EXCEEDS 512MB even with mmap

**D. Expected tok/s Impact:**
- Qwen2.5-0.5B: 40-60 tok/s on x86, 15-25 on ARM
- Gemma 3 1B: 20-35 tok/s on x86, 8-15 on ARM (2x params)
- Gemma 3n: Variable due to selective layers, potentially 15-25 on mobile

**E. Required Architectural Changes:**
1. Change model path in config (trivial)
2. Change prompt template in Rust wrapper (trivial — `<|im_start|>` → `<start_of_turn>`)
3. No C++ changes if using llama.cpp (handles internally)
4. Update training pipeline teacher model reference
5. Update vocab_size in TrainingConfig

**F. Recommendation: NOT WORTH IT YET**
- Qwen2.5-0.5B is the only model that fits the sub-512MB claim
- Gemma models are too large for the stated target
- The project's credibility depends on the sub-512MB claim being real
- Adding Gemma as a "supported alternative" is fine, but Qwen should remain primary
- Revisit when Gemma releases a sub-500M parameter model
- **For interview positioning**: Mention Gemma compatibility as "the architecture supports model-agnostic inference via llama.cpp, tested with both Qwen and Gemma families"

### Model-Swap Architecture (implement for credibility)

Add a `ModelConfig` abstraction:
```cpp
struct aslm_model_config {
    const char* model_path;       // path to any GGUF file
    const char* prompt_template;  // e.g. "chatml", "gemma", "llama"
    int32_t context_size;
};
```
This lets the project claim model-agnostic inference, which is technically stronger than being tied to one model.

---

## Phase 3: Code Quality & Systems Review

### 3.1 Critical Bugs

| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
| 1 | **CRITICAL** | `inference.cpp` | 183 | `aslm_generate()` returns hardcoded demo string |
| 2 | **CRITICAL** | `inference.cpp` | 86 | GGML context allocated (300MB) but never used for compute |
| 3 | **HIGH** | `embeddings.cpp` | 76 | Embeddings use `sin(id*1337+i*7)*0.1` — not semantic |
| 4 | **HIGH** | `cache.rs` | 307-315 | Cache embeddings use `sin(c as u32)*0.1` — not semantic |
| 5 | **HIGH** | `user_profile.cpp` | 130-161 | `fromJson()` never parses interests array |
| 6 | **HIGH** | `context_adapt.cpp` | 121-183 | Linux-only `/proc/` reads — crashes/returns zeros on Windows |
| 7 | **MEDIUM** | `CMakeLists.txt` | 9 | `-march=native -ffast-math` are GCC-only flags |
| 8 | **MEDIUM** | `CMakeLists.txt` | 21 | GGML fetched from `master` — no version pinning |
| 9 | **MEDIUM** | `train.py` | 767-776 | All training phases commented out |
| 10 | **MEDIUM** | `train.py` | 737-747 | `export_gguf()` just prints a shell command |
| 11 | **LOW** | `compare.py` | 61 | `tokens_per_sec = 50.0` hardcoded |
| 12 | **LOW** | `schema.sql` | ~60 | sqlite-vec virtual table creation commented out |

### 3.2 Memory Safety Issues

- `inference.cpp:159-160`: Raw pointer cast `reinterpret_cast<Context*>(handle)` — no null check on `ctx->acc`before use (UB if acc initialization failed silently)
- `user_profile.cpp:34`: `const_cast` removing constness from mutable member — design smell but technically OK
- `ffi.rs`: `NonNull<ffi::aslm_context>` — correct usage, but `Drop` impl should verify context is valid
- Rust cache: `bytemuck` casts for `f32 ↔ [u8]` — correct if alignment is maintained

### 3.3 Dead Code / Fake Abstractions

- `inference.cpp:41`: `void* model_mmap = nullptr` — declared, never used
- `inference.cpp:42`: `size_t model_size = 0` — tracks file size but never loaded
- `embeddings.h/cpp`: Entire `EmbeddingEngine::embed()` pipeline produces meaningless vectors
- `bench.cpp:83-131`: Inference benchmark measures string-copy speed, not model performance

### 3.4 Performance Bottlenecks (when inference is real)

- `cache.rs` search: O(n) scan of all entries for cosine similarity — needs sqlite-vec or FAISS
- `embeddings.cpp` SIMD: Only used for dot product, not for embedding generation
- `context_adapt.cpp:142-164`: CPU measurement reads `/proc/stat` twice per call — first call always returns incorrect value (prev_idle=0)

---

## Phase 4: Gap Analysis

### FULLY IMPLEMENTED ✓
1. **ACC algorithm** — `context_adapt.cpp` — Resource scoring, EMA smoothing, power-of-2 snapping, 4 test scenarios
2. **C API surface** — `adaptive_slm.h` — Clean, well-documented public interface
3. **FFI bridge** — `ffi.rs` — Complete Rust↔C++ bindings
4. **SCPD priority formula** — `cache.rs` — Decay, eviction, access boost, all correct
5. **SCPD SQLite backend** — `cache.rs` — Tables, indices, CRUD operations
6. **User Profile (Rust)** — `profile.rs` — Relevance, prompt modifiers, JSON serialization
7. **User Profile (C++)** — `user_profile.cpp` — Relevance computation, prompt modification (except fromJson)
8. **Web search integration** — `search.rs` — Tavily + DuckDuckGo fallback
9. **SIMD dot product** — `embeddings.cpp:118-178` — AVX2, NEON, scalar fallback
10. **Benchmark framework** — `bench.cpp` — Memory, ACC, SIMD benchmarks work
11. **Training architecture** — `train.py` — MobileLLM with GQA, SwiGLU, RMSNorm, PAKD loss
12. **Data preparation** — `prepare_data.py` — Quality filters, domain detection, profile annotation

### PARTIALLY IMPLEMENTED ⚠
1. **Embeddings engine** — SIMD ops work; actual embeddings are fake
2. **User Profile C++ JSON** — `toJson()` works; `fromJson()` doesn't parse interests
3. **Device state monitoring** — Works on Linux only; no Windows/macOS/Android
4. **SCPD semantic search** — Cache logic works; similarity is meaningless (fake embeddings)
5. **Benchmark suite** — Framework runs; inference benchmark measures nothing real
6. **Model initialization** — GGML context allocated; no weights loaded

### PLACEHOLDER / NON-FUNCTIONAL ✗
1. **Inference engine** — `aslm_generate()` returns `"[AdaptiveSLM] Demo response for: "`
2. **Model loading** — Checks file exists, never loads weights
3. **Tokenization** — `hash % 10000` word tokenizer, not BPE
4. **GGUF export** — `export_gguf()` prints a command string
5. **Training execution** — All phases commented out
6. **Benchmark comparisons** — `compare.py` uses hardcoded `50.0 tok/s`

### COMPLETELY MISSING ✗✗
1. **llama.cpp integration** — Required for GGUF loading, tokenization, KV cache, inference
2. **Real tokenizer** — BPE/SentencePiece for the target model
3. **KV cache** — No key-value cache implementation
4. **Token streaming** — No callback/stream API
5. **Real embedding model** — No all-MiniLM or similar for SCPD cache
6. **Windows/macOS device monitoring** — Only Linux `/proc/` paths
7. **Unit tests** — Zero test files (only bench.cpp)
8. **Integration tests** — None
9. **CI/CD** — No GitHub Actions, no Dockerfile
10. **Android build** — No NDK config, no cross-compilation
11. **Memory profiling** — No Valgrind/ASan integration
12. **Model-agnostic config** — Hardcoded paths and assumptions
13. **Prompt template system** — No chat template abstraction

---

## Phase 5: Prioritized Repair Roadmap

### Priority 0 — Make Inference REAL (Critical Path)

**Goal**: Replace fake inference with working llama.cpp-based generation

#### Step 0.1: Replace GGML with llama.cpp in CMake

**File**: `core/CMakeLists.txt`
**Action**: Replace GGML FetchContent with llama.cpp

```cmake
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG b5460  # Pin to stable release
)
FetchContent_MakeAvailable(llama_cpp)
```

Change link target from `ggml` to `llama common` (llama.cpp includes ggml internally).

**Difficulty**: Easy (30 min)
**Impact**: Enables all subsequent steps

#### Step 0.2: Rewrite inference.cpp with llama.cpp

**File**: `core/src/inference.cpp`
**Action**: Complete rewrite of Context struct and all API functions

New Context struct:
```cpp
struct Context {
    llama_model* model = nullptr;
    llama_context* llama_ctx = nullptr;
    llama_sampler* sampler = nullptr;
    aslm_init_params init_params;
    std::unique_ptr<ContextAdapter> acc;
    int32_t current_context_size = 512;
    const UserProfile* user_profile = nullptr;
    uint64_t peak_memory = 0;
    bool is_initialized = false;
};
```

New `aslm_init()`:
1. `llama_model_params mparams = llama_model_default_params();`
2. `mparams.use_mmap = params->use_mmap;`
3. `model = llama_model_load_from_file(params->model_path, mparams);`
4. `llama_context_params cparams = llama_context_default_params();`
5. `cparams.n_ctx = params->max_context;`
6. `cparams.n_threads = params->n_threads ? params->n_threads : std::thread::hardware_concurrency();`
7. `llama_ctx = llama_init_from_model(model, cparams);`
8. Set up sampler chain: temperature → top-k → top-p → repetition penalty

New `aslm_generate()`:
1. `llama_tokenize(model, prompt, tokens, max_tokens, true, true)` — tokenize with BOS
2. `llama_decode(llama_ctx, llama_batch_get_one(tokens, n_tokens))` — process prompt
3. Autoregressive loop:
   - `llama_sampler_sample(sampler, llama_ctx, -1)` — sample next token
   - Check for EOS token
   - `llama_decode()` with single token
   - `llama_token_to_piece()` — detokenize
   - Append to output buffer
4. Return token count

New `aslm_free()`:
1. `llama_sampler_free(sampler)`
2. `llama_free(llama_ctx)`
3. `llama_model_free(model)`

**Difficulty**: Medium (2-4 hours — llama.cpp API is well-documented)
**Impact**: TRANSFORMS PROJECT FROM DEMO TO REAL

#### Step 0.3: Wire ACC into llama.cpp context

**File**: `core/src/inference.cpp`
**Action**: After ACC computes new context size, resize KV cache

When `aslm_update_device_state()` triggers ACC context change:
- Call `llama_kv_cache_seq_rm(ctx, -1, new_size, -1)` to trim KV cache
- Store new effective size for future generations

**Difficulty**: Easy (30 min)
**Impact**: Makes ACC meaningful (currently adjusts a number nobody reads)

#### Step 0.4: Update benchmark to use real inference

**File**: `core/benchmark/bench.cpp`
**Action**: Verify `benchmark_inference()` now returns real text and measures actual tok/s

After Step 0.2, the existing benchmark framework should automatically produce real numbers since it calls `aslm_generate()`. Verify:
- Output text is coherent (not demo string)
- tok/s is in expected range (30-60 on x86)
- Memory usage matches estimates

**Difficulty**: Trivial (15 min)
**Impact**: First real benchmark numbers

### Priority 1 — Platform & Bug Fixes

#### Step 1.1: Cross-platform device monitoring

**File**: `core/src/context_adapt.cpp`
**Action**: Add `#ifdef _WIN32` and `#ifdef __APPLE__` blocks

Windows implementation:
```cpp
#ifdef _WIN32
#include <windows.h>
MEMORYSTATUSEX memInfo;
memInfo.dwLength = sizeof(memInfo);
GlobalMemoryStatusEx(&memInfo);
state.total_ram_bytes = memInfo.ullTotalPhys;
state.available_ram_bytes = memInfo.ullAvailPhys;
// CPU: GetSystemTimes() delta
// Battery: GetSystemPowerStatus()
#endif
```

**Difficulty**: Medium (1-2 hours)
**Impact**: ACC works on Windows

#### Step 1.2: Fix CMake for MSVC

**File**: `core/CMakeLists.txt`
**Action**: Use generator expressions

```cmake
target_compile_options(adaptive_slm PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native -ffast-math>
)
```

**Difficulty**: Easy (20 min)
**Impact**: Project builds on Windows with MSVC

#### Step 1.3: Fix UserProfile::fromJson()

**File**: `core/src/user_profile.cpp`, lines 130-161
**Action**: Add interests array parsing

Parse the JSON `"interests":["a","b","c"]` pattern with string find/substr, or add nlohmann/json as a lightweight dependency.

**Difficulty**: Easy (30 min)
**Impact**: Profile roundtripping works correctly

#### Step 1.4: Pin GGML/llama.cpp version

**File**: `core/CMakeLists.txt`
**Action**: Change `GIT_TAG master` to a specific release tag (e.g., `b5460`)

**Difficulty**: Trivial (5 min)
**Impact**: Reproducible builds

### Priority 2 — Real Embeddings & Cache

#### Step 2.1: Integrate real embeddings for SCPD

**Option A (recommended)**: Use llama.cpp's built-in embedding mode
- Load a small embedding model (all-MiniLM-L6-v2 in GGUF format, ~22MB)
- Or extract embeddings from the main model's hidden states

**Option B**: ONNX Runtime with all-MiniLM-L6-v2
- Add onnxruntime dependency
- Load 22MB ONNX model
- Run inference for 384-dim embeddings

**Files to modify**:
- `core/src/embeddings.cpp` — Replace `poolEmbeddings()` with real embedding extraction
- `rust-wrapper/src/cache.rs` — Replace `compute_embedding()` with FFI call to C++ engine

**Difficulty**: Medium (2-4 hours)
**Impact**: SCPD cache actually performs semantic matching

#### Step 2.2: Install sqlite-vec properly

**File**: `rust-wrapper/src/cache.rs`
**Action**: Bundle sqlite-vec as a static library or use the `sqlite-vec` Rust crate

```toml
[dependencies]
sqlite-vec = "0.1"  # or bundle the .so/.dll
```

**Difficulty**: Easy-Medium (1 hour)
**Impact**: Cache search goes from O(n) to O(log n)

### Priority 3 — Training Pipeline

#### Step 3.1: Uncomment and fix training

**File**: `training/train.py`
**Actions**:
1. Uncomment lines 767-776
2. Fix `vocab_size` from 32000 to 151665 (Qwen2.5 tokenizer)
3. Implement real `export_gguf()` using llama.cpp's `convert_hf_to_gguf.py`
4. Test with `--placeholder-only` data first
5. Run on Kaggle/Colab for actual training (needs GPU)

**Difficulty**: Medium (1-2 days for full training pipeline)
**Impact**: Defensible PAKD claim

### Priority 4 — Resume Credibility

#### Step 4.1: Real benchmark numbers

**Files**: `benchmarks/compare.py`, `core/benchmark/bench.cpp`
**Action**: Replace all hardcoded values with measured results from real inference

Must produce:
- Actual tok/s on specified hardware
- Actual memory usage (RSS) during inference
- Actual MMLU score from `mmlu_eval.py`
- Latency distribution (p50, p95, p99)

#### Step 4.2: Model-agnostic architecture

**File**: New `core/include/model_config.h`
**Action**: Create config struct supporting Qwen, Gemma, LLaMA prompt templates

```cpp
enum aslm_prompt_format { CHATML, GEMMA, LLAMA3, PLAIN };
struct aslm_model_config {
    const char* model_path;
    aslm_prompt_format format;
    int32_t max_context;
};
```

**Difficulty**: Easy (1 hour)
**Impact**: "Model-agnostic inference engine" claim becomes real

#### Step 4.3: Add unit tests

**Files**: New test files
- `core/tests/test_acc.cpp` — Google Test, verify all ACC scenarios
- `core/tests/test_inference.cpp` — Verify real generation, not demo
- `rust-wrapper/tests/cache_test.rs` — SCPD store/search/evict cycle
- `tests/integration_test.py` — End-to-end: prompt → real response

**Difficulty**: Medium (1-2 days)
**Impact**: Professional-grade project

### Priority 5 — Research-Grade Polish

#### Step 5.1: Token streaming API

Add callback-based streaming to `aslm_generate()`:
```cpp
typedef void (*aslm_token_callback)(const char* token, void* user_data);
int32_t aslm_generate_stream(aslm_context*, const char* prompt,
    const aslm_gen_params*, aslm_token_callback, void* user_data);
```

#### Step 5.2: Android cross-compilation

Add NDK toolchain file for CMake:
```cmake
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_NDK /path/to/ndk)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
```

#### Step 5.3: Memory profiling integration

Add ASan/MSan build option:
```cmake
option(ENABLE_SANITIZERS "Enable address/memory sanitizers" OFF)
if(ENABLE_SANITIZERS)
    target_compile_options(adaptive_slm PRIVATE -fsanitize=address,undefined)
    target_link_options(adaptive_slm PRIVATE -fsanitize=address,undefined)
endif()
```

#### Step 5.4: CI/CD with GitHub Actions

Create `.github/workflows/build.yml`:
- Build C++ on Ubuntu and Windows
- Build Rust wrapper
- Run bench.cpp
- Run unit tests
- Lint Python

---

## Phase 6: Project Intelligence Files

### 6.1 Create PROJECT_RESEARCHER.md

**File**: `C:\Anubhav\Adaptive-SLM\PROJECT_RESEARCHER.md`

Contents outline:
- Architecture overview with implementation truth map
- Three novel contributions analysis (PAKD strongest, ACC most complete, SCPD needs real embeddings)
- Defensible claims vs exaggerated claims
- Gemma migration analysis
- Memory budget calculations
- Performance hypotheses and validation plan
- Research decision log
- What would impress vs what would fail review
- References to relevant papers (MobileLLM, GQA, KD surveys)

### 6.2 Create PROJECT_CODER.md

**File**: `C:\Anubhav\Adaptive-SLM\PROJECT_CODER.md`

Contents outline:
- Exact repo structure with status annotations
- Inference pipeline walkthrough (current fake → target real)
- How to add a new model (change GGUF path + prompt template)
- How to replace Qwen with Gemma (trivial with llama.cpp)
- Build instructions for Linux/Windows/Android
- Subsystem status table with file:line references
- Technical debt register
- Engineering changelog
- Common failure points and debugging

### 6.3 Create PROJECT_TESTER.md

**File**: `C:\Anubhav\Adaptive-SLM\PROJECT_TESTER.md`

Contents outline:
- Test coverage: 0% (only bench.cpp exists)
- Missing tests inventory
- How to verify 40+ tok/s (run bench with real model, check output is not demo string)
- How to verify sub-512MB (run under `ulimit -v 524288` or cgroups)
- How to detect fake benchmarks (check if output contains "[AdaptiveSLM] Demo response")
- Benchmark procedures for each claim
- Memory leak detection strategy (ASan + Valgrind)
- Reproducible bugs list
- Reliability checklist

---

## Phase 7: Resume Claim Credibility Scoring

| Claim | Current Score (0-10) | After P0 | After P2 | After P4 |
|-------|---------------------|----------|----------|----------|
| "C++/Rust hybrid inference engine" | 3/10 (scaffolding only) | 7/10 | 8/10 | 9/10 |
| "40+ tokens/sec" | 0/10 (measures string copy) | 6/10 | 7/10 | 9/10 |
| "Sub-512MB devices" | 2/10 (plausible but unproven) | 5/10 | 7/10 | 9/10 |
| "GGML 4-bit quantized" | 2/10 (file exists, not loaded) | 8/10 | 8/10 | 9/10 |
| "PAKD (novel contribution)" | 4/10 (code exists, never run) | 4/10 | 4/10 | 7/10 |
| "ACC (novel contribution)" | 8/10 (fully working) | 9/10 | 9/10 | 10/10 |
| "SCPD (novel contribution)" | 4/10 (logic works, embeddings fake) | 4/10 | 8/10 | 9/10 |
| **Overall** | **3/10** | **6/10** | **7/10** | **9/10** |

---

## Execution Order Summary

```
Week 1 (P0):
  Day 1-2: Replace GGML with llama.cpp, rewrite inference.cpp
  Day 3:   Wire ACC into llama.cpp context, verify real generation
  Day 4:   Run first real benchmarks, record tok/s and memory
  Day 5:   Fix all critical bugs (fromJson, platform, CMake)

Week 2 (P1-P2):
  Day 1-2: Cross-platform device monitoring
  Day 3-4: Real embeddings for SCPD cache
  Day 5:   sqlite-vec integration, benchmark cache performance

Week 3 (P3-P4):
  Day 1-2: Enable training pipeline, test on placeholder data
  Day 3:   Model-agnostic config, add Gemma support
  Day 4:   Unit tests for all subsystems
  Day 5:   Real benchmark suite with measured numbers

Week 4 (P5):
  Token streaming, Android build, CI/CD, documentation update
```

---

## Next Best Action

**Immediately**: Rewrite `core/src/inference.cpp` to use llama.cpp for real model loading and inference. This single change transforms the project from a demo to a working engine and unblocks every other improvement. Everything else is polish until inference is real.