# PROJECT_CODER.md — Engineering Intelligence Agent
> Last updated: 2026-05-16 | Audit revision: 1.0

## 1. Repository Structure (with status annotations)

```
C:\Anubhav\Adaptive-SLM\
├── CLAUDE.md                          # Project intelligence for Claude Code
├── PROJECT_RESEARCHER.md              # Research context
├── PROJECT_CODER.md                   # THIS FILE
├── PROJECT_TESTER.md                  # Testing context
│
├── core/                              # C++ inference engine
│   ├── CMakeLists.txt                 # Build config — llama.cpp FetchContent
│   ├── include/
│   │   ├── adaptive_slm.h            # ✅ Public C API (165 lines)
│   │   ├── context_adapt.h           # ✅ ACC interface (78 lines)
│   │   ├── embeddings.h              # ✅ SIMD engine interface (94 lines)
│   │   └── user_profile.h            # ✅ User profile interface (74 lines)
│   ├── src/
│   │   ├── inference.cpp             # ⚠️ NEEDS REWRITE — uses llama.cpp for real inference
│   │   ├── context_adapt.cpp         # ✅ ACC fully working (186 lines)
│   │   ├── embeddings.cpp            # ⚠️ SIMD real, embeddings are placeholder (221 lines)
│   │   └── user_profile.cpp          # ⚠️ Works except fromJson() bug (201 lines)
│   ├── benchmark/
│   │   └── bench.cpp                 # ⚠️ Framework works, needs real inference (223 lines)
│   └── build/                        # CMake build directory
│
├── rust-wrapper/                      # Rust async orchestration layer
│   ├── Cargo.toml                     # ✅ Dependencies correct
│   └── src/
│       ├── lib.rs                     # ✅ Main wrapper — cache→search→inference (385 lines)
│       ├── ffi.rs                     # ✅ FFI bindings complete (92 lines)
│       ├── cache.rs                   # ⚠️ Logic real, embeddings fake (363 lines)
│       ├── profile.rs                 # ✅ User profile complete (155 lines)
│       └── search.rs                  # ✅ Tavily + DDG search (267 lines)
│
├── training/
│   ├── train.py                       # ⚠️ All phases commented out (783 lines)
│   └── prepare_data.py               # ✅ Data pipeline works (351 lines)
│
├── benchmarks/
│   ├── mmlu_eval.py                   # ⚠️ Needs real model to run (261 lines)
│   └── compare.py                    # ❌ Hardcoded results (131 lines)
│
├── storage/
│   └── schema.sql                    # ✅ Schema complete (74 lines)
│
├── docs/
│   ├── index.md                      # Documentation hub
│   ├── architecture.md               # System architecture
│   ├── api_reference.md              # API docs
│   └── training.md                   # Training guide
│
└── models/
    └── qwen2.5-0.5b-instruct-q4_k_m.gguf  # ✅ 469MB model file present
```

## 2. Runtime Entry Points

### C++ Core
- **Main API**: `core/include/adaptive_slm.h` — all public functions
- **Init**: `aslm_init()` in `core/src/inference.cpp` — loads model via llama.cpp
- **Generate**: `aslm_generate()` in `core/src/inference.cpp` — tokenize→decode→sample→detokenize
- **ACC update**: `aslm_update_device_state()` — adjusts context via ACC

### Rust Wrapper
- **Entry**: `AdaptiveSLM::new()` in `rust-wrapper/src/lib.rs` — initializes C++ context + SQLite cache
- **Generate**: `AdaptiveSLM::generate()` — orchestrates cache check → web search → C++ inference → cache store

### Training
- **Entry**: `python training/train.py` — runs pretrain → distill → PAKD → export

## 3. Inference Pipeline Walkthrough

### Current Flow (after llama.cpp integration):
```
1. User calls AdaptiveSLM::generate(prompt, params, use_search)  [lib.rs]
2. Compute query embedding → search SCPD cache                   [cache.rs]
3. If cache hit (similarity > 0.8): return cached response        [cache.rs]
4. If use_search: query Tavily/DDG for context                    [search.rs]
5. Build augmented prompt (profile modifier + search context)     [lib.rs]
6. FFI call: aslm_generate(ctx, prompt, params, output, size)    [ffi.rs → inference.cpp]
   6a. llama_tokenize(prompt) → token IDs                        [inference.cpp]
   6b. llama_decode(batch) → process prompt through model         [inference.cpp]
   6c. Loop: llama_sampler_sample() → sample next token           [inference.cpp]
   6d. Check EOS → llama_token_to_piece() → append to output     [inference.cpp]
7. Store result in SCPD cache with priority score                 [cache.rs]
8. Return generated text                                          [lib.rs]
```

## 4. Subsystem Details

### 4.1 Inference Engine (`core/src/inference.cpp`)

**Purpose**: Load GGUF model, tokenize, run forward pass, sample, detokenize.

**Key types**:
- `aslm::Context` — holds `llama_model*`, `llama_context*`, `llama_sampler*`, ACC, profile
- `aslm_init_params` — model_path, n_threads, max_context, use_mmap, verbose
- `aslm_gen_params` — max_tokens, temperature, top_p, top_k, repeat_penalty

**Dependencies**: llama.cpp (includes GGML, tokenizer, KV cache, sampler)

**Status**: Being rewritten to use llama.cpp for real inference.

**Known bugs**: None after rewrite (previous version was entirely fake).

### 4.2 ACC Module (`core/src/context_adapt.cpp`)

**Purpose**: Dynamically adjust context window based on device resource state.

**Key classes/functions**:
- `ContextAdapter(config)` — constructor, stores config + EMA state
- `computeResourceScore(state)` → float [0,1] — weighted RAM/CPU/battery
- `computeContextSize(state)` → int32 — maps score to context size
- `getSmoothedContextSize(state)` → int32 — applies EMA + power-of-2 snap
- `getCurrentDeviceState()` → aslm_device_state — reads /proc/ (Linux)

**Dependencies**: None (standalone algorithm)

**Status**: ✅ FULLY IMPLEMENTED AND TESTED

**Known bugs**:
- `getCurrentDeviceState()` is Linux-only — needs `#ifdef _WIN32` + `#ifdef __APPLE__`
- First CPU measurement returns incorrect value (prev_idle=0 on first call)

**Refactor recommendation**: Extract device state reading into a separate platform abstraction layer.

### 4.3 SIMD Embeddings (`core/src/embeddings.cpp`)

**Purpose**: SIMD-accelerated vector operations for cosine similarity and dot product.

**Key functions**:
- `EmbeddingEngine::dotProduct(a, b, dim)` — AVX2/NEON/scalar dot product
- `EmbeddingEngine::cosineSimilarity(a, b, dim)` — uses dotProduct + norms
- `EmbeddingEngine::normalize(vec, dim)` — L2 normalization
- `EmbeddingEngine::embed(text)` — ⚠️ PLACEHOLDER (hash-based, not semantic)
- `EmbeddingEngine::tokenize(text)` — ⚠️ PLACEHOLDER (word-level hash % 10000)

**Dependencies**: `<immintrin.h>` (AVX2) or `<arm_neon.h>` (NEON)

**Status**: SIMD math is REAL and correct. Embedding generation is FAKE.

**Known bugs**: `embed()` produces deterministic but semantically meaningless vectors.

**Refactor recommendation**: Replace `poolEmbeddings()` with real embedding extraction from llama.cpp model hidden states, or load a separate MiniLM GGUF model.

### 4.4 User Profile (`core/src/user_profile.cpp` + `rust-wrapper/src/profile.rs`)

**Purpose**: User personalization for PAKD and SCPD.

**Key functions (C++)**:
- `UserProfile(age, background, interests, expertise)` — constructor
- `computeRelevance(topic, embedding)` → float [0,1] — 70% interest + 30% background
- `getPromptModifier()` → string — age/expertise-appropriate system prompt
- `toJson()` / `fromJson()` — serialization

**Key functions (Rust)**:
- `UserProfile::compute_relevance(topic)` → f32 — Jaccard word similarity + containment
- `UserProfile::get_prompt_modifier()` → String — mirrors C++ logic
- `UserProfile::save()` / `UserProfile::load()` — JSON via serde

**Status**: Both implementations work. C++ `fromJson()` has a bug.

**Known bugs**: `fromJson()` at `user_profile.cpp:130-161` never parses the interests JSON array — always returns empty vector.

### 4.5 SCPD Cache (`rust-wrapper/src/cache.rs`)

**Purpose**: Semantic cache with priority decay for knowledge retention.

**Key types**:
- `KnowledgeCache` — main cache struct with SQLite connection
- `KnowledgeEntry` — id, topic, content, embedding, priority, access_count, timestamps
- `CacheConfig` — weights (α=0.4, β=0.35, γ=0.25), decay=0.95, threshold=0.1, max=1000

**Key functions**:
- `new(db_path, config)` — creates SQLite tables and indices
- `store(topic, content, priority, source_url)` — insert with embedding computation
- `search(query, threshold)` → Option<KnowledgeEntry> — cosine similarity search
- `update_priority(id, relevance, profile_match)` — applies SCPD formula
- `decay_and_evict()` — applies daily decay, removes below-threshold entries
- `compute_embedding(text)` — ⚠️ FAKE (hash-based at line 307-315)

**Status**: Logic fully working. Embeddings are fake. sqlite-vec fails to load.

### 4.6 Web Search (`rust-wrapper/src/search.rs`)

**Purpose**: RAG via web search for query augmentation.

**Key types**:
- `TavilyClient` — primary search via Tavily API
- `DuckDuckGoClient` — fallback (no API key needed)
- `SearchResult` — title, url, snippet, score

**Status**: ✅ FULLY IMPLEMENTED. Tavily primary + DDG fallback.

### 4.7 Training Pipeline (`training/train.py`)

**Purpose**: Train MobileLLM-style model with KD + PAKD.

**Key classes**:
- `AdaptiveSLMModel` — 30-layer transformer with GQA, SwiGLU, RMSNorm (~70M params)
- `KnowledgeDistillationLoss` — CE + KL divergence with temperature
- `PAKDLoss` — profile-weighted KD loss (beginner=1.2, intermediate=1.0, expert=0.8)
- `AdaptiveSLMTrainer` — orchestrates pretrain/distill/pakd/export

**Status**: Code complete but ALL PHASES COMMENTED OUT at lines 767-776.

## 5. How-To Guides

### How to Add a New Model

1. Download the GGUF file (e.g., from Hugging Face)
2. Place in `models/` directory
3. Pass the path to `aslm_init()` via `init_params.model_path`
4. llama.cpp handles architecture detection automatically from GGUF metadata
5. Update prompt template if needed (ChatML for Qwen, Gemma format for Gemma, etc.)

### How to Replace Qwen with Gemma

1. Download Gemma GGUF: `huggingface-cli download google/gemma-3-1b-it-GGUF`
2. Change model path in your application code
3. Update prompt template: `<|im_start|>` → `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
4. No C++ changes needed — llama.cpp handles Gemma architecture internally
5. Note: Gemma 1B Q4 is ~700MB — may exceed 512MB target

### How Quantization Currently Works

1. Pre-quantized GGUF file downloaded (not quantized locally)
2. Model file: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M = 4-bit with k-quant medium)
3. llama.cpp loads quantized weights directly from GGUF format
4. Dequantization happens on-the-fly during matrix multiplications
5. To requantize: use `llama-quantize` tool from llama.cpp build

### How to Benchmark Tokens/sec

```bash
cd core/build
./adaptive_slm_bench
# Look for "Inference Benchmark" section
# Verify output does NOT contain "[AdaptiveSLM] Demo response"
# Real output should be coherent text from the model
```

### How to Profile Memory

```bash
# Linux: Use /proc/self/status
cd core/build
./adaptive_slm_bench  # Reports memory via aslm_get_memory_usage()

# Detailed: Use Valgrind massif
valgrind --tool=massif ./adaptive_slm_bench
ms_print massif.out.<pid>

# ASan: Rebuild with sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
make -j$(nproc)
./adaptive_slm_bench
```

### How to Debug Inference Failures

1. Set `init_params.verbose = true` for llama.cpp logging
2. Check model file exists: `ls -la models/*.gguf`
3. Verify model loads: look for "model loaded" in verbose output
4. Check tokenization: verify prompt tokens count matches expected
5. Check memory: ensure enough RAM for model + KV cache
6. Common failure: model file corrupt or wrong format → re-download

## 6. Build Instructions

### Linux (GCC/Clang)
```bash
cd core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Output: libadaptive_slm.a + adaptive_slm_bench
```

### Windows (MSVC)
```powershell
cd core
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Rust Wrapper
```bash
cd rust-wrapper
cargo build --release
# Requires: libadaptive_slm.a built first (linked via FFI)
```

### Training
```bash
pip install -r requirements.txt
python training/prepare_data.py --placeholder-only  # Generate test data
python training/train.py  # Requires GPU
```

## 7. Technical Debt Register

| # | Severity | Location | Description | Fix Estimate |
|---|----------|----------|-------------|-------------|
| 1 | HIGH | `cache.rs:307-315` | Fake hash-based embeddings | 2-4 hours |
| 2 | HIGH | `embeddings.cpp:73-79` | Fake sin-based embeddings | 2-4 hours |
| 3 | MEDIUM | `user_profile.cpp:130-161` | fromJson() doesn't parse interests | 30 min |
| 4 | MEDIUM | `context_adapt.cpp:121-183` | Linux-only device monitoring | 1-2 hours |
| 5 | MEDIUM | `train.py:767-776` | Training phases commented out | 30 min |
| 6 | MEDIUM | `train.py:737-747` | export_gguf() is placeholder | 2 hours |
| 7 | LOW | `compare.py:61` | Hardcoded benchmark values | 30 min |
| 8 | LOW | `schema.sql:~60` | sqlite-vec table commented out | 15 min |
| 9 | LOW | `cache.rs` | O(n) brute-force search on cache miss | 1 hour |

## 8. Engineering Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2026-05-16 | Initial audit completed | Identified all fakes/placeholders |
| 2026-05-16 | Plan: Replace GGML with llama.cpp | Critical path for real inference |
| 2026-05-16 | Created CLAUDE.md + intelligence files | Future sessions have full context |

## 9. Performance Bottlenecks (Anticipated)

1. **Model loading**: First-time load of 469MB GGUF takes 1-3 seconds. With mmap, subsequent loads are near-instant.
2. **Prompt processing**: Long prompts (>512 tokens) increase first-token latency proportionally.
3. **Cache search**: O(n) without sqlite-vec. Degrades with >1000 entries.
4. **KV cache memory**: Grows linearly with context length. At 2048 tokens, ~24MB for this model.
5. **CPU thermal throttling**: On mobile devices, sustained inference causes thermal throttling → tok/s drops. ACC should detect this via CPU usage spike.
