# PROJECT_RESEARCHER.md — Research Intelligence Agent
> Last updated: 2026-05-16 | Audit revision: 1.0

## 1. Architecture Overview

AdaptiveSLM is a three-layer system:

```
┌─────────────────────────────────────────────────┐
│  Rust Async Wrapper (lib.rs)                    │
│  ├── SCPD Cache (cache.rs + SQLite)             │
│  ├── Web Search RAG (search.rs + Tavily/DDG)    │
│  ├── User Profile (profile.rs)                  │
│  └── FFI Bridge (ffi.rs)                        │
├─────────────────────────────────────────────────┤
│  C++ Core Engine                                │
│  ├── Inference (inference.cpp + llama.cpp)       │
│  ├── ACC Module (context_adapt.cpp)             │
│  ├── SIMD Embeddings (embeddings.cpp)           │
│  └── User Profile (user_profile.cpp)            │
├─────────────────────────────────────────────────┤
│  Model Layer                                    │
│  ├── Qwen2.5-0.5B Q4_K_M GGUF (469MB)          │
│  └── llama.cpp runtime (GGML tensor backend)    │
└─────────────────────────────────────────────────┘
```

**Design philosophy**: Thin C++ core for performance-critical inference + Rust wrapper for safe async orchestration + Python for offline training.

## 2. Three Novel Research Contributions

### 2.1 Profile-Aware Knowledge Distillation (PAKD)

**Theory**: Standard KD uses uniform loss across all samples. PAKD weights the distillation loss by user-profile relevance, so the student model retains more knowledge in domains matching the target user's expertise and interests.

**Loss function**: `L = (1-α)·CE(student, labels) + α·KL(student/T || teacher/T)·T² + β·L_profile`

Where L_profile applies per-sample weights: beginner=1.2, intermediate=1.0, expert=0.8 (beginners get boosted because they need simpler explanations).

**Implementation**: `training/train.py:239-280` — `PAKDLoss` class with domain weights (general=1.0, coding=1.2, science=1.1, creative=0.9).

**Status**: Code complete, mathematically sound, NEVER EXECUTED (training phases commented out at line 767-776).

**Confidence**: HIGH that the idea is novel and implementable. ZERO confidence in empirical results (none exist).

### 2.2 Adaptive Context Compression (ACC)

**Theory**: On resource-constrained devices, the available compute budget changes dynamically (battery drain, background apps, thermal throttling). ACC adjusts the effective context window in real-time based on device telemetry.

**Algorithm**:
1. Read device state: RAM available, CPU usage, battery level
2. Compute resource score: `S = 0.4·RAM_ratio + 0.3·CPU_avail + 0.3·battery`
3. Map to context size: linear interpolation between min(128) and max(2048)
4. Apply EMA smoothing: `ctx = 0.3·new + 0.7·old`
5. Snap to power-of-2: {128, 256, 512, 1024, 2048}
6. Emergency: RAM < 20% → force minimum (128 tokens)

**Implementation**: `core/src/context_adapt.cpp` — fully working, benchmarked with 4 scenarios.

**Status**: FULLY IMPLEMENTED AND TESTED. Best component in the project.

**Confidence**: HIGH. Algorithm is correct, tested, and would survive technical review.

### 2.3 Semantic Cache with Priority Decay (SCPD)

**Theory**: Cache previous inference results with semantic similarity matching. Priority decays exponentially over time, boosted by access frequency and user-profile relevance.

**Priority formula**: `P(t) = (α·Recency + β·Relevance + γ·ProfileMatch) · (1 + 0.1·ln(access_count + 1))`

Where: α=0.4, β=0.35, γ=0.25, Recency = e^(-0.1·days_ago), decay_factor = 0.95/day

**Implementation**: `rust-wrapper/src/cache.rs` — SQLite backend with store/search/evict/decay operations.

**Status**: Cache logic FULLY WORKING. Semantic similarity is FAKE (uses hash-based embeddings, not learned vectors). sqlite-vec extension fails to load, falls back to O(n) brute-force search.

**Confidence**: MEDIUM. Data structures and eviction logic are solid. Needs real embeddings to function as "semantic" cache.

## 3. Technical Feasibility Analysis

### 3.1 "40+ tokens/sec" — ACHIEVABLE

llama.cpp routinely achieves 30-60 tok/s for Q4 0.5B models on modern x86 CPUs (AVX2). On ARM (Raspberry Pi 4), expect 10-20 tok/s. On mobile ARM (Snapdragon 8 Gen 2), expect 20-40 tok/s.

**Requirements**: Proper llama.cpp integration with mmap, multi-threading, SIMD auto-detection (all provided by llama.cpp).

### 3.2 "Sub-512MB" — ACHIEVABLE WITH MMAP

| Scenario | Resident Memory | Verdict |
|----------|----------------|---------|
| Q4_K_M + mmap | ~220-330 MB | PASSES |
| Q4_K_M no mmap | ~540-550 MB | FAILS |
| Q3_K_S + mmap | ~180-280 MB | PASSES easily |
| Q3_K_S no mmap | ~420-480 MB | PASSES barely |

**Key insight**: mmap is essential. Without it, the model file must be fully loaded into RAM. With mmap, only active pages are resident.

### 3.3 Model Quality vs Size Tradeoff

| Model | Params | Q4 Size | MMLU | Fits 512MB? |
|-------|--------|---------|------|-------------|
| Qwen2.5-0.5B | 494M | 469MB | 43.7% | With mmap |
| Gemma 3 1B | 1B | ~700MB | ~50% | Barely |
| Gemma 3n E2B | 2B | ~1.2GB | ~55% | No |
| SmolLM2-360M | 360M | ~250MB | 27.3% | Yes |
| MobileLLM-125M | 125M | ~100MB | 25.6% | Yes easily |

**Recommendation**: Qwen2.5-0.5B is the sweet spot for quality/size. SmolLM2 if strict memory budget.

## 4. Comparison Against Existing Systems

| System | Approach | Our Advantage |
|--------|----------|--------------|
| llama.cpp (raw) | General-purpose inference | We add ACC, SCPD, profile-aware generation |
| MLC-LLM | Compiler-based optimization | We focus on runtime adaptation, not compilation |
| MediaPipe LLM | Google's mobile pipeline | We're model-agnostic, not Google-ecosystem locked |
| ExecuTorch | Meta's on-device framework | We're lighter weight, Rust safety layer |
| ONNX Runtime Mobile | General ML runtime | We're LLM-specific with semantic caching |

**What makes this project technically interesting**:
1. Runtime device adaptation (ACC) — most systems have static resource allocation
2. Semantic caching for on-device inference — novel for edge deployment
3. Profile-aware distillation — personalizing model behavior at training time
4. C++/Rust hybrid — combines performance with safety

## 5. What Claims Are Currently Defensible?

| Claim | Defensible? | Evidence Required |
|-------|-------------|-------------------|
| "ACC dynamically adjusts context based on device state" | YES | Code works, benchmarked |
| "SCPD caches with priority decay and eviction" | PARTIALLY | Logic works, needs real embeddings |
| "PAKD weights distillation by user profile" | PARTIALLY | Code exists, needs training run |
| "40+ tokens/sec on-device" | NOT YET | Needs llama.cpp integration + benchmarks |
| "Sub-512MB memory footprint" | NOT YET | Needs real memory profiling |
| "Novel contributions to edge ML" | YES for ACC | ACC is genuinely novel and implemented |

## 6. What Would Impress Senior AI Systems Engineers?

1. Real llama.cpp integration with measured tok/s on specific hardware
2. ACC adapting context size during actual inference with visible performance impact
3. SCPD cache demonstrating real semantic hit rates with measured latency savings
4. Memory profiling showing actual RSS under 512MB during inference
5. PAKD training showing measurable quality improvement over vanilla KD
6. Model-agnostic architecture supporting both Qwen and Gemma

## 7. What Would Fail a Technical Review?

1. Demo string in `aslm_generate()` — instant disqualification
2. Fake embeddings using sin() — shows the cache is theatrical
3. Hardcoded benchmark numbers in compare.py
4. Commented-out training pipeline — PAKD claim is vapor
5. Linux-only device monitoring on a "cross-platform" project
6. No unit tests whatsoever

## 8. Research Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-16 | Keep Qwen2.5-0.5B over Gemma | Only model fitting sub-512MB constraint |
| 2026-05-16 | Use llama.cpp instead of raw GGML | GGML cannot load GGUF; llama.cpp provides complete inference |
| 2026-05-16 | Prioritize real inference over training | No claim is defensible until inference works |

## 9. Open Research Questions

1. How much does PAKD actually improve over vanilla KD? (Hypothesis: 2-5% on domain-specific tasks)
2. What is the optimal ACC EMA factor? (Currently 0.3, may need tuning)
3. Does SCPD cache actually save latency on-device? (Hypothesis: 10-50x for cache hits)
4. What's the quality/size Pareto frontier for on-device models? (Need to benchmark multiple quantizations)
5. Can ACC predict resource drops before they happen? (Currently reactive, could be predictive)

## 10. References

- MobileLLM paper (Meta, 2024): Deep-thin architecture for mobile LLMs
- GGML/llama.cpp: https://github.com/ggerganov/llama.cpp
- Knowledge Distillation survey (Gou et al., 2021)
- Grouped-Query Attention (Ainslie et al., 2023)
- SwiGLU activation (Shazeer, 2020)
- sqlite-vec: https://github.com/asg017/sqlite-vec
