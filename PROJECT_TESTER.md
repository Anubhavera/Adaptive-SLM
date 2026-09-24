# PROJECT_TESTER.md — QA & Performance Intelligence Agent
> Last updated: 2026-05-16 | Audit revision: 1.0

## 1. Test Coverage Map

| Subsystem | Unit Tests | Integration Tests | Benchmarks | Coverage |
|-----------|-----------|-------------------|------------|----------|
| Inference engine | NONE | NONE | bench.cpp (framework only) | 0% |
| ACC module | NONE (bench.cpp has 4 scenarios) | NONE | bench.cpp ACC section | ~40% |
| SIMD embeddings | NONE | NONE | bench.cpp SIMD section | ~20% |
| User Profile (C++) | NONE | NONE | NONE | 0% |
| User Profile (Rust) | NONE | NONE | NONE | 0% |
| SCPD Cache | NONE | NONE | NONE | 0% |
| Web Search | NONE | NONE | NONE | 0% |
| FFI Bridge | NONE | NONE | NONE | 0% |
| Training Pipeline | NONE | NONE | NONE | 0% |
| Data Preparation | NONE | NONE | NONE | 0% |

**Overall test coverage: ~5%** (only bench.cpp provides any validation)

## 2. Missing Tests (Priority Ordered)

### Critical (Must Have)
1. **Inference smoke test**: Generate text, verify it's NOT "[AdaptiveSLM] Demo response"
2. **Inference determinism**: temperature=0 → same prompt → same output
3. **Memory budget test**: RSS during inference < 512MB
4. **Model loading test**: Load GGUF, verify no errors
5. **Tokenizer roundtrip**: tokenize(detokenize(tokens)) == tokens

### High Priority
6. **ACC scenario tests**: 4 resource states → verify context size ranges (exists in bench.cpp, needs gtest)
7. **ACC EMA smoothing**: Rapid state changes → verify smooth transitions
8. **SCPD store/search**: Store entry → search with same query → cache hit
9. **SCPD eviction**: Fill cache to max → verify lowest-priority evicted
10. **SCPD decay**: Time-travel test → verify priority decreases

### Medium Priority
11. **UserProfile prompt modifier**: Check output contains expertise-appropriate language
12. **UserProfile relevance**: Known-similar topic → high score, unrelated → low score
13. **UserProfile JSON roundtrip**: toJson(profile) → fromJson → verify fields match
14. **SIMD correctness**: dotProduct_simd(a,b) == dotProduct_scalar(a,b) within epsilon
15. **FFI memory safety**: Init → generate → free → no leaks (ASan)

### Low Priority
16. **Web search**: Mock HTTP → verify Tavily request format
17. **Web search fallback**: Tavily fails → DDG used
18. **Multi-threaded inference**: Two concurrent generate calls → no crashes
19. **Long context**: 2048-token prompt → no OOM, correct output
20. **Empty prompt**: Empty string → graceful handling, no crash

## 3. Benchmark Procedures

### How to Verify "40+ tokens/sec"

```bash
# Step 1: Build with release optimizations
cd core/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Step 2: Run benchmark
./adaptive_slm_bench

# Step 3: Verify output
# MUST see coherent text output (NOT "[AdaptiveSLM] Demo response")
# MUST see "tokens/sec: XX.X" where XX.X > 40

# Step 4: Cross-validate with llama.cpp's own benchmark
# Build llama.cpp directly and run:
./llama-bench -m ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf -t 4
# Compare tok/s numbers
```

**What hardware to test on:**
- x86 laptop (AVX2): Expect 30-60 tok/s
- Raspberry Pi 4 (NEON): Expect 10-20 tok/s
- Android phone (Snapdragon): Expect 20-40 tok/s

**How to detect fake benchmarks:**
1. Check if output text contains "Demo response" or "[AdaptiveSLM]"
2. Check if tok/s is suspiciously round (e.g., exactly 50.0)
3. Check if benchmark runs suspiciously fast (< 1 second for 100 tokens)
4. Verify that different prompts produce different outputs
5. Verify that temperature > 0 produces slightly different outputs each run

### How to Verify "sub-512MB" Claim

```bash
# Method 1: /proc/self/status (Linux)
# Add to bench.cpp:
# Read /proc/self/status → VmRSS line → physical memory

# Method 2: cgroups memory limit
cgcreate -g memory:slm_test
cgset -r memory.limit_in_bytes=536870912 slm_test  # 512MB
cgexec -g memory:slm_test ./adaptive_slm_bench
# If it runs: PASS. If OOM-killed: FAIL.

# Method 3: ulimit (simpler but less precise)
ulimit -v 524288  # 512MB virtual memory limit
./adaptive_slm_bench

# Method 4: Valgrind massif (most detailed)
valgrind --tool=massif --pages-as-heap=yes ./adaptive_slm_bench
ms_print massif.out.<pid> | head -30
# Check peak memory in snapshot
```

### How to Verify Gemma Migration

1. Download Gemma GGUF file
2. Change model path in benchmark
3. Run same benchmark suite
4. Compare: tok/s, memory usage, output quality
5. Verify no crashes or format errors
6. If llama.cpp loads it without code changes: migration validated

### How to Run Edge-Device Tests

```bash
# Cross-compile for ARM
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-toolchain.cmake
make -j$(nproc)

# Copy to target device
scp adaptive_slm_bench pi@raspberrypi:~/

# Run on device
ssh pi@raspberrypi
./adaptive_slm_bench
# Record: tok/s, memory, thermal throttling effects
```

## 4. Reproducible Bugs

| # | Bug | Reproduction | Severity | Status |
|---|-----|--------------|----------|--------|
| 1 | `fromJson()` always returns empty interests | `auto p = UserProfile::fromJson(profile.toJson()); p.getInterests().empty() == true` | HIGH | OPEN |
| 2 | `getCurrentDeviceState()` returns zeros on Windows | Call function on Windows → all fields 0 | HIGH | OPEN |
| 3 | First CPU measurement always wrong | Call `getCurrentDeviceState()` once → `cpu_usage` ≈ 1.0 (static prev_idle=0) | MEDIUM | OPEN |
| 4 | sqlite-vec extension fails to load | `cache.rs` logs warning, falls back to O(n) | MEDIUM | OPEN |
| 5 | CMake fails with MSVC | `-march=native` flag not recognized by cl.exe | MEDIUM | OPEN |

## 5. Memory Leak Detection Strategy

### Build with AddressSanitizer
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
make -j$(nproc)
./adaptive_slm_bench 2>&1 | tee asan_output.txt
# Check for "ERROR: AddressSanitizer" lines
```

### Valgrind (Linux)
```bash
valgrind --leak-check=full --show-leak-kinds=all ./adaptive_slm_bench
# Check for "definitely lost" and "possibly lost" bytes
```

### Key areas to watch for leaks:
1. `aslm_init()` → `aslm_free()` cycle: model, context, sampler must all be freed
2. `aslm_generate()`: output buffer is caller-owned, but internal temporaries must be freed
3. `EmbeddingEngine` constructor: if vocab_embeddings_ allocation fails, check cleanup
4. Rust Drop impl for `AdaptiveSLM`: must call `aslm_free()` on the C++ context

## 6. Stress Testing Procedures

### Sustained Inference
```bash
# Run 1000 consecutive generations
for i in $(seq 1 1000); do
    echo "Iteration $i"
    ./adaptive_slm_bench 2>&1 | grep "tokens/sec"
done | tee sustained_test.log
# Check: tok/s should NOT decrease over time (memory leak indicator)
# Check: no crashes or OOM
```

### Rapid Context Size Changes (ACC Stress)
```
Simulate rapid device state changes:
- High resources → Critical → High → Low → Critical → High
- Each state held for 1 generation
- Verify: no crashes, context size changes are smooth (EMA)
```

### Cache Saturation
```
1. Set max_entries = 100
2. Store 200 entries
3. Verify: only 100 remain (eviction worked)
4. Verify: lowest-priority entries were evicted
5. Search for an evicted entry → cache miss
6. Search for a kept entry → cache hit
```

## 7. Performance Validation Checklist

- [ ] Real inference produces coherent text (not demo string)
- [ ] tok/s > 40 on target x86 hardware
- [ ] tok/s > 10 on ARM hardware
- [ ] Peak RSS < 512MB with mmap
- [ ] Peak RSS < 512MB without mmap using Q3_K_S
- [ ] First-token latency < 500ms for short prompts
- [ ] KV cache grows linearly, not exponentially
- [ ] SCPD cache hit returns in < 1ms
- [ ] ACC context change completes in < 1ms
- [ ] Model load time < 3 seconds (mmap) / < 10 seconds (no mmap)
- [ ] No memory leaks over 1000 generations (ASan clean)
- [ ] Thermal throttling detected by ACC (CPU usage spike → context reduction)

## 8. Reliability Checklist

- [ ] `aslm_init(NULL)` → returns NULL gracefully (not crash)
- [ ] `aslm_generate(ctx, NULL, ...)` → returns error (not crash)
- [ ] `aslm_generate(ctx, "", ...)` → returns empty or error (not crash)
- [ ] `aslm_free(NULL)` → no-op (not crash)
- [ ] Double `aslm_free()` → no double-free crash
- [ ] Model file missing → clear error message at init
- [ ] Model file corrupt → clear error message at init
- [ ] Out of memory during generation → graceful failure (not segfault)
- [ ] Thread safety: concurrent `aslm_generate()` → no data races
- [ ] Very long prompt (>2048 tokens) → truncation or error (not crash)

## 9. Benchmark History

| Date | Hardware | tok/s | Memory (RSS) | Model | Notes |
|------|----------|-------|-------------|-------|-------|
| 2026-05-16 | N/A | N/A | N/A | N/A | Inference was fake — no real benchmarks exist yet |

*This table will be populated as real benchmarks are recorded.*

## 10. Regression History

| Date | Regression | Cause | Fix |
|------|-----------|-------|-----|
| *None yet* | | | |

## 11. Stability Score

**Current: 2/10** — Project builds but inference is a mock. No tests, no real benchmarks.

**Target: 8/10** — Real inference, unit tests for all subsystems, measured benchmarks, ASan-clean, cross-platform.

## 12. Quantization Validation Tests

When real inference is working, validate quantization quality:
1. Run same prompt through Q4_K_M and FP16 → compare outputs
2. Measure perplexity difference: FP16 vs Q4_K_M vs Q3_K_S
3. Run MMLU eval at each quantization level → track accuracy loss
4. Verify Q4_K_M file size matches expected (~469MB for 0.5B model)
5. Verify no NaN/Inf in model outputs (quantization corruption indicator)

## 13. Tokenizer Consistency Tests

1. `tokenize("Hello, world!")` → verify token count is reasonable (not hash % 10000)
2. `detokenize(tokenize(text)) == text` for simple ASCII
3. `detokenize(tokenize(text)) ≈ text` for Unicode (may have normalization differences)
4. Special tokens: BOS, EOS, PAD are handled correctly
5. Empty string tokenization → BOS token only (or empty, depending on add_special)
6. Very long string (>10K chars) → no buffer overflow
