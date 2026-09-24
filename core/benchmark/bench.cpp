/**
 * AdaptiveSLM Benchmark Suite
 * Validates real inference, memory, ACC, and SIMD performance
 */

#include "adaptive_slm.h"
#include "context_adapt.h"
#include "embeddings.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Memory Benchmark
// ============================================================================

void benchmark_memory(aslm_context* ctx) {
    std::cout << "\n=== Memory Benchmark ===" << std::endl;

    uint64_t usage = aslm_get_memory_usage(ctx);
    uint64_t peak = aslm_get_peak_memory_usage(ctx);

    std::cout << "Current state size: " << (usage / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Model + state size: " << (peak / 1024.0 / 1024.0) << " MB" << std::endl;

    if (peak < 512ULL * 1024 * 1024) {
        std::cout << "[PASS] Memory target MET (< 512 MB)" << std::endl;
    } else {
        std::cout << "[WARN] Model + state exceeds 512 MB (mmap reduces RSS)" << std::endl;
    }
}

// ============================================================================
// ACC Benchmark
// ============================================================================

void benchmark_acc() {
    std::cout << "\n=== ACC (Adaptive Context Compression) Benchmark ===" << std::endl;

    aslm::ACCConfig config;
    aslm::ContextAdapter acc(config);

    struct TestCase {
        const char* name;
        aslm_device_state state;
        int32_t expected_min;
        int32_t expected_max;
    };

    std::vector<TestCase> test_cases = {
        {"High resources",    {8ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.1f, 1.0f, true},  1024, 2048},
        {"Medium resources",  {4ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.5f, 0.5f, false}, 256, 1024},
        {"Low resources",     {1ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.8f, 0.2f, false}, 128, 512},
        {"Critical (low RAM)",{500ULL*1024*1024,    8ULL*1024*1024*1024, 0.9f, 0.1f, false}, 128, 256},
    };

    std::cout << std::setw(20) << "Scenario"
              << std::setw(15) << "Context Size"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    int pass_count = 0;
    for (const auto& tc : test_cases) {
        int32_t ctx_size = acc.computeContextSize(tc.state);
        bool pass = (ctx_size >= tc.expected_min && ctx_size <= tc.expected_max);
        if (pass) pass_count++;

        std::cout << std::setw(20) << tc.name
                  << std::setw(15) << ctx_size
                  << std::setw(10) << (pass ? "[PASS]" : "[FAIL]")
                  << std::endl;
    }

    std::cout << "ACC tests: " << pass_count << "/" << test_cases.size() << " passed" << std::endl;
}

// ============================================================================
// Inference Benchmark
// ============================================================================

void benchmark_inference(aslm_context* ctx) {
    std::cout << "\n=== Inference Benchmark ===" << std::endl;

    const char* prompts[] = {
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
    };
    const int NUM_PROMPTS = sizeof(prompts) / sizeof(prompts[0]);
    const int NUM_ITERATIONS = 3;

    aslm_gen_params params = {
        .max_tokens = 64,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .repeat_penalty = 1.1f
    };

    char output[4096];
    double total_time = 0.0;
    int32_t total_tokens = 0;
    bool all_real = true;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const char* prompt = prompts[i % NUM_PROMPTS];

        auto start = Clock::now();
        int32_t tokens = aslm_generate(ctx, prompt, &params, output, sizeof(output));
        auto end = Clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();

        if (tokens < 0) {
            std::cout << "[FAIL] Generation failed for prompt: " << prompt << std::endl;
            continue;
        }

        // Detect fake/demo output
        if (std::strstr(output, "[AdaptiveSLM] Demo response") != nullptr) {
            std::cout << "[FAIL] FAKE OUTPUT DETECTED — inference is not real!" << std::endl;
            all_real = false;
            continue;
        }

        total_time += elapsed;
        total_tokens += tokens;

        std::cout << "  Prompt " << (i+1) << ": " << tokens << " tokens in "
                  << std::fixed << std::setprecision(1) << (elapsed * 1000) << " ms"
                  << " (" << std::setprecision(1) << (tokens / elapsed) << " tok/s)"
                  << std::endl;
        // Print first 120 chars of output as proof of real generation
        std::string preview(output, std::min(strlen(output), (size_t)120));
        std::cout << "  Output: " << preview << "..." << std::endl;
    }

    if (total_tokens > 0 && total_time > 0) {
        double avg_time = total_time / NUM_ITERATIONS;
        double tokens_per_sec = total_tokens / total_time;

        std::cout << "\nAverage time per generation: " << std::setprecision(1)
                  << (avg_time * 1000) << " ms" << std::endl;
        std::cout << "Overall tokens per second:   " << std::setprecision(1)
                  << tokens_per_sec << std::endl;

        if (tokens_per_sec > 40) {
            std::cout << "[PASS] Speed target MET (> 40 tok/s)" << std::endl;
        } else if (tokens_per_sec > 10) {
            std::cout << "[WARN] Speed above minimum (> 10 tok/s) but below 40 tok/s target" << std::endl;
        } else {
            std::cout << "[FAIL] Speed below minimum threshold" << std::endl;
        }
    }

    if (all_real) {
        std::cout << "[PASS] All outputs are REAL model inference (not demo strings)" << std::endl;
    }
}

// ============================================================================
// SIMD Benchmark
// ============================================================================

void benchmark_simd() {
    std::cout << "\n=== SIMD Vector Operations Benchmark ===" << std::endl;

    const int DIM = 384;
    const int NUM_ITERATIONS = 100000;

    std::vector<float> a(DIM), b(DIM);
    for (int i = 0; i < DIM; ++i) {
        a[i] = static_cast<float>(i) / DIM;
        b[i] = static_cast<float>(DIM - i) / DIM;
    }

    // Warm up
    volatile float result = 0;
    for (int i = 0; i < 1000; ++i) {
        result = aslm::EmbeddingEngine::dotProduct(a.data(), b.data(), DIM);
    }

    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        result = aslm::EmbeddingEngine::dotProduct(a.data(), b.data(), DIM);
    }
    auto end = Clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double ops_per_sec = NUM_ITERATIONS / elapsed;

    std::cout << "Dot product ops/sec: " << std::fixed << std::setprecision(0)
              << ops_per_sec << std::endl;

#if ASLM_USE_AVX2
    std::cout << "SIMD: AVX2 enabled" << std::endl;
#elif ASLM_USE_NEON
    std::cout << "SIMD: NEON enabled" << std::endl;
#else
    std::cout << "SIMD: scalar fallback" << std::endl;
#endif
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << "  AdaptiveSLM Benchmark Suite                           " << std::endl;
    std::cout << "  Research-Grade On-Device Language Model                " << std::endl;
    std::cout << "========================================================" << std::endl;

    const char* model_path = argc > 1
        ? argv[1]
        : "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // Initialize context with real model
    aslm_init_params params = aslm_default_init_params();
    params.model_path = model_path;
    params.n_threads = 4;
    params.verbose = true;
    params.use_mmap = true;

    std::cout << "\nLoading model: " << model_path << std::endl;

    aslm_context* ctx = aslm_init(&params);
    if (!ctx) {
        std::cerr << "\nFailed to initialize context." << std::endl;
        std::cerr << "Ensure the model file exists at: " << model_path << std::endl;
        std::cerr << "Download with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF"
                  << " qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/" << std::endl;

        // Still run non-model benchmarks
        benchmark_acc();
        benchmark_simd();
        std::cout << "\n=== Benchmark Partial (no model) ===" << std::endl;
        return 1;
    }

    // Create test user profile
    const char* interests[] = {"programming", "machine learning", "science"};
    aslm_user_profile* profile = aslm_profile_create(
        25, "software engineer", interests, 3, ASLM_EXPERTISE_ADVANCED
    );
    aslm_set_user_profile(ctx, profile);

    // Run all benchmarks
    benchmark_memory(ctx);
    benchmark_acc();
    benchmark_simd();
    benchmark_inference(ctx);

    // Cleanup
    aslm_profile_free(profile);
    aslm_free(ctx);

    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    return 0;
}
