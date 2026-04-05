/**
 * AdaptiveSLM Benchmark Suite
 * Memory profiling and performance benchmarks
 */

#include "adaptive_slm.h"
#include "context_adapt.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Memory Benchmark
// ============================================================================

void benchmark_memory(aslm_context* ctx) {
    std::cout << "\n=== Memory Benchmark ===" << std::endl;
    
    uint64_t usage = aslm_get_memory_usage(ctx);
    uint64_t peak = aslm_get_peak_memory_usage(ctx);
    
    std::cout << "Current memory usage: " << (usage / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Peak memory usage:    " << (peak / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Target: < 512 MB
    if (peak < 512 * 1024 * 1024) {
        std::cout << "✓ Memory target MET (< 512 MB)" << std::endl;
    } else {
        std::cout << "✗ Memory target EXCEEDED" << std::endl;
    }
}

// ============================================================================
// ACC Benchmark
// ============================================================================

void benchmark_acc() {
    std::cout << "\n=== ACC (Adaptive Context Compression) Benchmark ===" << std::endl;
    
    aslm::ACCConfig config;
    aslm::ContextAdapter acc(config);
    
    // Simulate different device states
    struct TestCase {
        const char* name;
        aslm_device_state state;
        int32_t expected_min;
        int32_t expected_max;
    };
    
    std::vector<TestCase> test_cases = {
        {"High resources", {8ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.1f, 1.0f, true}, 1024, 2048},
        {"Medium resources", {4ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.5f, 0.5f, false}, 256, 1024},
        {"Low resources", {1ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.8f, 0.2f, false}, 128, 512},
        {"Critical (low RAM)", {500ULL*1024*1024, 8ULL*1024*1024*1024, 0.9f, 0.1f, false}, 128, 256},
    };
    
    std::cout << std::setw(20) << "Scenario" 
              << std::setw(15) << "Context Size"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (const auto& tc : test_cases) {
        int32_t ctx_size = acc.computeContextSize(tc.state);
        
        bool pass = (ctx_size >= tc.expected_min && ctx_size <= tc.expected_max);
        
        std::cout << std::setw(20) << tc.name
                  << std::setw(15) << ctx_size
                  << std::setw(10) << (pass ? "✓ PASS" : "✗ FAIL")
                  << std::endl;
    }
}

// ============================================================================
// Inference Benchmark
// ============================================================================

void benchmark_inference(aslm_context* ctx) {
    std::cout << "\n=== Inference Benchmark ===" << std::endl;
    
    const int NUM_ITERATIONS = 10;
    const char* prompts[] = {
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
    };
    const int NUM_PROMPTS = sizeof(prompts) / sizeof(prompts[0]);
    
    aslm_gen_params params = {
        .max_tokens = 256,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .repeat_penalty = 1.1f
    };
    
    char output[4096];
    
    double total_time = 0.0;
    int32_t total_tokens = 0;
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const char* prompt = prompts[i % NUM_PROMPTS];
        
        auto start = Clock::now();
        int32_t tokens = aslm_generate(ctx, prompt, &params, output, sizeof(output));
        auto end = Clock::now();
        
        double elapsed = std::chrono::duration<double>(end - start).count();
        total_time += elapsed;
        total_tokens += tokens;
    }
    
    double avg_time = total_time / NUM_ITERATIONS;
    double tokens_per_sec = total_tokens / total_time;
    
    std::cout << "Average time per generation: " << (avg_time * 1000) << " ms" << std::endl;
    std::cout << "Tokens per second:           " << tokens_per_sec << std::endl;
    
    // Target: > 10 tokens/sec on RPi (this benchmark is on laptop, should be faster)
    if (tokens_per_sec > 10) {
        std::cout << "✓ Speed target MET (> 10 tok/s)" << std::endl;
    } else {
        std::cout << "✗ Speed target needs improvement" << std::endl;
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
    
    // Benchmark
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        result = aslm::EmbeddingEngine::dotProduct(a.data(), b.data(), DIM);
    }
    auto end = Clock::now();
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    double ops_per_sec = NUM_ITERATIONS / elapsed;
    
    std::cout << "Dot product operations/sec: " << std::fixed << std::setprecision(0) 
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
    std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     AdaptiveSLM Benchmark Suite                       ║" << std::endl;
    std::cout << "║     Research-Grade On-Device Language Model           ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    
    // Initialize context
    aslm_init_params params = aslm_default_init_params();
    params.model_path = argc > 1 ? argv[1] : "models/qwen2.5-0.5b-q4.gguf";
    params.n_threads = 4;
    params.verbose = true;
    
    aslm_context* ctx = aslm_init(&params);
    if (!ctx) {
        std::cerr << "Failed to initialize context" << std::endl;
        return 1;
    }
    
    // Create test user profile
    const char* interests[] = {"programming", "machine learning", "science"};
    aslm_user_profile* profile = aslm_profile_create(
        25,
        "software engineer",
        interests,
        3,
        ASLM_EXPERTISE_ADVANCED
    );
    aslm_set_user_profile(ctx, profile);
    
    // Run benchmarks
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
