/**
 * Unit tests for Inference Engine (llama.cpp integration)
 */

#include "adaptive_slm.h"
#include <cstdio>
#include <cstring>

#define PASS(name) std::printf("[PASS] %s\n", name)
#define FAIL(name, msg) (std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg), ++failures)

static int failures = 0;

// ---------------------------------------------------------------------------
// Init/Params Tests
// ---------------------------------------------------------------------------

static void test_default_init_params() {
    aslm_init_params params = aslm_default_init_params();
    if (params.model_path == nullptr &&
        params.n_threads == 0 &&
        params.max_context == 2048 &&
        params.use_mmap == true &&
        params.verbose == false)
        PASS("default_init_params");
    else
        FAIL("default_init_params", "default params not as expected");
}

static void test_init_null_model_path() {
    aslm_init_params params = aslm_default_init_params();
    params.model_path = nullptr;
    aslm_context* ctx = aslm_init(&params);
    if (ctx == nullptr)
        PASS("init_null_model_path");
    else {
        FAIL("init_null_model_path", "should return null for missing model");
        aslm_free(ctx);
    }
}

static void test_init_invalid_model_path() {
    aslm_init_params params = aslm_default_init_params();
    params.model_path = "/nonexistent/model.gguf";
    aslm_context* ctx = aslm_init(&params);
    if (ctx == nullptr)
        PASS("init_invalid_model_path");
    else {
        FAIL("init_invalid_model_path", "should return null for nonexistent model");
        aslm_free(ctx);
    }
}

// ---------------------------------------------------------------------------
// Context Size Tests
// ---------------------------------------------------------------------------

static void test_get_effective_context_size_null() {
    int32_t ctx = aslm_get_effective_context_size(nullptr);
    if (ctx == 512)  // default fallback
        PASS("get_effective_context_size_null");
    else
        FAIL("get_effective_context_size_null", "expected default 512 for null");
}

// ---------------------------------------------------------------------------
// Profile Tests
// ---------------------------------------------------------------------------

static void test_profile_create_free() {
    const char* interests[] = {"programming", "ai"};
    aslm_user_profile* profile = aslm_profile_create(
        25, "engineer", interests, 2, ASLM_EXPERTISE_ADVANCED
    );
    if (profile != nullptr) {
        aslm_profile_free(profile);
        PASS("profile_create_free");
    } else {
        FAIL("profile_create_free", "profile creation returned null");
    }
}

static void test_profile_null_free() {
    aslm_profile_free(nullptr); // Should not crash
    PASS("profile_null_free");
}

static void test_set_user_profile_null_context() {
    const char* interests[] = {"test"};
    aslm_user_profile* profile = aslm_profile_create(25, "test", interests, 1, ASLM_EXPERTISE_BEGINNER);
    aslm_set_user_profile(nullptr, profile);
    aslm_profile_free(profile);
    PASS("set_user_profile_null_context");
}

// ---------------------------------------------------------------------------
// Memory Tests
// ---------------------------------------------------------------------------

static void test_get_memory_usage_null() {
    uint64_t mem = aslm_get_memory_usage(nullptr);
    if (mem == 0)
        PASS("get_memory_usage_null");
    else
        FAIL("get_memory_usage_null", "expected 0 for null context");
}

static void test_get_peak_memory_usage_null() {
    uint64_t mem = aslm_get_peak_memory_usage(nullptr);
    if (mem == 0)
        PASS("get_peak_memory_usage_null");
    else
        FAIL("get_peak_memory_usage_null", "expected 0 for null context");
}

// ---------------------------------------------------------------------------
// Device State Tests
// ---------------------------------------------------------------------------

static void test_update_device_state_null() {
    aslm_device_state state = {8ULL*1024*1024*1024, 8ULL*1024*1024*1024, 0.1f, 1.0f, true};
    aslm_update_device_state(nullptr, &state); // Should not crash
    PASS("update_device_state_null");
}

// ---------------------------------------------------------------------------
// Generation Tests (without model - should fail gracefully)
// ---------------------------------------------------------------------------

static void test_generate_null_context() {
    aslm_gen_params params = {64, 0.7f, 0.9f, 40, 1.1f};
    char output[256];
    int32_t tokens = aslm_generate(nullptr, "test", &params, output, sizeof(output));
    if (tokens == -1)
        PASS("generate_null_context");
    else
        FAIL("generate_null_context", "expected -1 for null context");
}

static void test_generate_null_prompt() {
    aslm_gen_params params = {64, 0.7f, 0.9f, 40, 1.1f};
    char output[256];
    int32_t tokens = aslm_generate(nullptr, nullptr, &params, output, sizeof(output));
    if (tokens == -1)
        PASS("generate_null_prompt");
    else
        FAIL("generate_null_prompt", "expected -1 for null prompt");
}

static void test_generate_null_output() {
    aslm_gen_params params = {64, 0.7f, 0.9f, 40, 1.1f};
    int32_t tokens = aslm_generate(nullptr, "test", &params, nullptr, 0);
    if (tokens == -1)
        PASS("generate_null_output");
    else
        FAIL("generate_null_output", "expected -1 for null output");
}

static void test_generate_null_params() {
    char output[256];
    int32_t tokens = aslm_generate(nullptr, "test", nullptr, output, sizeof(output));
    if (tokens == -1)
        PASS("generate_null_params");
    else
        FAIL("generate_null_params", "expected -1 for null params");
}

// ---------------------------------------------------------------------------
// Streaming Generation Tests
// ---------------------------------------------------------------------------

static void test_generate_stream_null_context() {
    aslm_gen_params params = {64, 0.7f, 0.9f, 40, 1.1f};
    int32_t tokens = aslm_generate_stream(nullptr, "test", &params, nullptr, nullptr);
    if (tokens == -1)
        PASS("generate_stream_null_context");
    else
        FAIL("generate_stream_null_context", "expected -1 for null context");
}

static void test_generate_stream_null_callback() {
    aslm_gen_params params = {64, 0.7f, 0.9f, 40, 1.1f};
    int32_t tokens = aslm_generate_stream(nullptr, "test", &params, nullptr, nullptr);
    if (tokens == -1)
        PASS("generate_stream_null_callback");
    else
        FAIL("generate_stream_null_callback", "expected -1 for null callback");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== Inference Unit Tests ===\n");

    test_default_init_params();
    test_init_null_model_path();
    test_init_invalid_model_path();
    test_get_effective_context_size_null();
    test_profile_create_free();
    test_profile_null_free();
    test_set_user_profile_null_context();
    test_get_memory_usage_null();
    test_get_peak_memory_usage_null();
    test_update_device_state_null();
    test_generate_null_context();
    test_generate_null_prompt();
    test_generate_null_output();
    test_generate_null_params();
    test_generate_stream_null_context();
    test_generate_stream_null_callback();

    std::printf("\n%s: %d failure(s)\n",
                failures == 0 ? "ALL PASSED" : "FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}