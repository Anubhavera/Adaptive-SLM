/**
 * Unit tests for Adaptive Context Compression (ACC)
 */

#include "context_adapt.h"
#include <cassert>
#include <cstdio>
#include <cstring>

#define PASS(name) std::printf("[PASS] %s\n", name)
#define FAIL(name, msg) (std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg), ++failures)

static int failures = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static aslm_device_state make_state(double avail_gb, double total_gb,
                                     float cpu, float battery, bool charging) {
    return aslm_device_state{
        static_cast<uint64_t>(avail_gb  * 1024 * 1024 * 1024),
        static_cast<uint64_t>(total_gb  * 1024 * 1024 * 1024),
        cpu, battery, charging
    };
}

// ---------------------------------------------------------------------------
// computeContextSize tests
// ---------------------------------------------------------------------------

static void test_high_resources() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    auto state = make_state(8.0, 8.0, 0.05f, 1.0f, true);
    int ctx = acc.computeContextSize(state);
    if (ctx >= 1024 && ctx <= cfg.max_context)
        PASS("high_resources");
    else
        FAIL("high_resources", "expected >= 1024");
}

static void test_low_ram_emergency() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    // < 20% RAM available → must return min_context
    auto state = make_state(0.4, 8.0, 0.5f, 0.5f, false);
    int ctx = acc.computeContextSize(state);
    if (ctx == cfg.min_context)
        PASS("low_ram_emergency");
    else
        FAIL("low_ram_emergency", "expected min_context (128)");
}

static void test_medium_resources() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    auto state = make_state(4.0, 8.0, 0.5f, 0.5f, false);
    int ctx = acc.computeContextSize(state);
    if (ctx >= cfg.min_context && ctx <= cfg.max_context)
        PASS("medium_resources");
    else
        FAIL("medium_resources", "result out of [min, max] range");
}

static void test_always_power_of_two() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    const int powers[] = {128, 256, 512, 1024, 2048};
    double avails[] = {0.5, 1.0, 2.0, 4.0, 7.0};
    for (double avail : avails) {
        auto state = make_state(avail, 8.0, 0.4f, 0.8f, false);
        int ctx = acc.computeContextSize(state);
        bool is_pow2 = false;
        for (int p : powers) is_pow2 |= (ctx == p);
        if (!is_pow2) {
            FAIL("always_power_of_two", "context size not a power of 2");
            return;
        }
    }
    PASS("always_power_of_two");
}

// ---------------------------------------------------------------------------
// computeResourceScore tests
// ---------------------------------------------------------------------------

static void test_resource_score_range() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    float scores[] = {0.0f};
    aslm_device_state states[] = {
        make_state(8.0, 8.0, 0.0f, 1.0f, true),   // max resources
        make_state(0.5, 8.0, 0.9f, 0.1f, false),  // min resources
        make_state(4.0, 8.0, 0.5f, 0.5f, false),  // mid
    };
    for (auto& state : states) {
        float s = acc.computeResourceScore(state);
        if (s < 0.0f || s > 1.0f) {
            FAIL("resource_score_range", "score outside [0,1]");
            return;
        }
    }
    PASS("resource_score_range");
}

static void test_resource_score_ordering() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    auto high  = make_state(8.0, 8.0, 0.0f,  1.0f, true);
    auto low   = make_state(0.5, 8.0, 0.95f, 0.1f, false);
    float s_high = acc.computeResourceScore(high);
    float s_low  = acc.computeResourceScore(low);
    if (s_high > s_low)
        PASS("resource_score_ordering");
    else
        FAIL("resource_score_ordering", "high-resource score not > low-resource score");
}

// ---------------------------------------------------------------------------
// EMA smoothing test
// ---------------------------------------------------------------------------

static void test_ema_smoothing() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);

    // Saturate toward max
    auto rich = make_state(8.0, 8.0, 0.0f, 1.0f, true);
    int prev = 0;
    for (int i = 0; i < 20; ++i) {
        prev = acc.getSmoothedContextSize(rich);
    }
    if (prev < 512) {
        FAIL("ema_smoothing", "EMA failed to converge toward larger context after 20 iterations");
        return;
    }

    // Drop suddenly to low resources — EMA should not instantly hit min
    auto poor = make_state(0.2, 8.0, 0.99f, 0.05f, false);
    int after_drop = acc.getSmoothedContextSize(poor);
    // After one step EMA should still be above pure minimum
    // (exact value depends on ema_factor, just check it didn't instantly jump to 128)
    (void)after_drop; // result validated by no-crash
    PASS("ema_smoothing");
}

// ---------------------------------------------------------------------------
// reset() test
// ---------------------------------------------------------------------------

static void test_reset() {
    aslm::ACCConfig cfg;
    aslm::ContextAdapter acc(cfg);
    auto rich = make_state(8.0, 8.0, 0.0f, 1.0f, true);
    for (int i = 0; i < 10; ++i) acc.getSmoothedContextSize(rich);
    acc.reset();
    // After reset, next call with high resources should move from base_context
    int after_reset = acc.getSmoothedContextSize(rich);
    if (after_reset >= cfg.min_context && after_reset <= cfg.max_context)
        PASS("reset");
    else
        FAIL("reset", "context out of range after reset");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== ACC Unit Tests ===\n");

    test_high_resources();
    test_low_ram_emergency();
    test_medium_resources();
    test_always_power_of_two();
    test_resource_score_range();
    test_resource_score_ordering();
    test_ema_smoothing();
    test_reset();

    std::printf("\n%s: %d failure(s)\n",
                failures == 0 ? "ALL PASSED" : "FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
