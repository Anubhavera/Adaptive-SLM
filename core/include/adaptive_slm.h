/**
 * AdaptiveSLM - Research-Grade On-Device Language Model
 *
 * Novel contributions:
 * 1. Profile-Aware Knowledge Distillation (PAKD)
 * 2. Adaptive Context Compression (ACC)
 * 3. Semantic Cache with Priority Decay (SCPD)
 */

#ifndef ADAPTIVE_SLM_H
#define ADAPTIVE_SLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types
// ============================================================================

typedef struct aslm_context aslm_context;
typedef struct aslm_user_profile aslm_user_profile;

/**
 * Device state for Adaptive Context Compression (ACC)
 */
typedef struct {
    uint64_t available_ram_bytes;
    uint64_t total_ram_bytes;
    float cpu_usage;          // 0.0 - 1.0
    float battery_level;      // 0.0 - 1.0 (optional, -1 if unknown)
    bool is_charging;
} aslm_device_state;

/**
 * User expertise level for personalization
 */
typedef enum {
    ASLM_EXPERTISE_BEGINNER = 0,
    ASLM_EXPERTISE_INTERMEDIATE = 1,
    ASLM_EXPERTISE_ADVANCED = 2,
    ASLM_EXPERTISE_EXPERT = 3
} aslm_expertise_level;

/**
 * Generation parameters
 */
typedef struct {
    int32_t max_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    float repeat_penalty;
} aslm_gen_params;

/**
 * Initialization parameters
 */
typedef struct {
    const char* model_path;       // Path to GGUF model file
    int32_t n_threads;            // CPU threads (0 = auto-detect)
    int32_t max_context;          // Max context window (ACC will adjust dynamically)
    bool use_mmap;                // Memory-map model file for lower RSS
    bool verbose;                 // Log debug info to stderr/stdout
} aslm_init_params;

/**
 * Token streaming callback.
 * Called for each generated token. Return false to stop generation early.
 */
typedef bool (*aslm_token_callback)(const char* token_text, int32_t token_count, void* user_data);

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Get default initialization parameters.
 * Defaults: n_threads=auto, max_context=2048, use_mmap=true, verbose=false
 */
aslm_init_params aslm_default_init_params(void);

/**
 * Initialize the SLM context. Loads the GGUF model via llama.cpp.
 * @return Context pointer or NULL on failure
 */
aslm_context* aslm_init(const aslm_init_params* params);

/**
 * Free the context, model, and all resources
 */
void aslm_free(aslm_context* ctx);

/**
 * Generate text completion (blocking, returns full result)
 * @param ctx       Context
 * @param prompt    Input prompt (null-terminated)
 * @param params    Generation parameters (temperature, top_p, etc.)
 * @param output    Pre-allocated buffer for generated text
 * @param output_size Size of output buffer in bytes
 * @return Number of tokens generated, or -1 on error
 */
int32_t aslm_generate(
    aslm_context* ctx,
    const char* prompt,
    const aslm_gen_params* params,
    char* output,
    size_t output_size
);

/**
 * Generate text with token streaming (non-blocking per-token callback)
 * @param ctx        Context
 * @param prompt     Input prompt (null-terminated)
 * @param params     Generation parameters
 * @param callback   Called for each token; return false to stop early
 * @param user_data  Passed through to callback
 * @return Total tokens generated, or -1 on error
 */
int32_t aslm_generate_stream(
    aslm_context* ctx,
    const char* prompt,
    const aslm_gen_params* params,
    aslm_token_callback callback,
    void* user_data
);

// ============================================================================
// Adaptive Context Compression (ACC) - Novel Contribution #2
// ============================================================================

/**
 * Update device state for ACC.
 * Call periodically (e.g., before each generation) to let ACC adjust
 * the effective context window based on current device resources.
 */
void aslm_update_device_state(aslm_context* ctx, const aslm_device_state* state);

/**
 * Get current effective context size (adjusted by ACC)
 */
int32_t aslm_get_effective_context_size(const aslm_context* ctx);

// ============================================================================
// User Profile (for PAKD & SCPD) - Novel Contributions #1 & #3
// ============================================================================

/**
 * Create user profile for personalization
 */
aslm_user_profile* aslm_profile_create(
    int32_t age,
    const char* background,
    const char** interests,
    int32_t n_interests,
    aslm_expertise_level expertise
);

/**
 * Free user profile
 */
void aslm_profile_free(aslm_user_profile* profile);

/**
 * Set active user profile (enables personalized prompt modification)
 */
void aslm_set_user_profile(aslm_context* ctx, const aslm_user_profile* profile);

// ============================================================================
// Memory Info
// ============================================================================

/**
 * Get current memory usage in bytes (llama.cpp state size)
 */
uint64_t aslm_get_memory_usage(const aslm_context* ctx);

/**
 * Get peak memory usage estimate in bytes (state + model weights)
 */
uint64_t aslm_get_peak_memory_usage(const aslm_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_SLM_H
