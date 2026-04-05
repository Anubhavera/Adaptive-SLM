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
    const char* model_path;       // Path to GGUF model
    int32_t n_threads;            // CPU threads (0 = auto)
    int32_t max_context;          // Max context window (ACC will adjust)
    bool use_mmap;                // Memory-map model file
    bool verbose;                 // Log debug info
} aslm_init_params;

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Get default initialization parameters
 */
aslm_init_params aslm_default_init_params(void);

/**
 * Initialize the SLM context
 * @return Context pointer or NULL on failure
 */
aslm_context* aslm_init(const aslm_init_params* params);

/**
 * Free the context and all resources
 */
void aslm_free(aslm_context* ctx);

/**
 * Generate text completion
 * @param ctx Context
 * @param prompt Input prompt
 * @param params Generation parameters
 * @param output Buffer for output (must be pre-allocated)
 * @param output_size Size of output buffer
 * @return Number of tokens generated, or -1 on error
 */
int32_t aslm_generate(
    aslm_context* ctx,
    const char* prompt,
    const aslm_gen_params* params,
    char* output,
    size_t output_size
);

// ============================================================================
// Adaptive Context Compression (ACC) - Novel Contribution #2
// ============================================================================

/**
 * Update device state for ACC
 * Call this periodically to let ACC adjust context window
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
 * Create user profile
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
 * Set active user profile (enables personalization)
 */
void aslm_set_user_profile(aslm_context* ctx, const aslm_user_profile* profile);

// ============================================================================
// Memory Info
// ============================================================================

/**
 * Get current memory usage in bytes
 */
uint64_t aslm_get_memory_usage(const aslm_context* ctx);

/**
 * Get peak memory usage in bytes
 */
uint64_t aslm_get_peak_memory_usage(const aslm_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_SLM_H
