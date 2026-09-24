/**
 * Adaptive Context Compression (ACC) Module
 * Novel Contribution #2
 * 
 * Dynamically adjusts context window based on real-time device constraints.
 * First work to use runtime device telemetry for LLM context adaptation.
 */

#ifndef CONTEXT_ADAPT_H
#define CONTEXT_ADAPT_H

#include "adaptive_slm.h"
#include <cstdint>

namespace aslm {

/**
 * ACC Configuration
 */
struct ACCConfig {
    int32_t min_context = 128;      // Minimum context window
    int32_t max_context = 2048;     // Maximum context window
    int32_t base_context = 512;     // Base context size
    
    float ram_weight = 0.4f;        // Weight for RAM availability
    float cpu_weight = 0.3f;        // Weight for CPU availability  
    float battery_weight = 0.3f;    // Weight for battery (mobile)
    
    float low_ram_threshold = 0.2f; // Below this, use minimum context
    float high_ram_threshold = 0.7f;// Above this, can use maximum
};

/**
 * Adaptive Context Compression Engine
 */
class ContextAdapter {
public:
    explicit ContextAdapter(const ACCConfig& config = ACCConfig{});
    
    /**
     * Compute optimal context size based on device state
     * Core ACC algorithm
     */
    int32_t computeContextSize(const aslm_device_state& state) const;
    
    /**
     * Get adaptive context with smoothing (prevents rapid changes)
     */
    int32_t getSmoothedContextSize(const aslm_device_state& state);
    
    /**
     * Reset smoothing history
     */
    void reset();
    
    /**
     * Compute resource availability score [0, 1]
     * Public so unit tests can verify scoring independently.
     */
    float computeResourceScore(const aslm_device_state& state) const;

    // Getters
    int32_t getCurrentContextSize() const { return current_context_; }
    const ACCConfig& getConfig() const { return config_; }

private:
    ACCConfig config_;
    int32_t current_context_;
    float ema_factor_ = 0.3f;
};

/**
 * Utility: Get current device state (Linux implementation)
 */
aslm_device_state getCurrentDeviceState();

} // namespace aslm

#endif // CONTEXT_ADAPT_H
