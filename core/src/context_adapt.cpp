/**
 * Adaptive Context Compression (ACC) Implementation
 * Novel Contribution #2
 */

#include "context_adapt.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <powerbase.h>
    #pragma comment(lib, "PowrProf.lib")
#endif

namespace aslm {

ContextAdapter::ContextAdapter(const ACCConfig& config)
    : config_(config)
    , current_context_(config.base_context)
{}

float ContextAdapter::computeResourceScore(const aslm_device_state& state) const {
    // RAM availability score
    float ram_ratio = 0.5f;
    if (state.total_ram_bytes > 0) {
        ram_ratio = static_cast<float>(state.available_ram_bytes) / 
                    static_cast<float>(state.total_ram_bytes);
    }
    
    // CPU availability score (inverse of usage)
    float cpu_avail = 1.0f - std::clamp(state.cpu_usage, 0.0f, 1.0f);
    
    // Battery score (if available)
    float battery_score = 1.0f;
    if (state.battery_level >= 0.0f) {
        battery_score = state.battery_level;
        if (state.is_charging) {
            battery_score = 1.0f;  // Full score when charging
        }
    }
    
    // Weighted combination
    float score = config_.ram_weight * ram_ratio +
                  config_.cpu_weight * cpu_avail +
                  config_.battery_weight * battery_score;
    
    return std::clamp(score, 0.0f, 1.0f);
}

int32_t ContextAdapter::computeContextSize(const aslm_device_state& state) const {
    float resource_score = computeResourceScore(state);
    
    // Handle extreme cases
    float ram_ratio = static_cast<float>(state.available_ram_bytes) / 
                      static_cast<float>(state.total_ram_bytes);
    
    if (ram_ratio < config_.low_ram_threshold) {
        // Emergency: use minimum context
        return config_.min_context;
    }
    
    if (ram_ratio > config_.high_ram_threshold && state.cpu_usage < 0.3f) {
        // Plenty of resources: use maximum context
        return config_.max_context;
    }
    
    // Linear interpolation based on resource score
    float range = static_cast<float>(config_.max_context - config_.min_context);
    int32_t context = config_.min_context + 
                      static_cast<int32_t>(resource_score * range);
    
    // Round to nearest power of 2 for efficiency
    int32_t powers[] = {128, 256, 512, 1024, 2048};
    int32_t best = config_.min_context;
    for (int32_t p : powers) {
        if (p <= context && p <= config_.max_context) {
            best = p;
        }
    }
    
    return best;
}

int32_t ContextAdapter::getSmoothedContextSize(const aslm_device_state& state) {
    int32_t new_context = computeContextSize(state);
    
    // Exponential moving average for smooth transitions
    float smoothed = ema_factor_ * static_cast<float>(new_context) +
                     (1.0f - ema_factor_) * static_cast<float>(current_context_);
    
    current_context_ = static_cast<int32_t>(smoothed);
    
    // Snap to power of 2
    int32_t powers[] = {128, 256, 512, 1024, 2048};
    int32_t best = 128;
    for (int32_t p : powers) {
        if (p <= current_context_) {
            best = p;
        }
    }
    current_context_ = best;
    
    return current_context_;
}

void ContextAdapter::reset() {
    current_context_ = config_.base_context;
}

// ============================================================================
// System resource monitoring — cross-platform
// ============================================================================

#ifdef _WIN32

aslm_device_state getCurrentDeviceState() {
    aslm_device_state state = {0, 0, 0.0f, -1.0f, false};

    // RAM via GlobalMemoryStatusEx
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    if (GlobalMemoryStatusEx(&mem_info)) {
        state.total_ram_bytes = mem_info.ullTotalPhys;
        state.available_ram_bytes = mem_info.ullAvailPhys;
    }

    // CPU via GetSystemTimes delta
    static ULARGE_INTEGER prev_idle_time = {0}, prev_kernel_time = {0}, prev_user_time = {0};
    FILETIME idle_ft, kernel_ft, user_ft;
    if (GetSystemTimes(&idle_ft, &kernel_ft, &user_ft)) {
        ULARGE_INTEGER idle, kernel, user;
        idle.LowPart = idle_ft.dwLowDateTime;   idle.HighPart = idle_ft.dwHighDateTime;
        kernel.LowPart = kernel_ft.dwLowDateTime; kernel.HighPart = kernel_ft.dwHighDateTime;
        user.LowPart = user_ft.dwLowDateTime;   user.HighPart = user_ft.dwHighDateTime;

        uint64_t idle_diff = idle.QuadPart - prev_idle_time.QuadPart;
        uint64_t total_diff = (kernel.QuadPart + user.QuadPart)
                            - (prev_kernel_time.QuadPart + prev_user_time.QuadPart);

        if (total_diff > 0) {
            state.cpu_usage = 1.0f - static_cast<float>(idle_diff) / static_cast<float>(total_diff);
            state.cpu_usage = std::clamp(state.cpu_usage, 0.0f, 1.0f);
        }

        prev_idle_time = idle;
        prev_kernel_time = kernel;
        prev_user_time = user;
    }

    // Battery via GetSystemPowerStatus
    SYSTEM_POWER_STATUS power;
    if (GetSystemPowerStatus(&power)) {
        if (power.BatteryLifePercent != 255) {
            state.battery_level = static_cast<float>(power.BatteryLifePercent) / 100.0f;
        }
        state.is_charging = (power.ACLineStatus == 1);
    }

    return state;
}

#else // Linux / POSIX

aslm_device_state getCurrentDeviceState() {
    aslm_device_state state = {0, 0, 0.0f, -1.0f, false};

    // RAM via /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line.substr(9));
                uint64_t kb; iss >> kb;
                state.total_ram_bytes = kb * 1024;
            } else if (line.find("MemAvailable:") == 0) {
                std::istringstream iss(line.substr(13));
                uint64_t kb; iss >> kb;
                state.available_ram_bytes = kb * 1024;
            }
        }
    }

    // CPU via /proc/stat delta
    static uint64_t prev_idle = 0, prev_total = 0;
    std::ifstream stat("/proc/stat");
    if (stat.is_open()) {
        std::string line;
        std::getline(stat, line);
        if (line.find("cpu ") == 0) {
            std::istringstream iss(line.substr(4));
            uint64_t user, nice, system, idle, iowait, irq, softirq;
            iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq;

            uint64_t total = user + nice + system + idle + iowait + irq + softirq;
            uint64_t total_diff = total - prev_total;
            uint64_t idle_diff = idle - prev_idle;

            if (total_diff > 0) {
                state.cpu_usage = 1.0f - static_cast<float>(idle_diff) /
                                         static_cast<float>(total_diff);
            }

            prev_idle = idle;
            prev_total = total;
        }
    }

    // Battery (Linux sysfs)
    std::ifstream battery("/sys/class/power_supply/BAT0/capacity");
    if (battery.is_open()) {
        int capacity;
        battery >> capacity;
        state.battery_level = static_cast<float>(capacity) / 100.0f;
    }

    std::ifstream charging("/sys/class/power_supply/BAT0/status");
    if (charging.is_open()) {
        std::string status;
        charging >> status;
        state.is_charging = (status == "Charging" || status == "Full");
    }

    return state;
}

#endif

} // namespace aslm
