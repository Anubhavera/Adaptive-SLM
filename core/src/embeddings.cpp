/**
 * SIMD-Optimized Embeddings Implementation
 */

#include "embeddings.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cctype>

// SIMD includes
#if defined(__AVX2__)
    #include <immintrin.h>
    #define ASLM_USE_AVX2 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define ASLM_USE_NEON 1
#endif

namespace aslm {

EmbeddingEngine::EmbeddingEngine(const EmbeddingConfig& config)
    : config_(config)
{
    // In a full implementation, we would load vocabulary embeddings from file
    // For now, use simple averaging of character-level features
}

EmbeddingEngine::~EmbeddingEngine() = default;

std::vector<int32_t> EmbeddingEngine::tokenize(const std::string& text) const {
    // Simple word-level tokenization
    std::vector<int32_t> tokens;
    std::string word;
    
    for (char c : text) {
        if (std::isalnum(c)) {
            word += std::tolower(c);
        } else if (!word.empty()) {
            // Hash word to get token ID
            uint32_t hash = 0;
            for (char wc : word) {
                hash = hash * 31 + wc;
            }
            tokens.push_back(static_cast<int32_t>(hash % 10000));
            word.clear();
        }
    }
    
    if (!word.empty()) {
        uint32_t hash = 0;
        for (char wc : word) {
            hash = hash * 31 + wc;
        }
        tokens.push_back(static_cast<int32_t>(hash % 10000));
    }
    
    return tokens;
}

std::vector<float> EmbeddingEngine::poolEmbeddings(
    const std::vector<int32_t>& token_ids
) const {
    std::vector<float> result(config_.dim, 0.0f);

    if (token_ids.empty()) {
        return result;
    }

    // Feature hashing with signed projections.
    // Each token hashes to a dimension and contributes +1 or -1.
    // This produces a sparse-ish vector where similar word sets cluster.
    for (int32_t token_id : token_ids) {
        uint64_t h = fnv1a(reinterpret_cast<const uint8_t*>(&token_id), sizeof(token_id));
        int32_t idx = static_cast<int32_t>(h % config_.dim);
        float sign = ((h >> 17) & 1) == 0 ? 1.0f : -1.0f;
        result[idx] += sign;
    }

    // Also hash consecutive token pairs (bigram features)
    for (size_t i = 0; i + 1 < token_ids.size(); ++i) {
        int32_t pair[2] = {token_ids[i], token_ids[i+1]};
        uint64_t h = fnv1a(reinterpret_cast<const uint8_t*>(pair), sizeof(pair));
        int32_t idx = static_cast<int32_t>(h % config_.dim);
        float sign = ((h >> 17) & 1) == 0 ? 1.0f : -1.0f;
        result[idx] += sign * 0.7f;
    }

    if (config_.normalize) {
        normalize(result.data(), config_.dim);
    }

    return result;
}

uint64_t EmbeddingEngine::fnv1a(const uint8_t* data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

std::vector<float> EmbeddingEngine::embed(const std::string& text) const {
    auto tokens = tokenize(text);
    return poolEmbeddings(tokens);
}

std::vector<std::vector<float>> EmbeddingEngine::embedBatch(
    const std::vector<std::string>& texts
) const {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(embed(text));
    }
    
    return results;
}

// ============================================================================
// SIMD-Optimized Vector Operations
// ============================================================================

#if ASLM_USE_AVX2

float EmbeddingEngine::dotProduct(const float* a, const float* b, int32_t dim) {
    __m256 sum = _mm256_setzero_ps();
    
    int32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#elif ASLM_USE_NEON

float EmbeddingEngine::dotProduct(const float* a, const float* b, int32_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    
    int32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }
    
    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    
    // Handle remaining
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#else

// Fallback scalar implementation
float EmbeddingEngine::dotProduct(const float* a, const float* b, int32_t dim) {
    float result = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

#endif

float EmbeddingEngine::cosineSimilarity(
    const float* a,
    const float* b,
    int32_t dim
) {
    float dot = dotProduct(a, b, dim);
    
    float norm_a = 0.0f, norm_b = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-8f) {
        return 0.0f;
    }
    
    return dot / denom;
}

void EmbeddingEngine::normalize(float* vec, int32_t dim) {
    float norm = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    
    norm = std::sqrt(norm);
    if (norm < 1e-8f) {
        return;
    }
    
    float inv_norm = 1.0f / norm;
    for (int32_t i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

} // namespace aslm
