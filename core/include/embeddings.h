/**
 * Embeddings Module
 * SIMD-optimized vector operations for semantic search
 */

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <vector>
#include <string>
#include <cstdint>

namespace aslm {

/**
 * Embedding configuration
 */
struct EmbeddingConfig {
    int32_t dim = 384;           // all-MiniLM-L6-v2 dimension
    bool normalize = true;       // L2 normalize embeddings
};

/**
 * Lightweight embedding generator
 * Uses pre-computed vocabulary embeddings for speed
 */
class EmbeddingEngine {
public:
    explicit EmbeddingEngine(const EmbeddingConfig& config = EmbeddingConfig{});
    ~EmbeddingEngine();
    
    /**
     * Generate embedding for text
     * @return 384-dim float vector
     */
    std::vector<float> embed(const std::string& text) const;
    
    /**
     * Batch embedding for efficiency
     */
    std::vector<std::vector<float>> embedBatch(
        const std::vector<std::string>& texts
    ) const;
    
    /**
     * Compute cosine similarity between two embeddings
     * SIMD-optimized
     */
    static float cosineSimilarity(
        const float* a, 
        const float* b, 
        int32_t dim
    );
    
    /**
     * Compute dot product (for normalized vectors)
     * SIMD-optimized with AVX2/NEON
     */
    static float dotProduct(
        const float* a,
        const float* b,
        int32_t dim
    );
    
    /**
     * L2 normalize vector in-place
     */
    static void normalize(float* vec, int32_t dim);
    
    int32_t getDim() const { return config_.dim; }
    
private:
    EmbeddingConfig config_;
    
    // Vocabulary embeddings (loaded from file)
    std::vector<float> vocab_embeddings_;
    std::vector<std::string> vocab_;
    
    /**
     * Simple tokenizer (word-level)
     */
    std::vector<int32_t> tokenize(const std::string& text) const;
    
    /**
     * Average pooling over token embeddings
     */
    std::vector<float> poolEmbeddings(
        const std::vector<int32_t>& token_ids
    ) const;

    /**
     * FNV-1a hash for feature hashing
     */
    static uint64_t fnv1a(const uint8_t* data, size_t len);
};

} // namespace aslm

#endif // EMBEDDINGS_H
