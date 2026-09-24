/**
 * Unit tests for EmbeddingEngine (FNV-1a feature hashing + SIMD ops)
 */

#include "embeddings.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#define PASS(name) std::printf("[PASS] %s\n", name)
#define FAIL(name, msg) (std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg), ++failures)

static int failures = 0;

// ---------------------------------------------------------------------------
// embed() tests
// ---------------------------------------------------------------------------

static void test_embed_dimension() {
    aslm::EmbeddingEngine eng;
    auto v = eng.embed("hello world");
    if ((int)v.size() == eng.getDim())
        PASS("embed_dimension");
    else
        FAIL("embed_dimension", "wrong dimension");
}

static void test_embed_normalized() {
    aslm::EmbeddingEngine eng;
    auto v = eng.embed("machine learning");
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (std::fabs(norm - 1.0f) < 1e-5f)
        PASS("embed_normalized");
    else
        FAIL("embed_normalized", "vector not unit-normalized");
}

static void test_embed_empty_string() {
    aslm::EmbeddingEngine eng;
    auto v = eng.embed("");
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    // Empty string should return zero vector (norm == 0)
    if (norm < 1e-8f)
        PASS("embed_empty_string");
    else
        FAIL("embed_empty_string", "empty string should produce zero vector");
}

static void test_similar_texts_closer() {
    aslm::EmbeddingEngine eng;
    auto v1 = eng.embed("machine learning neural network");
    auto v2 = eng.embed("machine learning deep learning");
    auto v3 = eng.embed("photosynthesis plant biology");

    float sim_related = aslm::EmbeddingEngine::cosineSimilarity(
        v1.data(), v2.data(), eng.getDim());
    float sim_unrelated = aslm::EmbeddingEngine::cosineSimilarity(
        v1.data(), v3.data(), eng.getDim());

    if (sim_related > sim_unrelated)
        PASS("similar_texts_closer");
    else
        FAIL("similar_texts_closer", "related texts should have higher cosine similarity");
}

static void test_identical_texts_sim_one() {
    aslm::EmbeddingEngine eng;
    auto v = eng.embed("artificial intelligence");
    float sim = aslm::EmbeddingEngine::cosineSimilarity(
        v.data(), v.data(), eng.getDim());
    if (std::fabs(sim - 1.0f) < 1e-5f)
        PASS("identical_texts_sim_one");
    else
        FAIL("identical_texts_sim_one", "cosine similarity of vector with itself must be 1.0");
}

static void test_batch_matches_single() {
    aslm::EmbeddingEngine eng;
    std::vector<std::string> texts = {"hello", "world", "machine learning"};
    auto batch = eng.embedBatch(texts);
    for (size_t i = 0; i < texts.size(); ++i) {
        auto single = eng.embed(texts[i]);
        float sim = aslm::EmbeddingEngine::cosineSimilarity(
            batch[i].data(), single.data(), eng.getDim());
        if (std::fabs(sim - 1.0f) > 1e-4f) {
            FAIL("batch_matches_single", "batch result differs from single embed");
            return;
        }
    }
    PASS("batch_matches_single");
}

// ---------------------------------------------------------------------------
// SIMD dotProduct tests
// ---------------------------------------------------------------------------

static void test_dot_product_basic() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float expected = 1*4 + 2*5 + 3*6;  // = 32
    float got = aslm::EmbeddingEngine::dotProduct(a.data(), b.data(), 8);
    if (std::fabs(got - expected) < 1e-4f)
        PASS("dot_product_basic");
    else
        FAIL("dot_product_basic", "wrong dot product value");
}

static void test_dot_product_large() {
    const int DIM = 384;
    std::vector<float> a(DIM), b(DIM);
    float expected = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        a[i] = static_cast<float>(i) / DIM;
        b[i] = static_cast<float>(DIM - i) / DIM;
        expected += a[i] * b[i];
    }
    float got = aslm::EmbeddingEngine::dotProduct(a.data(), b.data(), DIM);
    if (std::fabs(got - expected) < 1e-3f)
        PASS("dot_product_large_384");
    else
        FAIL("dot_product_large_384", "SIMD result differs from scalar");
}

static void test_cosine_orthogonal() {
    // Two orthogonal unit vectors → cosine = 0
    std::vector<float> a(8, 0.0f), b(8, 0.0f);
    a[0] = 1.0f;
    b[1] = 1.0f;
    float sim = aslm::EmbeddingEngine::cosineSimilarity(a.data(), b.data(), 8);
    if (std::fabs(sim) < 1e-6f)
        PASS("cosine_orthogonal");
    else
        FAIL("cosine_orthogonal", "orthogonal vectors should have cosine sim = 0");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== Embeddings Unit Tests ===\n");

#if ASLM_USE_AVX2
    std::printf("SIMD backend: AVX2\n");
#elif ASLM_USE_NEON
    std::printf("SIMD backend: NEON\n");
#else
    std::printf("SIMD backend: scalar\n");
#endif

    test_embed_dimension();
    test_embed_normalized();
    test_embed_empty_string();
    test_similar_texts_closer();
    test_identical_texts_sim_one();
    test_batch_matches_single();
    test_dot_product_basic();
    test_dot_product_large();
    test_cosine_orthogonal();

    std::printf("\n%s: %d failure(s)\n",
                failures == 0 ? "ALL PASSED" : "FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
