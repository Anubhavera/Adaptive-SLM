/**
 * Unit tests for UserProfile (PAKD)
 */

#include "user_profile.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define PASS(name) std::printf("[PASS] %s\n", name)
#define FAIL(name, msg) (std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg), ++failures)

static int failures = 0;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

static aslm::UserProfile make_profile(
    int32_t age = 25,
    const char* background = "software engineer",
    const char** interests = nullptr,
    int32_t n_interests = 0,
    aslm_expertise_level expertise = ASLM_EXPERTISE_INTERMEDIATE
) {
    return aslm::UserProfile(age, background ? background : "", 
        interests ? std::vector<std::string>(interests, interests + n_interests) 
                  : std::vector<std::string>{}, 
        expertise);
}

// ---------------------------------------------------------------------------
// Prompt Modifier Tests
// ---------------------------------------------------------------------------

static void test_prompt_modifier_age() {
    // Child
    auto p = make_profile(8, "student", nullptr, 0, ASLM_EXPERTISE_BEGINNER);
    std::string mod = p.getPromptModifier();
    if (mod.find("young child") != std::string::npos)
        PASS("prompt_modifier_child");
    else
        FAIL("prompt_modifier_child", "missing child language");

    // Teenager
    p = make_profile(15, "student", nullptr, 0, ASLM_EXPERTISE_BEGINNER);
    mod = p.getPromptModifier();
    if (mod.find("teenager") != std::string::npos)
        PASS("prompt_modifier_teen");
    else
        FAIL("prompt_modifier_teen", "missing teenager language");

    // Adult beginner
    p = make_profile(25, "engineer", nullptr, 0, ASLM_EXPERTISE_BEGINNER);
    mod = p.getPromptModifier();
    if (mod.find("simple terms") != std::string::npos)
        PASS("prompt_modifier_beginner");
    else
        FAIL("prompt_modifier_beginner", "missing beginner language");

    // Adult expert
    p = make_profile(35, "scientist", nullptr, 0, ASLM_EXPERTISE_EXPERT);
    mod = p.getPromptModifier();
    if (mod.find("expert-level") != std::string::npos)
        PASS("prompt_modifier_expert");
    else
        FAIL("prompt_modifier_expert", "missing expert language");
}

static void test_prompt_modifier_background() {
    const char* interests[] = {"programming"};
    auto p = make_profile(25, "medical doctor", interests, 1, ASLM_EXPERTISE_ADVANCED);
    std::string mod = p.getPromptModifier();
    if (mod.find("medical doctor") != std::string::npos)
        PASS("prompt_modifier_background");
    else
        FAIL("prompt_modifier_background", "background not in modifier");
}

static void test_prompt_modifier_interests() {
    const char* interests[] = {"programming", "machine learning", "science"};
    auto p = make_profile(25, "engineer", interests, 3, ASLM_EXPERTISE_ADVANCED);
    std::string mod = p.getPromptModifier();
    if (mod.find("programming") != std::string::npos &&
        mod.find("machine learning") != std::string::npos &&
        mod.find("science") != std::string::npos)
        PASS("prompt_modifier_interests");
    else
        FAIL("prompt_modifier_interests", "interests not in modifier");
}

// ---------------------------------------------------------------------------
// Relevance Computation Tests
// ---------------------------------------------------------------------------

static void test_compute_relevance_empty_interests() {
    auto p = make_profile(25, "engineer", nullptr, 0, ASLM_EXPERTISE_INTERMEDIATE);
    aslm::EmbeddingEngine eng;
    auto emb = eng.embed("machine learning");
    float rel = p.computeRelevance("machine learning", emb);
    // With empty interests, should return neutral (0.5)
    if (rel >= 0.4f && rel <= 0.6f)
        PASS("relevance_empty_interests");
    else
        FAIL("relevance_empty_interests", "expected neutral relevance ~0.5");
}

static void test_compute_relevance_matching_interests() {
    const char* interests[] = {"programming", "machine learning"};
    auto p = make_profile(25, "engineer", interests, 2, ASLM_EXPERTISE_ADVANCED);
    aslm::EmbeddingEngine eng;
    auto emb = eng.embed("machine learning neural networks");
    float rel = p.computeRelevance("machine learning neural networks", emb);
    // Should be high relevance (> 0.7)
    if (rel > 0.7f)
        PASS("relevance_matching_interests");
    else
        FAIL("relevance_matching_interests", "expected high relevance for matching topic");
}

static void test_compute_relevance_non_matching_interests() {
    const char* interests[] = {"cooking", "gardening"};
    auto p = make_profile(25, "chef", interests, 2, ASLM_EXPERTISE_INTERMEDIATE);
    aslm::EmbeddingEngine eng;
    auto emb = eng.embed("quantum computing algorithms");
    float rel = p.computeRelevance("quantum computing algorithms", emb);
    // Should be low relevance (< 0.5)
    if (rel < 0.5f)
        PASS("relevance_non_matching_interests");
    else
        FAIL("relevance_non_matching_interests", "expected low relevance for unrelated topic");
}

// ---------------------------------------------------------------------------
// JSON Serialization Tests
// ---------------------------------------------------------------------------

static void test_json_serialization() {
    const char* interests[] = {"programming", "ai"};
    auto p = make_profile(30, "researcher", interests, 2, ASLM_EXPERTISE_EXPERT);
    std::string json = p.toJson();
    
    if (json.find("\"age\":30") != std::string::npos &&
        json.find("\"background\":\"researcher\"") != std::string::npos &&
        json.find("\"expertise\":4") != std::string::npos &&
        json.find("\"programming\"") != std::string::npos &&
        json.find("\"ai\"") != std::string::npos)
        PASS("json_serialization");
    else
        FAIL("json_serialization", "JSON missing expected fields");
}

static void test_json_deserialization() {
    std::string json = R"({"age":28,"background":"designer","expertise":1,"interests":["ui","ux"]})";
    auto p = aslm::UserProfile::fromJson(json);
    
    if (p.age_ == 28 &&
        p.background_ == "designer" &&
        p.expertise_ == ASLM_EXPERTISE_INTERMEDIATE &&
        p.interests_.size() == 2 &&
        p.interests_[0] == "ui" &&
        p.interests_[1] == "ux")
        PASS("json_deserialization");
    else
        FAIL("json_deserialization", "deserialization failed");
}

// ---------------------------------------------------------------------------
// C API Tests
// ---------------------------------------------------------------------------

static void test_c_api_create_free() {
    const char* interests[] = {"coding", "math"};
    aslm_user_profile* profile = aslm_profile_create(
        25, "engineer", interests, 2, ASLM_EXPERTISE_ADVANCED
    );
    if (profile != nullptr) {
        aslm_profile_free(profile);
        PASS("c_api_create_free");
    } else {
        FAIL("c_api_create_free", "profile creation returned null");
    }
}

static void test_c_api_null_free() {
    aslm_profile_free(nullptr); // Should not crash
    PASS("c_api_null_free");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== UserProfile Unit Tests ===\n");

    test_prompt_modifier_age();
    test_prompt_modifier_background();
    test_prompt_modifier_interests();
    test_compute_relevance_empty_interests();
    test_compute_relevance_matching_interests();
    test_compute_relevance_non_matching_interests();
    test_json_serialization();
    test_json_deserialization();
    test_c_api_create_free();
    test_c_api_null_free();

    std::printf("\n%s: %d failure(s)\n",
                failures == 0 ? "ALL PASSED" : "FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}