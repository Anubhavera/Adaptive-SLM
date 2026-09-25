/**
 * User Profile Implementation
 */

#include "user_profile.h"
#include "embeddings.h"
#include <sstream>
#include <algorithm>

namespace aslm {

UserProfile::UserProfile(
    int32_t age,
    const std::string& background,
    const std::vector<std::string>& interests,
    aslm_expertise_level expertise
)
    : age_(age)
    , background_(background)
    , interests_(interests)
    , expertise_(expertise)
{}

void UserProfile::computeInterestEmbeddings() const {
    if (embeddings_computed_) {
        return;
    }
    
    EmbeddingEngine emb;
    interest_embeddings_ = emb.embedBatch(interests_);
    embeddings_computed_ = true;
}

float UserProfile::computeRelevance(
    const std::string& topic,
    const std::vector<float>& topic_embedding
) const {
    if (interests_.empty()) {
        return 0.5f;  // Neutral relevance
    }
    
    computeInterestEmbeddings();
    
    // Find max similarity to any interest
    float max_sim = 0.0f;
    for (const auto& interest_emb : interest_embeddings_) {
        float sim = EmbeddingEngine::cosineSimilarity(
            topic_embedding.data(),
            interest_emb.data(),
            static_cast<int32_t>(topic_embedding.size())
        );
        max_sim = std::max(max_sim, sim);
    }
    
    // Also boost if topic matches background
    EmbeddingEngine emb;
    auto bg_emb = emb.embed(background_);
    float bg_sim = EmbeddingEngine::cosineSimilarity(
        topic_embedding.data(),
        bg_emb.data(),
        static_cast<int32_t>(topic_embedding.size())
    );
    
    // Interest score: cosine rescaled so a typical containment match
    // (~0.75) maps to 0.9 — the score rust-wrapper/src/profile.rs gives
    // for a substring interest match. No overlap -> 0.
    float interest_score = std::clamp(max_sim / 0.75f, 0.0f, 1.0f);
    
    // Background score: 0.3 prior with no evidence (mirrors rust-wrapper)
    float bg_score = std::clamp(0.3f + 0.7f * bg_sim, 0.0f, 1.0f);
    
    // Combine: 70% interest match, 30% background match
    float relevance = 0.7f * interest_score + 0.3f * bg_score;
    
    // Normalize to [0, 1]
    return std::clamp(relevance, 0.0f, 1.0f);
}

std::string UserProfile::getPromptModifier() const {
    std::ostringstream oss;
    
    // Age-appropriate language
    if (age_ < 10) {
        oss << "Explain in very simple terms that a young child can understand. ";
    } else if (age_ < 18) {
        oss << "Explain clearly for a teenager. ";
    } else {
        // Adult - adjust by expertise
        switch (expertise_) {
            case ASLM_EXPERTISE_BEGINNER:
                oss << "Explain in simple terms without jargon. ";
                break;
            case ASLM_EXPERTISE_INTERMEDIATE:
                oss << "Explain with some technical detail. ";
                break;
            case ASLM_EXPERTISE_ADVANCED:
                oss << "Provide detailed technical explanation. ";
                break;
            case ASLM_EXPERTISE_EXPERT:
                oss << "Provide expert-level analysis with full technical depth. ";
                break;
        }
    }
    
    // Background context
    if (!background_.empty()) {
        oss << "Consider the user's background in " << background_ << ". ";
    }
    
    // Interest hints
    if (!interests_.empty()) {
        oss << "The user is interested in: ";
        for (size_t i = 0; i < interests_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << interests_[i];
        }
        oss << ". ";
    }
    
    return oss.str();
}

std::string UserProfile::toJson() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"age\":" << age_ << ",";
    oss << "\"background\":\"" << background_ << "\",";
    oss << "\"expertise\":" << static_cast<int>(expertise_) << ",";
    oss << "\"interests\":[";
    for (size_t i = 0; i < interests_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << interests_[i] << "\"";
    }
    oss << "]}";
    return oss.str();
}

UserProfile UserProfile::fromJson(const std::string& json) {
    // Simplified JSON parsing (in production, use a proper JSON library)
    int32_t age = 25;
    std::string background;
    std::vector<std::string> interests;
    aslm_expertise_level expertise = ASLM_EXPERTISE_INTERMEDIATE;
    
    // Parse age
    auto age_pos = json.find("\"age\":");
    if (age_pos != std::string::npos) {
        age = std::stoi(json.substr(age_pos + 6));
    }
    
    // Parse background
    auto bg_pos = json.find("\"background\":\"");
    if (bg_pos != std::string::npos) {
        auto start = bg_pos + 14;
        auto end = json.find("\"", start);
        if (end != std::string::npos) {
            background = json.substr(start, end - start);
        }
    }
    
    // Parse expertise
    auto exp_pos = json.find("\"expertise\":");
    if (exp_pos != std::string::npos) {
        expertise = static_cast<aslm_expertise_level>(
            std::stoi(json.substr(exp_pos + 12))
        );
    }

    // Parse interests array: ["item1","item2",...]
    auto int_pos = json.find("\"interests\":[");
    if (int_pos != std::string::npos) {
        auto arr_start = int_pos + 13; // after "interests":[
        auto arr_end = json.find(']', arr_start);
        if (arr_end != std::string::npos) {
            std::string arr = json.substr(arr_start, arr_end - arr_start);
            size_t pos = 0;
            while (pos < arr.size()) {
                auto q1 = arr.find('"', pos);
                if (q1 == std::string::npos) break;
                auto q2 = arr.find('"', q1 + 1);
                if (q2 == std::string::npos) break;
                interests.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                pos = q2 + 1;
            }
        }
    }

    return UserProfile(age, background, interests, expertise);
}

} // namespace aslm

// ============================================================================
// C API Wrappers
// ============================================================================

extern "C" {

aslm_user_profile* aslm_profile_create(
    int32_t age,
    const char* background,
    const char** interests,
    int32_t n_interests,
    aslm_expertise_level expertise
) {
    std::vector<std::string> interest_vec;
    for (int32_t i = 0; i < n_interests; ++i) {
        interest_vec.emplace_back(interests[i]);
    }
    
    auto* profile = new aslm::UserProfile(
        age,
        background ? background : "",
        interest_vec,
        expertise
    );
    
    return reinterpret_cast<aslm_user_profile*>(profile);
}

void aslm_profile_free(aslm_user_profile* profile) {
    if (profile) {
        delete reinterpret_cast<aslm::UserProfile*>(profile);
    }
}

}
