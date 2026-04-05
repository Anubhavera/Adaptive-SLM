/**
 * User Profile Module
 * For Profile-Aware Knowledge Distillation (PAKD) and SCPD
 */

#ifndef USER_PROFILE_H
#define USER_PROFILE_H

#include "adaptive_slm.h"
#include <string>
#include <vector>

namespace aslm {

/**
 * User profile for personalization
 */
class UserProfile {
public:
    UserProfile(
        int32_t age,
        const std::string& background,
        const std::vector<std::string>& interests,
        aslm_expertise_level expertise
    );
    
    // Getters
    int32_t getAge() const { return age_; }
    const std::string& getBackground() const { return background_; }
    const std::vector<std::string>& getInterests() const { return interests_; }
    aslm_expertise_level getExpertise() const { return expertise_; }
    
    /**
     * Compute relevance score for a topic [0, 1]
     * Used by PAKD for distillation weighting
     * Used by SCPD for cache priority
     */
    float computeRelevance(
        const std::string& topic,
        const std::vector<float>& topic_embedding
    ) const;
    
    /**
     * Get prompt modifier based on profile
     * E.g., "Explain like I'm 10" vs technical jargon
     */
    std::string getPromptModifier() const;
    
    /**
     * Serialize to JSON
     */
    std::string toJson() const;
    
    /**
     * Deserialize from JSON
     */
    static UserProfile fromJson(const std::string& json);
    
private:
    int32_t age_;
    std::string background_;
    std::vector<std::string> interests_;
    aslm_expertise_level expertise_;
    
    // Pre-computed interest embeddings for fast relevance
    mutable std::vector<std::vector<float>> interest_embeddings_;
    mutable bool embeddings_computed_ = false;
    
    void computeInterestEmbeddings() const;
};

} // namespace aslm

#endif // USER_PROFILE_H
