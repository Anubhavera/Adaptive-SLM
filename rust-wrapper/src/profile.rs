//! User Profile management

use serde::{Deserialize, Serialize};
use crate::ffi::aslm_expertise_level;

/// User profile for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub age: i32,
    pub background: String,
    pub interests: Vec<String>,
    pub expertise: ExpertiseLevel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl From<ExpertiseLevel> for aslm_expertise_level {
    fn from(level: ExpertiseLevel) -> Self {
        match level {
            ExpertiseLevel::Beginner => aslm_expertise_level::Beginner,
            ExpertiseLevel::Intermediate => aslm_expertise_level::Intermediate,
            ExpertiseLevel::Advanced => aslm_expertise_level::Advanced,
            ExpertiseLevel::Expert => aslm_expertise_level::Expert,
        }
    }
}

impl UserProfile {
    /// Create a new user profile
    pub fn new(
        age: i32,
        background: &str,
        interests: Vec<String>,
        expertise: ExpertiseLevel,
    ) -> Self {
        Self {
            age,
            background: background.to_string(),
            interests,
            expertise,
        }
    }
    
    /// Compute relevance score for a topic based on profile
    pub fn compute_relevance(&self, topic: &str) -> f32 {
        let topic_lower = topic.to_lowercase();
        
        // Check interest match
        let mut max_interest_match = 0.0f32;
        for interest in &self.interests {
            let interest_lower = interest.to_lowercase();
            
            // Simple string similarity
            if topic_lower.contains(&interest_lower) || interest_lower.contains(&topic_lower) {
                max_interest_match = max_interest_match.max(0.9);
            } else {
                // Jaccard similarity on words
                let topic_words: std::collections::HashSet<_> = 
                    topic_lower.split_whitespace().collect();
                let interest_words: std::collections::HashSet<_> = 
                    interest_lower.split_whitespace().collect();
                
                let intersection = topic_words.intersection(&interest_words).count();
                let union = topic_words.union(&interest_words).count();
                
                if union > 0 {
                    let jaccard = intersection as f32 / union as f32;
                    max_interest_match = max_interest_match.max(jaccard);
                }
            }
        }
        
        // Check background match
        let bg_lower = self.background.to_lowercase();
        let bg_match = if topic_lower.contains(&bg_lower) || bg_lower.contains(&topic_lower) {
            0.8
        } else {
            0.3
        };
        
        // Combine: 70% interest, 30% background
        (0.7 * max_interest_match + 0.3 * bg_match).clamp(0.0, 1.0)
    }
    
    /// Get prompt modifier based on profile
    pub fn get_prompt_modifier(&self) -> String {
        let mut modifier = String::new();
        
        // Age-appropriate language
        if self.age < 10 {
            modifier.push_str("Explain in very simple terms for a young child. ");
        } else if self.age < 18 {
            modifier.push_str("Explain clearly for a teenager. ");
        } else {
            match self.expertise {
                ExpertiseLevel::Beginner => {
                    modifier.push_str("Explain in simple terms without jargon. ");
                }
                ExpertiseLevel::Intermediate => {
                    modifier.push_str("Explain with some technical detail. ");
                }
                ExpertiseLevel::Advanced => {
                    modifier.push_str("Provide detailed technical explanation. ");
                }
                ExpertiseLevel::Expert => {
                    modifier.push_str("Provide expert-level analysis with full technical depth. ");
                }
            }
        }
        
        // Background context
        if !self.background.is_empty() {
            modifier.push_str(&format!("Consider the user's background in {}. ", self.background));
        }
        
        // Interest hints
        if !self.interests.is_empty() {
            modifier.push_str(&format!("The user is interested in: {}. ", self.interests.join(", ")));
        }
        
        modifier
    }
    
    /// Save profile to JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
    
    /// Load profile from JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            age: 25,
            background: String::new(),
            interests: Vec::new(),
            expertise: ExpertiseLevel::Intermediate,
        }
    }
}
