//! User Profile management

use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::ptr::NonNull;
use crate::ffi;
use crate::{Result, SLMError};

/// Alias for [`ExpertiseLevel`] used by the model-family selector API.
pub type Expertise = ExpertiseLevel;

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

impl From<ExpertiseLevel> for ffi::aslm_expertise_level {
    fn from(level: ExpertiseLevel) -> Self {
        match level {
            ExpertiseLevel::Beginner => ffi::aslm_expertise_level::Beginner,
            ExpertiseLevel::Intermediate => ffi::aslm_expertise_level::Intermediate,
            ExpertiseLevel::Advanced => ffi::aslm_expertise_level::Advanced,
            ExpertiseLevel::Expert => ffi::aslm_expertise_level::Expert,
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

/// RAII handle to a user profile allocated in the C++ core
/// (`aslm_profile_create` / `aslm_profile_free`).
///
/// The engine stores a raw pointer when the profile is set
/// (`aslm_set_user_profile` does not copy), so the handle must stay alive
/// while the engine uses it. `AdaptiveSLM::set_engine_profile` takes
/// ownership and keeps the profile alive until it is cleared or the engine
/// is dropped.
pub struct CUserProfile {
    ptr: NonNull<ffi::aslm_user_profile>,
}

impl CUserProfile {
    /// Allocate the profile object in the C++ core.
    pub fn new(profile: &UserProfile) -> Result<Self> {
        let background = CString::new(profile.background.as_str())
            .map_err(|_| SLMError::ProfileError("background contains a NUL byte".into()))?;

        let interests: Vec<CString> = profile
            .interests
            .iter()
            .map(|i| CString::new(i.as_str()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| SLMError::ProfileError("interest contains a NUL byte".into()))?;

        let interest_ptrs: Vec<*const std::ffi::c_char> =
            interests.iter().map(|s| s.as_ptr()).collect();

        let ptr = unsafe {
            ffi::aslm_profile_create(
                profile.age,
                background.as_ptr(),
                if interest_ptrs.is_empty() {
                    std::ptr::null()
                } else {
                    interest_ptrs.as_ptr()
                },
                interest_ptrs.len() as i32,
                profile.expertise.into(),
            )
        };

        let ptr = NonNull::new(ptr)
            .ok_or_else(|| SLMError::ProfileError("aslm_profile_create failed".into()))?;

        Ok(Self { ptr })
    }

    /// Raw pointer to hand to `aslm_set_user_profile`.
    pub fn as_ptr(&self) -> *const ffi::aslm_user_profile {
        self.ptr.as_ptr()
    }
}

impl Drop for CUserProfile {
    fn drop(&mut self) {
        unsafe {
            ffi::aslm_profile_free(self.ptr.as_ptr());
        }
    }
}
