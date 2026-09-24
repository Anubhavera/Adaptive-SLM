//! AdaptiveSLM Rust Wrapper
//! 
//! Provides memory-safe Rust bindings to the C++ core,
//! plus the SCPD cache and Tavily search integration.

pub mod cache;
pub mod search;
pub mod profile;
mod ffi;

use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use thiserror::Error;

pub use cache::{KnowledgeCache, KnowledgeEntry, CacheConfig};
pub use search::{TavilyClient, SearchResult};
pub use profile::UserProfile;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum SLMError {
    #[error("Failed to initialize model: {0}")]
    InitError(String),
    
    #[error("Generation failed: {0}")]
    GenerationError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Search error: {0}")]
    SearchError(String),
    
    #[error("Profile error: {0}")]
    ProfileError(String),
}

pub type Result<T> = std::result::Result<T, SLMError>;

// ============================================================================
// Device State
// ============================================================================

/// Device state for ACC (Adaptive Context Compression)
#[derive(Debug, Clone, Default)]
pub struct DeviceState {
    pub available_ram_bytes: u64,
    pub total_ram_bytes: u64,
    pub cpu_usage: f32,
    pub battery_level: f32,
    pub is_charging: bool,
}

impl DeviceState {
    /// Get current device state (Linux implementation)
    pub fn current() -> Self {
        let mut state = Self::default();
        
        // Read /proc/meminfo
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        state.total_ram_bytes = kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        state.available_ram_bytes = kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
        }
        
        // Battery (laptop)
        if let Ok(capacity) = std::fs::read_to_string("/sys/class/power_supply/BAT0/capacity") {
            state.battery_level = capacity.trim().parse::<f32>().unwrap_or(-1.0) / 100.0;
        }
        
        if let Ok(status) = std::fs::read_to_string("/sys/class/power_supply/BAT0/status") {
            state.is_charging = status.trim() == "Charging" || status.trim() == "Full";
        }
        
        state
    }
}

// ============================================================================
// Generation Parameters
// ============================================================================

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

// ============================================================================
// Main SLM Context
// ============================================================================

/// AdaptiveSLM context with integrated caching and search
pub struct AdaptiveSLM {
    ctx: NonNull<ffi::aslm_context>,
    cache: KnowledgeCache,
    search: Option<TavilyClient>,
    profile: Option<UserProfile>,
}

// Safety: The C context is thread-safe
unsafe impl Send for AdaptiveSLM {}

impl AdaptiveSLM {
    /// Initialize the SLM with optional Tavily API key
    pub async fn new(
        model_path: &str,
        db_path: &str,
        tavily_api_key: Option<&str>,
    ) -> Result<Self> {
        let model_path_c = CString::new(model_path)
            .map_err(|_| SLMError::InitError("Invalid model path".into()))?;
        
        let mut params = unsafe { ffi::aslm_default_init_params() };
        params.model_path = model_path_c.as_ptr();
        params.n_threads = 0; // Auto-detect
        params.max_context = 2048;
        params.use_mmap = true;
        params.verbose = false;
        
        let ctx = unsafe { ffi::aslm_init(&params) };
        let ctx = NonNull::new(ctx)
            .ok_or_else(|| SLMError::InitError("Failed to initialize C context".into()))?;
        
        let cache = KnowledgeCache::new(db_path)
            .await
            .map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        let search = tavily_api_key.map(|key| TavilyClient::new(key));
        
        Ok(Self {
            ctx,
            cache,
            search,
            profile: None,
        })
    }
    
    /// Set user profile for personalization
    pub fn set_profile(&mut self, profile: UserProfile) {
        self.profile = Some(profile);
    }
    
    /// Update device state for ACC
    pub fn update_device_state(&self, state: &DeviceState) {
        let c_state = ffi::aslm_device_state {
            available_ram_bytes: state.available_ram_bytes,
            total_ram_bytes: state.total_ram_bytes,
            cpu_usage: state.cpu_usage,
            battery_level: state.battery_level,
            is_charging: state.is_charging,
        };
        
        unsafe {
            ffi::aslm_update_device_state(self.ctx.as_ptr(), &c_state);
        }
    }
    
    /// Generate response with caching and optional web search
    pub async fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        use_search: bool,
    ) -> Result<String> {
        // 1. Check cache first
        if let Some(cached) = self.cache.search(prompt, 0.8).await? {
            tracing::info!("Cache hit for prompt");
            return Ok(cached.content);
        }
        
        // 2. Try web search if enabled and connected
        let mut context = String::new();
        if use_search {
            if let Some(ref search) = self.search {
                match search.search(prompt).await {
                    Ok(results) => {
                        context = results.iter()
                            .take(3)
                            .map(|r| format!("{}: {}", r.title, r.snippet))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        
                        tracing::info!("Got {} search results", results.len());
                    }
                    Err(e) => {
                        tracing::warn!("Search failed: {}", e);
                    }
                }
            }
        }
        
        // 3. Build augmented prompt
        let augmented_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("Context:\n{}\n\nQuestion: {}", context, prompt)
        };
        
        // 4. Generate with C++ core
        let prompt_c = CString::new(augmented_prompt.as_str())
            .map_err(|_| SLMError::GenerationError("Invalid prompt".into()))?;
        
        let c_params = ffi::aslm_gen_params {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
        };
        
        let mut output = vec![0u8; 8192];
        let tokens = unsafe {
            ffi::aslm_generate(
                self.ctx.as_ptr(),
                prompt_c.as_ptr(),
                &c_params,
                output.as_mut_ptr() as *mut i8,
                output.len(),
            )
        };
        
        if tokens < 0 {
            return Err(SLMError::GenerationError("Generation failed".into()));
        }
        
        let response = unsafe {
            CStr::from_ptr(output.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned()
        };
        
        // 5. Cache the result with SCPD priority
        let priority = self.compute_priority(prompt);
        self.cache.store(prompt, &response, priority).await?;
        
        Ok(response)
    }
    
    /// Compute SCPD priority based on profile relevance
    fn compute_priority(&self, topic: &str) -> f32 {
        match &self.profile {
            Some(profile) => profile.compute_relevance(topic),
            None => 0.5, // Default priority
        }
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> u64 {
        unsafe { ffi::aslm_get_memory_usage(self.ctx.as_ptr()) }
    }
    
    /// Get effective context size (after ACC adjustment)
    pub fn context_size(&self) -> i32 {
        unsafe { ffi::aslm_get_effective_context_size(self.ctx.as_ptr()) }
    }
}

impl Drop for AdaptiveSLM {
    fn drop(&mut self) {
        unsafe {
            ffi::aslm_free(self.ctx.as_ptr());
        }
    }
}
