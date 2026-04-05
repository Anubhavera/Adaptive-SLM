//! FFI bindings to C++ core

use libc::{c_char, c_int, size_t};

#[repr(C)]
pub struct aslm_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct aslm_user_profile {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct aslm_device_state {
    pub available_ram_bytes: u64,
    pub total_ram_bytes: u64,
    pub cpu_usage: f32,
    pub battery_level: f32,
    pub is_charging: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct aslm_gen_params {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct aslm_init_params {
    pub model_path: *const c_char,
    pub n_threads: i32,
    pub max_context: i32,
    pub use_mmap: bool,
    pub verbose: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum aslm_expertise_level {
    Beginner = 0,
    Intermediate = 1,
    Advanced = 2,
    Expert = 3,
}

#[link(name = "adaptive_slm")]
extern "C" {
    pub fn aslm_default_init_params() -> aslm_init_params;
    pub fn aslm_init(params: *const aslm_init_params) -> *mut aslm_context;
    pub fn aslm_free(ctx: *mut aslm_context);
    
    pub fn aslm_generate(
        ctx: *mut aslm_context,
        prompt: *const c_char,
        params: *const aslm_gen_params,
        output: *mut c_char,
        output_size: size_t,
    ) -> i32;
    
    pub fn aslm_update_device_state(
        ctx: *mut aslm_context,
        state: *const aslm_device_state,
    );
    
    pub fn aslm_get_effective_context_size(ctx: *const aslm_context) -> i32;
    pub fn aslm_get_memory_usage(ctx: *const aslm_context) -> u64;
    pub fn aslm_get_peak_memory_usage(ctx: *const aslm_context) -> u64;
    
    pub fn aslm_profile_create(
        age: i32,
        background: *const c_char,
        interests: *const *const c_char,
        n_interests: i32,
        expertise: aslm_expertise_level,
    ) -> *mut aslm_user_profile;
    
    pub fn aslm_profile_free(profile: *mut aslm_user_profile);
    
    pub fn aslm_set_user_profile(
        ctx: *mut aslm_context,
        profile: *const aslm_user_profile,
    );
}
