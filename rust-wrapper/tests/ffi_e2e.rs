//! End-to-end FFI test: Rust wrapper → C++ core (libadaptive_slm.so) → llama.cpp.
//!
//! Runs against a real GGUF model from the repo's `models/` directory:
//! the AdaptiveSLM 30L family model when present, else the
//! Qwen2.5-0.5B-Instruct Q4_K_M baseline.

use std::path::{Path, PathBuf};

use adaptive_slm::family::{DepthChoice, ModelFamily};
use adaptive_slm::profile::{Expertise, UserProfile};
use adaptive_slm::{
    AdaptiveSLM, CUserProfile, DeviceState, EngineConfig, GenerationParams,
};

const MODELS_DIR: &str = "/home/anubhav/Anubhav/Adaptive-SLM/models";
const ADAPTIVE_30L: &str = "/home/anubhav/Anubhav/Adaptive-SLM/models/adaptive_slm-30l-q4_k_m.gguf";
const QWEN25: &str = "/home/anubhav/Anubhav/Adaptive-SLM/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn e2e_model() -> PathBuf {
    if Path::new(ADAPTIVE_30L).exists() {
        println!("[e2e] model: adaptive_slm-30l (project family model, untrained weights)");
        PathBuf::from(ADAPTIVE_30L)
    } else {
        assert!(Path::new(QWEN25).exists(), "no test model found in {MODELS_DIR}");
        println!("[e2e] model: qwen2.5-0.5b-instruct-q4_k_m (baseline)");
        PathBuf::from(QWEN25)
    }
}

fn temp_db_path() -> PathBuf {
    let path = std::env::temp_dir().join(format!("aslm_ffi_e2e_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

#[test]
fn model_family_matches_repo_layout() {
    let family = ModelFamily::new(MODELS_DIR);
    let depths = family.available_depths();
    println!("[family] available depths in {MODELS_DIR}: {depths:?}");

    let beginner = family
        .pick(Expertise::Beginner)
        .expect("models dir must contain at least one .gguf");
    println!("[family] Beginner → {}", beginner.display());
    assert_eq!(beginner.extension().unwrap(), "gguf");

    if !depths.is_empty() {
        // Beginner falls back to the smallest available depth (next-larger rule).
        assert_eq!(
            beginner.file_name().unwrap().to_str().unwrap(),
            format!("adaptive_slm-{}-q4_k_m.gguf", depths[0].tag())
        );
        assert!(depths.contains(&DepthChoice::D30));
        let expert = family.pick(Expertise::Expert).unwrap();
        assert_eq!(
            expert.file_name().unwrap().to_str().unwrap(),
            "adaptive_slm-30l-q4_k_m.gguf"
        );
        println!("[family] Expert → {}", expert.display());
    }
}

#[tokio::test]
async fn ffi_end_to_end() {
    // 1. Safe-wrapper init with explicit engine params.
    let model = e2e_model();
    let engine = EngineConfig {
        n_threads: 4,
        max_context: 512,
        use_mmap: true,
        verbose: false,
    };
    let db_path = temp_db_path();
    let mut slm = AdaptiveSLM::with_engine_config(
        model.to_str().unwrap(),
        db_path.to_str().unwrap(),
        None,
        engine,
    )
    .await
    .expect("engine init failed");
    println!(
        "[e2e] init OK (effective context: {} tokens)",
        slm.context_size()
    );
    assert!(slm.context_size() > 0);

    let params = GenerationParams {
        max_tokens: 32,
        temperature: 0.7,
        ..Default::default()
    };

    // 2. Blocking generate with the ChatML prompt.
    let prompt = "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n";
    let (text, tokens) = slm
        .generate_with_token_count(prompt, &params, false)
        .await
        .expect("generate failed");
    println!("[e2e] generate: {tokens} tokens, output: {text:?}");
    assert!(tokens > 0, "expected a positive token count, got {tokens}");
    assert!(!text.trim().is_empty(), "generated text must be non-empty");

    // 3. Streaming via the C token callback.
    let mut chunks: Vec<String> = Vec::new();
    let stream_tokens = slm
        .generate_streaming("Count slowly from one to five.", &params, |token| {
            chunks.push(token.to_string());
        })
        .expect("streaming generate failed");
    let streamed: String = chunks.concat();
    println!(
        "[e2e] stream: {stream_tokens} tokens across {} chunks, output: {streamed:?}",
        chunks.len()
    );
    assert!(stream_tokens > 0, "expected streamed tokens, got {stream_tokens}");
    assert!(!chunks.is_empty(), "callback should have fired");
    assert!(!streamed.trim().is_empty(), "streamed text must be non-empty");

    // 4. Memory reporting.
    let mem = slm.memory_usage();
    let peak = slm.peak_memory_usage();
    println!(
        "[e2e] memory: {mem} bytes ({:.1} MB), peak: {peak} bytes ({:.1} MB)",
        mem as f64 / (1024.0 * 1024.0),
        peak as f64 / (1024.0 * 1024.0)
    );
    assert!(mem > 0, "aslm_get_memory_usage must report > 0");
    assert!(peak >= mem, "peak must be >= current usage");

    // 5. ACC: high-RAM state first, then a starved state; context must shrink.
    let initial = slm.context_size();
    let high = DeviceState {
        available_ram_bytes: 15 * 1024 * 1024 * 1024,
        total_ram_bytes: 16 * 1024 * 1024 * 1024,
        cpu_usage: 0.05,
        battery_level: 1.0,
        is_charging: true,
    };
    slm.update_device_state(&high);
    let ctx_high = slm.context_size();
    let low = DeviceState {
        available_ram_bytes: 64 * 1024 * 1024,
        total_ram_bytes: 16 * 1024 * 1024 * 1024,
        cpu_usage: 0.9,
        battery_level: 0.1,
        is_charging: false,
    };
    slm.update_device_state(&low);
    let ctx_low = slm.context_size();
    println!(
        "[e2e] ACC: initial={initial}, high-RAM={ctx_high}, low-RAM(64MB avail)={ctx_low}"
    );
    assert!(
        ctx_low < ctx_high || ctx_low < initial,
        "ACC must shrink the effective context under low RAM (low={ctx_low}, high={ctx_high})"
    );

    // 6. Profile: create in the C++ core, set on the context, generate, free.
    let profile = UserProfile::new(
        30,
        "software engineering",
        vec!["systems programming".into(), "large language models".into()],
        Expertise::Advanced,
    );
    let c_profile = CUserProfile::new(&profile).expect("aslm_profile_create failed");
    slm.set_engine_profile(c_profile);
    let (profiled_text, profiled_tokens) = slm
        .generate_with_token_count("Explain what a mutex is, in one sentence.", &params, false)
        .await
        .expect("generate with profile failed");
    println!(
        "[e2e] profile generate: {profiled_tokens} tokens, output: {profiled_text:?}"
    );
    assert!(profiled_tokens > 0, "profile generate must produce tokens");
    slm.clear_engine_profile();

    // Profile is freed; a further generation proves the engine no longer
    // dereferences the released pointer.
    let (post_text, post_tokens) = slm
        .generate_with_token_count("Name one programming language.", &params, false)
        .await
        .expect("generate after profile free failed");
    println!(
        "[e2e] post-profile-free generate: {post_tokens} tokens, output: {post_text:?}"
    );
    assert!(post_tokens > 0);

    // 7. Drop (aslm_free) runs implicitly at scope end; no crash = pass.
    let _ = std::fs::remove_file(&db_path);
}

/// Same FFI chain, but against the trained Qwen2.5 baseline so generated
/// text quality can be verified by eye.
#[tokio::test]
async fn qwen_baseline_end_to_end() {
    assert!(Path::new(QWEN25).exists(), "qwen2.5 baseline model missing");

    let engine = EngineConfig {
        n_threads: 4,
        max_context: 512,
        use_mmap: true,
        verbose: false,
    };
    let db_path = std::env::temp_dir().join(format!("aslm_ffi_e2e_qwen_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&db_path);

    let mut slm = AdaptiveSLM::with_engine_config(QWEN25, db_path.to_str().unwrap(), None, engine)
        .await
        .expect("qwen2.5 engine init failed");
    println!("[e2e:qwen] init OK (effective context: {} tokens)", slm.context_size());

    let params = GenerationParams {
        max_tokens: 32,
        temperature: 0.7,
        ..Default::default()
    };
    let prompt = "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n";
    let (text, tokens) = slm
        .generate_with_token_count(prompt, &params, false)
        .await
        .expect("qwen2.5 generate failed");
    println!("[e2e:qwen] generate: {tokens} tokens, output: {text:?}");
    assert!(tokens > 0, "qwen2.5 must generate tokens, got {tokens}");
    assert!(!text.trim().is_empty());

    let mut chunks: Vec<String> = Vec::new();
    let stream_tokens = slm
        .generate_streaming("Give me one tip for writing good code.", &params, |token| {
            chunks.push(token.to_string());
        })
        .expect("qwen2.5 streaming generate failed");
    let streamed: String = chunks.concat();
    println!(
        "[e2e:qwen] stream: {stream_tokens} tokens across {} chunks, output: {streamed:?}",
        chunks.len()
    );
    assert!(stream_tokens > 0);
    assert!(!streamed.trim().is_empty());

    let _ = std::fs::remove_file(&db_path);
}
