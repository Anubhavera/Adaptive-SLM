# AdaptiveSLM

**Experimental on-device language model and runtime research**

This project explores a custom SLM and adaptive runtime. Model quality, Android agent behavior, and sustained phone performance still require evaluation. The three experimental components are:

1. **Profile-Aware Knowledge Distillation (PAKD)** - User-adaptive model specialization
2. **Adaptive Context Compression (ACC)** - Runtime context window based on device state
3. **Semantic Cache with Priority Decay (SCPD)** - Intelligent knowledge retention

## Current research and training entry point

Read the [2026-10-01 audit and implementation plan](docs/research/2026-10-01-audit-and-plan.md) before interpreting older results. Start cloud training with [training/mobile_baseline.ipynb](training/mobile_baseline.ipynb) and the [training guide](docs/training.md).

On 2026-10-01, the existing local Qwen2.5 GGUF produced 31.9 tokens/s across three short **desktop CPU** generations, with 592.7 MiB peak process RSS. This is neither a phone measurement nor evidence of the custom student's quality. The old benchmark's weight/state estimate cannot establish a 512 MB total app memory claim. No validated custom-student MMLU result is available.

## Build

### Prerequisites

- CMake 3.16+
- C++17 compiler (GCC 9+ or Clang 10+)
- Rust 1.70+ (for wrapper)

### C++ Core

```bash
cd core
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Rust Wrapper

```bash
cd rust-wrapper
cargo build --release
```

## Usage

### C API

```c
#include "adaptive_slm.h"

// Initialize
aslm_init_params params = aslm_default_init_params();
params.model_path = "models/qwen2.5-0.5b-q4.gguf";
aslm_context* ctx = aslm_init(&params);

// Set user profile
const char* interests[] = {"programming", "science"};
aslm_user_profile* profile = aslm_profile_create(25, "engineer", interests, 2, ASLM_EXPERTISE_ADVANCED);
aslm_set_user_profile(ctx, profile);

// Generate
char output[4096];
aslm_gen_params gen = {.max_tokens = 256, .temperature = 0.7};
aslm_generate(ctx, "What is machine learning?", &gen, output, sizeof(output));

// Cleanup
aslm_profile_free(profile);
aslm_free(ctx);
```

### Rust API

```rust
use adaptive_slm::{AdaptiveSLM, UserProfile, GenerationParams};

#[tokio::main]
async fn main() {
    let mut slm = AdaptiveSLM::new(
        "models/qwen2.5-0.5b-q4.gguf",
        "cache.db",
        Some("your-tavily-api-key"),
    ).await.unwrap();
    
    slm.set_profile(UserProfile::new(25, "engineer", vec!["programming".into()], ExpertiseLevel::Advanced));
    
    let response = slm.generate("What is Rust?", &GenerationParams::default(), true).await.unwrap();
    println!("{}", response);
}
```

## Running Benchmarks

```bash
# Full benchmark suite
cd core/build
./adaptive_slm_bench models/qwen2.5-0.5b-q4.gguf

# Compare with baselines
python benchmarks/compare.py --compare
```

## License

MIT

## 📚 Full Documentation

Comprehensive documentation is available in the `docs/` directory:

- [**Home**](docs/index.md)
- [**Architecture**](docs/architecture.md)
- [**API Reference**](docs/api_reference.md)
- [**Training Guide**](docs/training.md)
