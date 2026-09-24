# AdaptiveSLM Documentation

**Research-Grade On-Device Small Language Model**

AdaptiveSLM is a novel language model architecture optimized for ultra-low-resource devices (target <512MB RAM). It introduces three key research contributions to achieving high performance with minimal resource usage.

## 📚 Documentation Sections

- [**Architecture Overview**](architecture.md)  
  Deep dive into the MobileLLM-style architecture, C++ core, Rust wrapper, and our three novel contributions (PAKD, ACC, SCPD).

- [**API Reference**](api_reference.md)  
  Complete reference for the C++ Core API and the Rust high-level API.

- [**Training Guide**](training.md)  
  Comprehensive guide to training AdaptiveSLM, including Knowledge Distillation, PAKD, and quantization-aware training.

## 🚀 Novel Contributions

1.  **Profile-Aware Knowledge Distillation (PAKD)**  
    A distillation technique that weights loss based on user profile relevance, creating models specialized for specific user personas.

2.  **Adaptive Context Compression (ACC)**  
    A runtime mechanism that dynamically adjusts the context window size based on real-time device telemetry (RAM, CPU, Battery).

3.  **Semantic Cache with Priority Decay (SCPD)**  
    An intelligent caching system that retains knowledge based on recency, semantic relevance, and profile alignment.

## 🛠️ Quick Start

### Build C++ Core
```bash
cd core
mkdir build && cd build
cmake ..
make -j
```

### Build Rust Wrapper
```bash
cd rust-wrapper
cargo build --release
```

## 📊 Benchmarks

| Metric | AdaptiveSLM Target | Baseline (Qwen 0.5B) |
|--------|-------------------|----------------------|
| RAM | **<512 MB** | ~600 MB |
| MMLU | **>45%** | 43.7% |
| Speed | **>40 t/s** | 35 t/s |
