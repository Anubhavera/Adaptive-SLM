use std::env;
use std::path::PathBuf;

fn main() {
    // Locate the C++ build directory.
    // Default: ../core/build (relative to rust-wrapper/).
    // Override with ASLM_BUILD_DIR env var for CI or custom layouts.
    let build_dir = env::var("ASLM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../core/build")
        });

    let build_dir = build_dir.canonicalize().unwrap_or_else(|_| {
        panic!(
            "C++ build directory not found at {:?}.\n\
             Build the C++ core first:\n\
             cd core/build && cmake --build . --parallel\n\
             Or set ASLM_BUILD_DIR to the correct path.",
            build_dir
        )
    });

    // Search paths for libraries
    let llama_ggml_dir = build_dir.join("_deps/llama_cpp-build/ggml/src");
    let llama_src_dir = build_dir.join("_deps/llama_cpp-build/src");

    // MinGW/Ninja builds ggml sub-libs as `foo.a` not `libfoo.a`.
    // Create libfoo.a copies so `-lfoo` resolves correctly.
    for lib in &["ggml", "ggml-cpu", "ggml-base"] {
        let src = llama_ggml_dir.join(format!("{}.a", lib));
        let dst = llama_ggml_dir.join(format!("lib{}.a", lib));
        if src.exists() && !dst.exists() {
            std::fs::copy(&src, &dst)
                .unwrap_or_else(|e| panic!("failed to copy {:?} → {:?}: {}", src, dst, e));
        }
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}", llama_ggml_dir.display());
    println!("cargo:rustc-link-search=native={}", llama_src_dir.display());

    // Link our shared library first (for FFI), then its dependencies
    // On Linux: libadaptive_slm.so, on Windows: adaptive_slm.dll, on macOS: libadaptive_slm.dylib
    println!("cargo:rustc-link-lib=adaptive_slm");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Platform-specific system libraries
    #[cfg(target_os = "windows")]
    {
        // llama.cpp C++ objects need GCC C++ stdlib, exception runtime, and OpenMP.
        // Rust doesn't link these automatically when using static C++ archives.
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gcc_eh");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=PowrProf");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=ntdll");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // Re-run if the build dir contents change
    println!("cargo:rerun-if-env-changed=ASLM_BUILD_DIR");
    println!("cargo:rerun-if-changed=../core/build/libadaptive_slm.so");
    println!("cargo:rerun-if-changed=../core/src/inference.cpp");
}
