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

    // Link the C++ core.
    // On Linux (shared layout): core/build has libadaptive_slm.so, which itself
    // NEEDs libllama.so etc. — linking the dylib is sufficient at link time.
    // On Windows MinGW (static layout): adaptive_slm.dll / libadaptive_slm.a.
    let core_shared = [
        build_dir.join("libadaptive_slm.so"),
        build_dir.join("libadaptive_slm.dylib"),
        build_dir.join("adaptive_slm.dll"),
    ]
    .iter()
    .any(|p| p.exists());

    if core_shared {
        println!("cargo:rustc-link-lib=dylib=adaptive_slm");
    } else {
        println!("cargo:rustc-link-lib=static=adaptive_slm");
    }

    // llama.cpp + ggml libs are only needed as static archives when the core
    // itself was built statically (Windows/Ninja). In the shared layout they
    // are pulled in at runtime through the core library's NEEDED entries —
    // linking them here would fail (no archives exist) or duplicate code.
    for lib in &["llama", "ggml", "ggml-cpu", "ggml-base"] {
        let found = [
            build_dir.join(format!("lib{}.a", lib)),
            build_dir.join(format!("{}.a", lib)),
            llama_src_dir.join(format!("lib{}.a", lib)),
            llama_src_dir.join(format!("{}.a", lib)),
            llama_ggml_dir.join(format!("lib{}.a", lib)),
            llama_ggml_dir.join(format!("{}.a", lib)),
        ]
        .iter()
        .any(|p| p.exists());
        if found {
            println!("cargo:rustc-link-lib=static={}", lib);
        }
    }

    // Embed an rpath to the C++ build directory so binaries and test
    // executables locate libadaptive_slm.so without LD_LIBRARY_PATH
    // (cargo overwrites LD_LIBRARY_PATH for test binaries, so the [env]
    // entry in .cargo/config.toml is not sufficient on its own).
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
        let bin_dir = build_dir.join("bin");
        if bin_dir.is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bin_dir.display());
        }
    }

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
        // The C++ core is written in C++; the shared library already links
        // libstdc++, but ensure the C++ ABI is present for static layouts.
        println!("cargo:rustc-link-lib=dylib=stdc++");
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
