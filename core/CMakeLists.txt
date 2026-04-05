cmake_minimum_required(VERSION 3.16)
project(adaptive_slm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags for low-resource devices
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")

# GGML options
option(GGML_STATIC "Build GGML as static library" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Include FetchContent for GGML
include(FetchContent)

# Fetch GGML (the tensor library llama.cpp uses)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG master
)
FetchContent_MakeAvailable(ggml)

# Our library sources
set(ADAPTIVE_SLM_SOURCES
    src/inference.cpp
    src/context_adapt.cpp
    src/embeddings.cpp
    src/user_profile.cpp
)

set(ADAPTIVE_SLM_HEADERS
    include/adaptive_slm.h
    include/context_adapt.h
    include/embeddings.h
    include/user_profile.h
)

# Main library
add_library(adaptive_slm STATIC ${ADAPTIVE_SLM_SOURCES})

target_include_directories(adaptive_slm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ggml_SOURCE_DIR}/include
)

target_link_libraries(adaptive_slm PUBLIC ggml)

# Enable SIMD if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)

if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(adaptive_slm PRIVATE -mavx2 -mfma)
    message(STATUS "AVX2 support enabled")
elseif(COMPILER_SUPPORTS_NEON)
    target_compile_options(adaptive_slm PRIVATE -mfpu=neon)
    message(STATUS "NEON support enabled")
endif()

# Benchmark executable
add_executable(adaptive_slm_bench benchmark/bench.cpp)
target_link_libraries(adaptive_slm_bench PRIVATE adaptive_slm)

# Install
install(TARGETS adaptive_slm ARCHIVE DESTINATION lib)
install(FILES ${ADAPTIVE_SLM_HEADERS} DESTINATION include/adaptive_slm)
