cmake_minimum_required(VERSION 3.16)
project(adaptive_slm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform-aware optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# llama.cpp options — build only what we need
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG b5233
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

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
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(adaptive_slm PUBLIC llama ggml)

# SIMD detection (for our embeddings module — llama.cpp handles its own SIMD)
include(CheckCXXCompilerFlag)
if(NOT MSVC)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(adaptive_slm PRIVATE -mavx2 -mfma)
        message(STATUS "AVX2 support enabled for adaptive_slm embeddings")
    elseif(COMPILER_SUPPORTS_NEON)
        target_compile_options(adaptive_slm PRIVATE -mfpu=neon)
        message(STATUS "NEON support enabled for adaptive_slm embeddings")
    endif()
else()
    check_cxx_compiler_flag("/arch:AVX2" MSVC_SUPPORTS_AVX2)
    if(MSVC_SUPPORTS_AVX2)
        target_compile_options(adaptive_slm PRIVATE /arch:AVX2)
        message(STATUS "AVX2 support enabled (MSVC)")
    endif()
endif()

# Platform libraries for device state monitoring
if(WIN32)
    target_link_libraries(adaptive_slm PRIVATE PowrProf)
endif()

# Sanitizer option
option(ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)
if(ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(adaptive_slm PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(adaptive_slm PRIVATE -fsanitize=address,undefined)
endif()

# Benchmark executable
add_executable(adaptive_slm_bench benchmark/bench.cpp)
target_link_libraries(adaptive_slm_bench PRIVATE adaptive_slm)

# Install
install(TARGETS adaptive_slm ARCHIVE DESTINATION lib)
install(FILES ${ADAPTIVE_SLM_HEADERS} DESTINATION include/adaptive_slm)
