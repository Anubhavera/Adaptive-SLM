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
    GIT_TAG b5260
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

# Main library (both static and shared for FFI)
add_library(adaptive_slm_static STATIC ${ADAPTIVE_SLM_SOURCES})
add_library(adaptive_slm_shared SHARED ${ADAPTIVE_SLM_SOURCES})

set_target_properties(adaptive_slm_shared PROPERTIES
    OUTPUT_NAME "adaptive_slm"
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    POSITION_INDEPENDENT_CODE ON
)

# Alias for convenience
add_library(adaptive_slm ALIAS adaptive_slm_static)

target_include_directories(adaptive_slm_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_include_directories(adaptive_slm_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(adaptive_slm_static PUBLIC llama ggml)
target_link_libraries(adaptive_slm_shared PUBLIC llama ggml)

# SIMD detection (for our embeddings module — llama.cpp handles its own SIMD)
include(CheckCXXCompilerFlag)
set(SIMD_COMPILE_OPTIONS "")
set(SIMD_DEFINITIONS "")
if(NOT MSVC)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)

    if(COMPILER_SUPPORTS_AVX2)
        set(SIMD_COMPILE_OPTIONS -mavx2 -mfma)
        set(SIMD_DEFINITIONS "ASLM_USE_AVX2=1")
        message(STATUS "AVX2 support enabled for adaptive_slm embeddings")
    elseif(COMPILER_SUPPORTS_NEON)
        set(SIMD_COMPILE_OPTIONS -mfpu=neon)
        set(SIMD_DEFINITIONS "ASLM_USE_NEON=1")
        message(STATUS "NEON support enabled for adaptive_slm embeddings")
    endif()
else()
    check_cxx_compiler_flag("/arch:AVX2" MSVC_SUPPORTS_AVX2)
    if(MSVC_SUPPORTS_AVX2)
        set(SIMD_COMPILE_OPTIONS "/arch:AVX2")
        set(SIMD_DEFINITIONS "ASLM_USE_AVX2=1")
        message(STATUS "AVX2 support enabled (MSVC)")
    endif()
endif()

if(SIMD_COMPILE_OPTIONS)
    target_compile_options(adaptive_slm_static PRIVATE ${SIMD_COMPILE_OPTIONS})
    target_compile_definitions(adaptive_slm_static PUBLIC ${SIMD_DEFINITIONS})
    target_compile_options(adaptive_slm_static INTERFACE ${SIMD_COMPILE_OPTIONS})
    target_compile_definitions(adaptive_slm_static INTERFACE ${SIMD_DEFINITIONS})
    
    target_compile_options(adaptive_slm_shared PRIVATE ${SIMD_COMPILE_OPTIONS})
    target_compile_definitions(adaptive_slm_shared PUBLIC ${SIMD_DEFINITIONS})
    target_compile_options(adaptive_slm_shared INTERFACE ${SIMD_COMPILE_OPTIONS})
    target_compile_definitions(adaptive_slm_shared INTERFACE ${SIMD_DEFINITIONS})
endif()

# Platform libraries for device state monitoring
if(WIN32)
    target_link_libraries(adaptive_slm_static PRIVATE PowrProf)
    target_link_libraries(adaptive_slm_shared PRIVATE PowrProf)
endif()

# Sanitizer option
option(ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)
if(ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(adaptive_slm_static PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(adaptive_slm_static PRIVATE -fsanitize=address,undefined)
    target_compile_options(adaptive_slm_shared PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(adaptive_slm_shared PRIVATE -fsanitize=address,undefined)
endif()

# Benchmark executable (link to shared for FFI testing)
add_executable(adaptive_slm_bench benchmark/bench.cpp)
target_link_libraries(adaptive_slm_bench PRIVATE adaptive_slm_shared)

# Unit tests
enable_testing()

add_executable(test_acc tests/test_acc.cpp)
target_link_libraries(test_acc PRIVATE adaptive_slm_static)
add_test(NAME test_acc COMMAND test_acc)

add_executable(test_embeddings tests/test_embeddings.cpp)
target_link_libraries(test_embeddings PRIVATE adaptive_slm_static)
add_test(NAME test_embeddings COMMAND test_embeddings)

add_executable(test_user_profile tests/test_user_profile.cpp)
target_link_libraries(test_user_profile PRIVATE adaptive_slm_static)
add_test(NAME test_user_profile COMMAND test_user_profile)

add_executable(test_inference tests/test_inference.cpp)
target_link_libraries(test_inference PRIVATE adaptive_slm_static)
add_test(NAME test_inference COMMAND test_inference)

# Install
install(TARGETS adaptive_slm_static adaptive_slm_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${ADAPTIVE_SLM_HEADERS} DESTINATION include/adaptive_slm)
