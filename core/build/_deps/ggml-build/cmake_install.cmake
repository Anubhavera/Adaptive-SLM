# Install script for directory: /home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build/src/libggml.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-cpu.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-alloc.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-backend.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-blas.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-cann.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-cpp.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-cuda.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-opt.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-metal.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-rpc.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-virtgpu.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-sycl.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-vulkan.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-webgpu.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/ggml-zendnn.h"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build/src/libggml-base.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build/ggml-config.cmake"
    "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build/ggml-version.cmake"
    )
endif()

