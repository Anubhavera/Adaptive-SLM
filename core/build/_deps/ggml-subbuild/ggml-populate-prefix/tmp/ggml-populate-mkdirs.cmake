# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-src"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-build"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/tmp"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/src"
  "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/anubhav-hooda/Anubhav/SLM/adaptive-slm/core/build/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
