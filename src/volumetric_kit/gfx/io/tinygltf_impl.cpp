// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Single translation unit that instantiates tinygltf's implementation (and the
// stb_image decoder it drives). Mirrors vma_impl.cpp: the header-only library
// is compiled exactly once here so every other TU includes <tiny_gltf.h> as a
// declarations-only header.
//
// Only the _IMPLEMENTATION switches are local to this TU. The feature macros
// that change the header's *declarations* (TINYGLTF_NO_STB_IMAGE_WRITE, which
// drops the default write callback this read-only tier never uses) are set
// target-wide in CMake so every io TU sees the same TinyGLTF definition --
// a per-file define here would mismatch the gltf_loader.cpp TU and break the
// one-definition rule (the missing WriteImageData symbol).

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include <tiny_gltf.h>
