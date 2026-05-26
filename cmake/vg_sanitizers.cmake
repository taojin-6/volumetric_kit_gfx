# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Sanitizer flags, applied GLOBALLY (compile + link) when VG_SANITIZE is set.
#
# Unlike vg_warnings (per-target PRIVATE), sanitizers are applied globally on
# purpose: the flags must reach the link line, and ASan's ABI is contagious —
# mixing a sanitized library with an un-sanitized executable is broken.
# Including this before add_subdirectory(third_party) also instruments the
# vendored static deps, which otherwise raise alloc/dealloc-mismatch false
# positives when linked into a sanitized test binary. Off by default, so normal
# builds are unaffected.
#
# Example: cmake -B build -DCMAKE_BUILD_TYPE=Debug
# -DVG_SANITIZE="address;undefined"
if(VG_SANITIZE)
  if(MSVC)
    message(
      FATAL_ERROR
        "VG_SANITIZE is not supported with MSVC; use a Clang or GCC build.")
  endif()

  # address;undefined -> address,undefined (the -fsanitize= argument form).
  list(JOIN VG_SANITIZE "," _vg_sanitize_list)
  set(_vg_sanitize_flags
      -fsanitize=${_vg_sanitize_list}
      -fno-omit-frame-pointer # readable sanitizer stack traces
      -fno-sanitize-recover=all) # a finding fails the run, matching -Werror's
                                 # posture

  add_compile_options(${_vg_sanitize_flags})
  add_link_options(${_vg_sanitize_flags})

  message(STATUS "Sanitizers enabled (VG_SANITIZE): ${VG_SANITIZE}")
endif()
