# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Exercise each real executable before any window/device initialization. This
# catches an example retaining its old parser or ignoring a failed config load.
function(check_invocation expected_result expected_text)
  execute_process(
    COMMAND "${EXAMPLE_EXE}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 10)
  if(NOT result STREQUAL "${expected_result}")
    message(FATAL_ERROR "${ARGN}: expected ${expected_result}, got ${result}\n"
                        "${output}${error}")
  endif()
  if(NOT "${output}${error}" MATCHES "${expected_text}")
    message(FATAL_ERROR "${ARGN}: missing ${expected_text}\n${output}${error}")
  endif()
  if("${output}${error}" MATCHES "glfwInit|glfwCreateWindow|rendered|MoltenVK")
    message(FATAL_ERROR "${ARGN}: configuration must finish before startup\n"
                        "${output}${error}")
  endif()
endfunction()

check_invocation(0 "--config" --help)
check_invocation(2 "--frames" --frames 1)
check_invocation(2 "--config" --config)
check_invocation(2 "does-not-exist.json" --config
                 "${CMAKE_CURRENT_BINARY_DIR}/does-not-exist.json")
file(WRITE "${INVALID_CONFIG}" "{ invalid JSON")
check_invocation(2 "JSON|json|parse" --config "${INVALID_CONFIG}")
