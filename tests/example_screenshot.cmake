# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# CI runs this where Vulkan is required. An unrelated working directory checks
# that the output path is anchored to the config, and the PPM dimensions check
# that the validated window settings actually reach the headless renderer.
file(MAKE_DIRECTORY "${WORK_DIR}")
file(REMOVE "${EXPECTED_SCREENSHOT}")
execute_process(
  COMMAND "${EXAMPLE_EXE}" --config "${SCREENSHOT_CONFIG}"
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 60)
if(NOT result STREQUAL "0" OR "${output}${error}" MATCHES "VUID-")
  message(
    FATAL_ERROR "headless screenshot failed: ${result}\n${output}${error}")
endif()
if(NOT EXISTS "${EXPECTED_SCREENSHOT}")
  message(FATAL_ERROR "screenshot was not written relative to its config\n"
                      "${output}${error}")
endif()
file(READ "${EXPECTED_SCREENSHOT}" header LIMIT 13)
file(SIZE "${EXPECTED_SCREENSHOT}" size)
if(NOT header STREQUAL "P6\n96 64\n255\n" OR NOT size EQUAL 18445)
  message(FATAL_ERROR "screenshot did not use configured 96 x 64 output: "
                      "header=${header}, bytes=${size}")
endif()
