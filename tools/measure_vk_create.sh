#!/usr/bin/env bash
# Temporary measurement (draft PR, not for merging). Earlier runs on this branch
# measured what a Vulkan instance/device create-destroy cycle costs, and found
# vkCreateInstance failing after ~26-31 instances in one process on NVIDIA:
# libnvidia-tls cannot get static TLS once the driver library has been loaded
# and unloaded that many times. This run checks that keeping it loaded fixes
# that, and times the pipelines tests in one process. Run after the build, from the repository root:
#
#   tools/measure_vk_create.sh [build-dir]
#
# Portable to macOS's bash 3.2 and to the bare Ubuntu containers (no python).
set -uo pipefail

build="${1:-build}"
bench="${build}/tests/vg_vk_create_bench"

echo "=== machine: $(uname -sm)"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo' | head -3
fi
# Something else using the GPU skews the numbers; record it.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU load (name, utilization, memory used):"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
fi
echo "--- limits: open files $(ulimit -n), processes $(ulimit -u)"

echo "=== A. 40 create/destroy cycles in one process (failed at cycle 27 last run)"
"${bench}" --leak-check 40

echo "=== A2. 100 cycles with one extra instance kept alive throughout"
"${bench}" --leak-check --anchor 100

echo "=== C. The pipelines tests: one process per test against one process for all"
ctest_log="${build}/ctest_one_per_test.log"
ctest --test-dir "${build}" -R '^pipelines\.' -j1 > "${ctest_log}" 2>&1
echo "--- ctest, one process per test, one at a time:"
grep -E 'tests passed|Total Test time' "${ctest_log}"
echo "skipped under ctest: $(grep -cE 'Test +#[0-9]+: .*Skipped' "${ctest_log}")"
# On NVIDIA, preload libnvidia-tls so it is loaded once, at start-up, and never
# unloaded: the static-TLS exhaustion seen in the last run then cannot happen.
preload=""
for f in /lib/x86_64-linux-gnu/libnvidia-tls.so.* /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.*; do
  [ -e "${f}" ] && { preload="${f}"; break; }
done
echo "--- the test binary, one process, all tests (preloading: ${preload:-nothing}); wall seconds, then gtest's own total:"
log="${build}/tests/pipelines_one_process.log"
TIMEFORMAT='%R'
( cd "${build}/tests" && time env ${preload:+LD_PRELOAD="${preload}"} ./vg_pipelines_test > pipelines_one_process.log 2>&1 )
grep -E '^\[==========\] .* ran\.' "${log}"
grep -E '^\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ test' "${log}"
echo "--- skip reasons (count, message):"
grep -A1 -E ': Skipped$' "${log}" | grep -vE ': Skipped$|^--$' | sort | uniq -c | sort -rn | head -5
