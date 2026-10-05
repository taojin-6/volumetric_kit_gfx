#!/usr/bin/env bash
# Temporary measurement (draft PR, not for merging): where a GPU test's
# start-up time goes, and whether it scales when several processes create
# Vulkan devices at once. Run after the build, from the repository root:
#
#   tools/measure_vk_create.sh [build-dir]
#
# Portable to macOS's bash 3.2 and to the bare Ubuntu containers (no python).
set -uo pipefail

build="${1:-build}"
bench="${build}/tests/vg_vk_create_bench"
TIMEFORMAT='%R'

echo "=== machine: $(uname -sm)"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo' | head -3
fi
# Something else using the GPU skews every number below; record it.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU load before measuring (name, utilization, memory used):"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
fi

echo "=== 1. The Vulkan calls themselves: one process, 20 create/destroy cycles"
"${bench}" 20

echo "=== 2. The same with several processes at once (10 cycles each)"
for k in 1 4 8; do
  "${bench}" --copies "${k}" 10
done

echo "=== 3. The pipelines tests: one process per test against one process for all"
echo "--- ctest, one process per test, one at a time:"
ctest_log="${build}/ctest_one_per_test.log"
ctest --test-dir "${build}" -R '^pipelines\.' -j1 > "${ctest_log}" 2>&1
grep -E 'tests passed|Total Test time' "${ctest_log}"
echo "skipped under ctest: $(grep -cE 'Test +#[0-9]+: .*Skipped' "${ctest_log}")"
echo "--- the test binary, one process, all tests (wall seconds, then gtest's own total):"
log="${build}/tests/pipelines_one_process.log"
( cd "${build}/tests" && time ./vg_pipelines_test > pipelines_one_process.log 2>&1 )
grep -E '^\[==========\] .* ran\.' "${log}"
grep -E '^\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ test' "${log}"
# A GPU test skips itself when it cannot get a device. Show why, and where
# in the sequence the skipping started.
echo "--- skip reasons (count, message):"
grep -A1 -E ': Skipped$' "${log}" | grep -vE ': Skipped$|^--$' | sort | uniq -c | sort -rn | head -5
echo "--- first test to skip, and the two tests before it:"
awk '/^\[ RUN      \]/{prev2=prev1; prev1=cur; cur=$0} /: Skipped$/{print prev2; print prev1; print cur; exit}' "${log}"
echo "--- tests run before the first skip: $(awk '/^\[ RUN      \]/{n++} /: Skipped$/{print n-1; exit}' "${log}")"
