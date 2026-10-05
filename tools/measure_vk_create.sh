#!/usr/bin/env bash
# Temporary measurement (draft PR, not for merging). Earlier runs on this branch
# measured what a Vulkan instance/device create-destroy cycle costs; this one
# looks for why vkCreateInstance starts failing after ~31 GPU tests run in one
# process on the NVIDIA boxes. Run after the build, from the repository root:
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

echo "=== A. 100 create/destroy cycles in one process, no layers"
"${bench}" --leak-check 100

echo "=== B. The same with the validation layer"
"${bench}" --leak-check --layers 100

echo "=== C. The pipelines tests in one process, with the loader's errors"
log="${build}/tests/pipelines_one_process.log"
(
  cd "${build}/tests" || exit 1
  VK_LOADER_DEBUG=error,warn ./vg_pipelines_test > pipelines_one_process.log 2>&1 &
  pid=$!
  # Sample the test process's open file descriptors while it runs (Linux).
  max_fds=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [ -d "/proc/${pid}/fd" ]; then
      n=$(ls "/proc/${pid}/fd" 2>/dev/null | wc -l)
      [ "${n}" -gt "${max_fds}" ] && max_fds=${n}
    fi
    sleep 0.2
  done
  wait "${pid}"
  echo "most open file descriptors seen: ${max_fds}"
)
grep -E '^\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ test' "${log}"
echo "--- skip reasons (count, message):"
grep -A1 -E ': Skipped$' "${log}" | grep -vE ': Skipped$|^--$' | sort | uniq -c | sort -rn | head -5
echo "--- tests run before the first skip: $(awk '/^\[ RUN      \]/{n++} /: Skipped$/{print n-1; exit}' "${log}")"
echo "--- loader errors and warnings (count, message), up to the first skip:"
awk '/: Skipped$/{exit} {print}' "${log}" | grep -E 'ERROR|WARNING' | sed -E 's/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -10
echo "--- loader output at the first failed vkCreateInstance:"
awk '/^\[ RUN      \]/{buf=""} {buf=buf "\n" $0} /: Skipped$/{print buf; exit}' "${log}" | grep -E 'ERROR|WARNING|RUN|Skipped|instance' | tail -15
