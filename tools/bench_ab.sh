#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
#
# Frame times of a base commit against this tree: builds example_03_model at
# <base> from the dependencies ./build already fetched, then runs `--bench` on
# the two binaries alternately, pair by pair, so drift on a shared machine
# falls on both sides. Prints the median of each side's per-run medians -- GPU
# frame time and wall frame time -- as a Markdown table, appended to
# $GITHUB_STEP_SUMMARY in CI. A base without `--bench` is reported, and only
# this tree is timed.
#
# Usage: tools/bench_ab.sh <base-commit> [build-dir]
#   BENCH_RUNS    pairs per grid (default 5)
#   BENCH_FRAMES  frames per run (default 500)
#   BENCH_GRIDS   cube-grid sizes to time (default "32 64")
#   BENCH_SIZE    render size (default "1920 1080")
# Run from the repository root, with <build-dir> a Release build.
set -euo pipefail

base=$1
build=${2:-build}
runs=${BENCH_RUNS:-5}
frames=${BENCH_FRAMES:-500}
grids=${BENCH_GRIDS:-32 64}
read -r width height <<<"${BENCH_SIZE:-1920 1080}"

head_bin="$build/examples/example_03_model"
work=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/bench_ab.XXXXXX")
trap 'rm -rf "$work"' EXIT

# The base's sources; a shallow checkout fetches the commit first.
if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  git fetch --no-tags --depth=1 origin "$base"
fi
mkdir -p "$work/src"
git archive "$base" | tar -x -C "$work/src"

flags=()
for src in "$build"/_deps/*-src; do
  [ -d "$src" ] || continue
  dep=$(basename "$src" -src | tr '[:lower:]' '[:upper:]')
  flags+=("-DFETCHCONTENT_SOURCE_DIR_${dep}=$(cd "$src" && pwd)")
done
# ${flags[@]+...}: an empty array is unbound under `set -u` in bash 3.2.
cmake -S "$work/src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DVG_BUILD_TESTS=OFF ${flags[@]+"${flags[@]}"} >"$work/configure.log" 2>&1
cmake --build "$work/build" --target example_03_model --parallel \
  >"$work/build.log" 2>&1
base_bin="$work/build/examples/example_03_model"
if ! "$base_bin" --bench 1 --grid 1 >/dev/null 2>&1; then
  echo "base ${base} has no --bench; timing this tree only"
  base_bin=""
fi

device=$("$head_bin" --bench 1 --grid 1 | sed -n 's/^03_model bench: //p' |
  head -n 1)

# One run: "<gpu frame median> <wall frame median>", in milliseconds.
run() {
  "$1" --bench "$frames" --grid "$2" --width "$width" --height "$height" |
    awk '/gpu frame/ { g = $3 } /wall frame/ { w = $3 } END { print g, w }'
}

results="$work/results.txt"
: >"$results"
for grid in $grids; do
  for ((i = 0; i < runs; ++i)); do
    if [ -n "$base_bin" ]; then
      echo "base $grid $(run "$base_bin" "$grid")" >>"$results"
    fi
    echo "head $grid $(run "$head_bin" "$grid")" >>"$results"
  done
done

python3 - "$results" "$base" "$device" "$runs" "$frames" "$width" "$height" \
  <<'PY' | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
import statistics
import sys

path, base, device, runs, frames, width, height = sys.argv[1:]
rows = [line.split() for line in open(path)]


def median(side, grid, column):
    values = [float(r[column]) for r in rows if r[0] == side and r[1] == grid]
    return statistics.median(values) if values else None


def cell(value):
    return "--" if value is None else f"{value:.3f}"


def change(before, after):
    if before is None or after is None or before == 0:
        return "--"
    return f"{(after - before) / before * 100:+.1f}%"


print(f"### Frame times: `{base[:12]}` (base) against this tree (head)\n")
print(f"{device}; {width}x{height}, {frames} frames a run, {runs} "
      "alternating run(s) a side; medians of the per-run medians, in ms.\n")
print("| draws | GPU base | GPU head | GPU change | wall base | wall head "
      "| wall change |")
print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for grid in dict.fromkeys(r[1] for r in rows):
    g = [median(s, grid, 2) for s in ("base", "head")]
    w = [median(s, grid, 3) for s in ("base", "head")]
    print(f"| {int(grid) ** 2} | {cell(g[0])} | {cell(g[1])} | "
          f"{change(*g)} | {cell(w[0])} | {cell(w[1])} | {change(*w)} |")
PY
