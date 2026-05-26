# Contributing

## One-time setup

Install [`pre-commit`](https://pre-commit.com) (`pipx install pre-commit`, or
`pip install --user pre-commit`), then, from a fresh clone, install the git hook:

```sh
pre-commit install
```

This wires the hooks in [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
(clang-format, cmake-format, trailing-whitespace, end-of-file-fixer) into
`.git/hooks/`. They run on staged files at `git commit` and abort the commit if
anything is reformatted, so style is fixed locally instead of in CI. The hook
lives in `.git/` and is **not** tracked, so each clone must run this once.

## Build and test

Out-of-source CMake build:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

GPU-backed tests skip automatically when no Vulkan device is present, so the
suite runs on headless machines.

## Formatting and CI

- The `lint` check (`pre-commit run --all-files`) and the per-platform builds are
  **required** to merge: a red `lint` blocks the PR regardless of local setup, so
  CI — not the local hook — is the real guarantee.
- [pre-commit.ci](https://pre-commit.ci) runs the same hooks on every PR and
  pushes the auto-fixes back to the branch, so formatting still lands if you skip
  the local `pre-commit install`.
- clang-format is pinned (see `.pre-commit-config.yaml`) so local and CI
  formatting are byte-identical; don't reformat with a different version.

## Commits

Follow the existing Conventional Commits style in the history, e.g.
`feat(core): …`, `build: …`, `refactor(build): …`.
