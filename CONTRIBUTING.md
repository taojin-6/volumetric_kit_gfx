# Contributing

[AGENTS.md](AGENTS.md) is the canonical shared guide for contributors, Codex,
and Claude Code. It owns the working rules, validation commands, and task map;
`CLAUDE.md` imports it. [DECISIONS.md](DECISIONS.md) keeps the recorded choices.

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
gfx_root="$(git rev-parse --show-toplevel)"
cmake -S "$gfx_root" -B "$gfx_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$gfx_root/build" --parallel
ctest --test-dir "$gfx_root/build" --output-on-failure
```

GPU-backed tests skip automatically when no Vulkan device is present, so the
suite runs on headless machines.

## Formatting and CI

- The `lint` check (`pre-commit run --all-files`, then volumetric_kit_core's
  `check_fork_guards.py` on `.github/workflows`) and the per-platform builds are
  **required** to merge: a red `lint` blocks the PR regardless of local setup, so
  CI — not the local hook — is the real guarantee.
- [pre-commit.ci](https://pre-commit.ci) runs the same hooks on every PR and
  pushes the auto-fixes back to the branch, so formatting still lands if you skip
  the local `pre-commit install`.
- clang-format is pinned (see `.pre-commit-config.yaml`) so local and CI
  formatting are byte-identical; don't reformat with a different version.

## Pull requests from forks

A pull request from a fork runs none of the `ci` workflow's jobs, and
`ci / required` fails it; to test one, a maintainer pushes its branch to this
repository and opens a pull request from there. The workflows skip the jobs
with `if:` guards, which a fork's own pull request can edit, so they are not
the boundary: the first gate is the repository's approval setting for fork pull
requests (Settings → Actions → General, *Require approval for all external
contributors*), and the backstop is the hook each self-hosted runner runs
before a job, which refuses a fork's job on the host.

## Commits

Follow the existing Conventional Commits style in the history, e.g.
`feat(core): …`, `build: …`, `refactor(build): …`.
