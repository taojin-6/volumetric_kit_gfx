#!/usr/bin/env bash
# Stop, deregister, and delete this machine's self-hosted runners for THIS repo
# (REPO below), leaving any runners it hosts for OTHER repos untouched.
# Use when decommissioning or migrating a runner host (e.g. switching Macs, or
# rebuilding the Linux box). Works on macOS and Linux. Run as the user that owns
# the runner dirs.  Best-effort: keeps going if a single step fails.
set -o pipefail

REPO="taojin-6/volumetric_kit_gfx"
# Linux runs the runner as a systemd service (root); macOS as a per-user LaunchAgent.
if [ "$(uname -s)" = "Linux" ]; then SUDO=(sudo); else SUDO=(); fi

# Runners live in ~/ci-runners/<repo>/runner-<i>. Hosts set up before that layout
# still have them flat in ~/actions-runner-<i>, so sweep both: a host that was
# never migrated has to tear down cleanly rather than silently find nothing and
# leave its runners registered.
shopt -s nullglob
dirs=("$HOME"/ci-runners/"${REPO#*/}"/runner-*/ "$HOME"/actions-runner-*/)
if [ "${#dirs[@]}" -eq 0 ]; then
  echo "No runner dirs on this machine (looked in ~/ci-runners/${REPO#*/}/ and"
  echo "~/actions-runner-*) — nothing to remove."
  exit 0
fi

for dir in "${dirs[@]}"; do
  [ -f "${dir}config.sh" ] || continue
  # Only touch runners registered to THIS repo. One machine can host runners for
  # several repos — each dir's .runner records its gitHubUrl — so without this
  # guard the legacy ~/actions-runner-* glob, which matches every repo's flat
  # dirs, would stop and delete the others' runners too.
  # The trailing quote pins the match to the full repo name (no prefix collision).
  if ! grep -qsF "github.com/${REPO}\"" "${dir}.runner"; then
    echo "==> Skipping ${dir} — not registered to ${REPO}"
    continue
  fi
  echo "==> Removing runner in ${dir}"
  (
    cd "$dir" || exit 1
    "${SUDO[@]}" ./svc.sh stop      2>/dev/null || true
    "${SUDO[@]}" ./svc.sh uninstall 2>/dev/null || true
    # Deregister from GitHub so it doesn't linger as an offline runner.
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
      RM_TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/remove-token" --jq .token)"
      ./config.sh remove --token "$RM_TOKEN" \
        || echo "   config.sh remove failed — delete it under Settings -> Actions -> Runners"
    else
      echo "   gh not authed — service stopped, but the runner is still registered."
      echo "   Remove it under Settings -> Actions -> Runners (or run 'gh auth login' and re-run)."
    fi
  )
  rm -rf "$dir"
done

echo "Done. Runners still registered on the repo:"
if command -v gh >/dev/null 2>&1; then
  gh api "repos/${REPO}/actions/runners" --jq '.runners[].name' || true
fi
