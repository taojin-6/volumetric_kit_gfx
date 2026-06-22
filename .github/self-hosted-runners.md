# Self-hosted CI runners (Linux GPU box)

The three Ubuntu legs of `ci.yml` run on a self-hosted runner labelled
`vk-linux-gpu`, each in an OS-matched container (`ubuntu:22.04` / `:24.04` /
`:26.04`) with the host GPU passed through via `--gpus all`. macOS, `lint`, and
`sanitizers` stay on GitHub-hosted runners (the sanitizers job deliberately runs
on lavapipe — ASan/LSan against the proprietary NVIDIA driver report
driver-internal allocations as false leaks).

A self-hosted runner takes **one job at a time**, so to run the legs in parallel
you register **several runner instances** on the box, all sharing the
`vk-linux-gpu` label; GitHub then dispatches the queued legs across them. The
build matrix emits 6 Linux jobs (3 OSes × Debug/Release), so 6 instances give
full parallelism — fewer just means some legs queue.

## Host prerequisites (one-time)

1. **Docker Engine** — <https://docs.docker.com/engine/install/ubuntu/>. The apt
   repo method is identical on 22.04 / 24.04 / 26.04 (`$VERSION_CODENAME`
   resolves to `jammy` / `noble` / `resolute`, all published).
2. **NVIDIA Container Toolkit**, so containers can use the GPU:
   ```bash
   # add the toolkit apt repo first (see NVIDIA docs), then:
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   # verify the GPU is visible inside a container:
   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ubuntu:24.04 nvidia-smi
   ```

## Register N parallel runners

The registration token and the exact tarball URL come from the repo →
**Settings → Actions → Runners → New self-hosted runner (Linux x64)**. One token
registers all N (valid ~1h).

```bash
TOKEN="<REGISTRATION_TOKEN>"
RUNNER_VERSION="2.330.0"          # whatever the runner page shows
URL="https://github.com/taojin-6/volumetric_kit_gfx"
N=6                                # one per Linux build leg (3 OS x Debug/Release)
THREADS=4                          # N*THREADS ~= core count -> no oversubscription

curl -o ~/actions-runner.tar.gz -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

for i in $(seq 1 "$N"); do
  dir=~/actions-runner-$i
  mkdir -p "$dir" && tar xzf ~/actions-runner.tar.gz -C "$dir"
  ( cd "$dir"
    # Loaded into every job on this runner -> caps cmake/ctest fan-out so the
    # parallel legs share the cores instead of each grabbing all of them.
    printf 'CMAKE_BUILD_PARALLEL_LEVEL=%s\nCTEST_PARALLEL_LEVEL=%s\n' "$THREADS" "$THREADS" > .env
    ./config.sh --unattended --url "$URL" --token "$TOKEN" \
      --labels vk-linux-gpu --name "$(hostname)-$i" --work _work   # do NOT sudo config.sh
    sudo ./svc.sh install "$USER"   # one systemd service per runner name
    sudo ./svc.sh start )
done
```

Each runner becomes its own `actions.runner.*` systemd service that auto-starts
on boot. **Until at least one `vk-linux-gpu` runner is online the Ubuntu legs
stay _pending_** and the `ci / required` gate waits on them — bring the runners
up before merging.

> Membership in the `docker` group is root-equivalent; fine for a personal box,
> reconsider for a shared one.
