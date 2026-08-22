#!/usr/bin/env bash
# Run the gates `main` has to pass, locally and hermetically.
#
# PR CI is path-gated and now runs every deterministic check for the changed
# surfaces before merge. This script provides the same suites locally, plus an
# opt-in GPU/Nix loop for machine-dependent checks.
#
# The other half of the job is the environment. This machine's own
# `~/.config/mold/config.toml` and its direnv `MOLD_*` exports leak into
# `cargo test` and fail a dozen tests that are green on a clean runner, which
# makes a local run useless as a signal. Every step therefore runs against a
# throwaway `HOME`/`XDG_CONFIG_HOME` with the `MOLD_*` variables stripped,
# while `CARGO_HOME`, `RUSTUP_HOME` and the target directory stay put so
# nothing is rebuilt or re-downloaded for the sake of isolation.
#
#   scripts/ci-local.sh                 # rust + web + docs + contracts
#   scripts/ci-local.sh rust            # one suite
#   scripts/ci-local.sh -k              # run everything, summarise failures
#   scripts/ci-local.sh --list          # show the steps without running them
#   scripts/ci-local.sh gpu nix         # the machine-dependent extras
set -uo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 1

keep_going=0
list_only=0
dirty=0
# A step that hangs must end the run with a named TIMEOUT rather than sitting
# there: `cargo test --workspace` currently blocks forever on this repo's
# mold-ai-server route tests on some hosts, and an unbounded wait looks
# identical to slow progress.
step_timeout=${CI_LOCAL_STEP_TIMEOUT:-2400}
suites=()

usage() {
  # Everything from the second line up to the first line that is not a comment,
  # so the header cannot drift out of the help text.
  sed -n '2,${/^[^#]/q;p;}' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Suites:
  rust       fmt, generated profiles, clippy, the full test suite, feature gates
  web        knip/architecture, Studio + web tests, prettier, the SPA build
  docs       website prettier, reference verification, VitePress build
  contracts  the CI routing, release and MiniMax H3 contract scripts
  gpu        CUDA (Linux) or Metal (macOS) forced-local clippy — needs the toolchain
  nix        `nix build .#mold-web`, the sandboxed web bundle CI builds

Options:
  -k, --keep-going  run every step even after a failure, then summarise
      --list        print the steps that would run
      --dirty       keep the ambient environment (skips the hermetic HOME)
      --timeout N   per-step limit in seconds (default 2400, 0 disables;
                    a step that exceeds it is reported as TIME, not left to hang)
  -h, --help        this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -k|--keep-going) keep_going=1 ;;
    --list) list_only=1 ;;
    --dirty) dirty=1 ;;
    --timeout) shift; step_timeout="${1:-0}" ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) suites+=("$1") ;;
  esac
  shift
done
if [ ${#suites[@]} -eq 0 ]; then
  suites=(rust web docs contracts)
fi

# A misspelled suite must not run nothing and then report success — that is the
# exact false green this script exists to prevent.
known_suites=" rust web docs contracts gpu nix all "
for suite in "${suites[@]}"; do
  case "$known_suites" in
    *" $suite "*) ;;
    *) echo "unknown suite: $suite" >&2; usage >&2; exit 2 ;;
  esac
done

wants() {
  local suite="$1"
  local candidate
  for candidate in "${suites[@]}"; do
    [ "$candidate" = "$suite" ] && return 0
    [ "$candidate" = "all" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Hermetic environment
# ---------------------------------------------------------------------------
sandbox_home=""
cleanup() {
  [ -n "$sandbox_home" ] && rm -rf "$sandbox_home"
}
trap cleanup EXIT

if [ "$dirty" -eq 0 ]; then
  # Anchor the caches that live under the real HOME *before* moving it, so
  # isolation costs nothing in rebuild or download time.
  export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
  export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
  export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
  sandbox_home=$(mktemp -d)
  export HOME="$sandbox_home"
  export XDG_CONFIG_HOME="$sandbox_home/.config"
  export XDG_DATA_HOME="$sandbox_home/.local/share"
  export XDG_CACHE_HOME="$sandbox_home/.cache"
  mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
  # `MOLD_*` shapes engine behaviour and config resolution; a runner has none.
  while IFS='=' read -r name _; do
    case "$name" in MOLD_*) unset "$name" ;; esac
  done < <(env)
fi

# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
declare -a results=()
failed=0
skipped=0

# Run one command under the step timeout, killing everything it started.
#
# `timeout --foreground` is not enough: it signals the command it launched and
# explicitly leaves that command's children alone, so a hung `cargo test` would
# be reported as a timeout while its test binary kept running, holding ports
# and CPU for every later step. Enabling job control around the launch puts the
# command in its own process group whose id is `$!`, so the watcher can signal
# the whole tree — `setsid` is not usable here because it re-forks when it is
# already a group leader, which leaves `$!` pointing at the wrong group (and
# macOS does not ship it at all).
run_bounded() {
  if [ "$step_timeout" -le 0 ]; then
    "$@"
    return $?
  fi
  local pid watcher status monitor=0
  case "$-" in *m*) monitor=1 ;; esac
  set -m
  "$@" &
  pid=$!
  [ "$monitor" -eq 1 ] || set +m
  (
    sleep "$step_timeout"
    kill -TERM -"$pid" 2>/dev/null
    sleep 10
    kill -KILL -"$pid" 2>/dev/null
  ) &
  watcher=$!
  wait "$pid"
  status=$?
  kill "$watcher" 2>/dev/null
  wait "$watcher" 2>/dev/null
  # 128+SIGTERM: the watcher fired, which is a timeout rather than a failure.
  [ "$status" -eq 143 ] && status=124
  return "$status"
}

step() {
  local name="$1"
  shift
  if [ "$list_only" -eq 1 ]; then
    printf '  %-42s %s\n' "$name" "$*"
    return 0
  fi
  if [ "$failed" -gt 0 ] && [ "$keep_going" -eq 0 ]; then
    results+=("SKIP  $name")
    skipped=$((skipped + 1))
    return 0
  fi
  printf '\n\033[1m==> %s\033[0m\n' "$name"
  local started elapsed status
  started=$(date +%s)
  run_bounded "$@"
  status=$?
  elapsed=$(( $(date +%s) - started ))
  case "$status" in
    0) results+=("$(printf 'PASS  %-42s %4ds' "$name" "$elapsed")") ;;
    124)
      results+=("$(printf 'TIME  %-42s %4ds (exceeded %ss)' "$name" "$elapsed" "$step_timeout")")
      failed=$((failed + 1))
      printf '\033[31m--- %s timed out after %ss\033[0m\n' "$name" "$step_timeout"
      ;;
    *)
      results+=("$(printf 'FAIL  %-42s %4ds' "$name" "$elapsed")")
      failed=$((failed + 1))
      printf '\033[31m--- %s failed\033[0m\n' "$name"
      ;;
  esac
}

skip() {
  local name="$1" reason="$2"
  if [ "$list_only" -eq 1 ]; then
    printf '  %-42s (conditional: %s)\n' "$name" "$reason"
    return 0
  fi
  results+=("$(printf 'SKIP  %-42s %s' "$name" "$reason")")
  skipped=$((skipped + 1))
}

# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
msrv_toolchain() {
  local msrv
  msrv=$(sed -n 's/^rust-version = "\([^"]*\)"$/\1/p' Cargo.toml)
  [ -n "$msrv" ] || return 1
  case "$msrv" in
    *.*.*) printf '%s' "$msrv" ;;
    *.*) printf '%s.0' "$msrv" ;;
    *) return 1 ;;
  esac
}

if wants rust; then
  # The MSRV gate is a different compiler, not a different command: an API
  # newer than the declared minimum passes every step below on the ambient
  # toolchain and fails the required job.
  msrv=$(msrv_toolchain || true)
  if [ -z "${msrv:-}" ]; then
    # An unreadable `rust-version` is a failure, not a skip: the gate exists and
    # this run cannot honour it.
    step "rust: MSRV (unreadable rust-version in Cargo.toml)" false
  else
    if ! rustup toolchain list 2>/dev/null | grep -q "^$msrv"; then
      # CI installs it rather than assuming it, and so does this: skipping the
      # gate would let a post-MSRV API pass here and fail the required job.
      step "rust: install MSRV $msrv" \
        rustup toolchain install "$msrv" --profile minimal
    fi
    step "rust: MSRV $msrv check" \
      cargo "+$msrv" check --workspace --all-targets --locked
    step "rust: MSRV $msrv feature check" \
      cargo "+$msrv" check -p mold-ai --locked \
      --features preview,discord,expand,tui,metrics,webp,mp4,mdns,pulid
  fi
  step "rust: fmt" cargo fmt --all -- --check
  step "rust: generated generation profiles" \
    cargo run -p mold-ai-core --bin generate_generation_profiles -- --check
  step "rust: clippy" cargo clippy --workspace --all-targets -- -D warnings
  step "rust: test (full main suite)" cargo test --workspace
  step "rust: optional feature check" \
    cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4,mdns,pulid
  step "rust: MiniMax H3 private foundations" \
    cargo test -p mold-ai-inference --lib --features h3-private-uat minimax_h3
  for bin in h3_artifact_qualification h3_qwen_layer50_capture \
             h3_transformer_capture; do
    step "rust: clippy $bin" \
      cargo clippy -p mold-ai-inference --features dev-bins,h3-private-uat --bin "$bin" -- -D warnings
  done
fi

if wants contracts; then
  step "contracts: CI routing policy" bash scripts/tests/ci-routing-contract.sh
  # All three graphs, not just the root: the desktop and mobile crates are
  # excluded from the workspace and have their own lockfiles.
  step "contracts: locked Cargo graphs" bash -c '
    set -e
    cargo metadata --locked --no-deps --format-version 1 >/dev/null
    cargo metadata --locked --no-deps --format-version 1 \
      --manifest-path desktop/src-tauri/Cargo.toml >/dev/null
    cargo metadata --locked --no-deps --format-version 1 \
      --manifest-path apps/mobile/src-tauri/Cargo.toml >/dev/null'
  # The protected release contracts main runs when release-classified files
  # change. They are cheap, so run them every time rather than reimplementing
  # the workflow's path classification here and getting it subtly wrong.
  for contract in release-sync-pr crates-publish-contract ci-coverage-disk-guard \
                  docker-web-context desktop-candle-lock-sync \
                  desktop-candle-nix-source-hash desktop-dmg-packaging \
                  cuda-distribution-contract \
                  install-cuda-arch cuda-qualification-contract \
                  minimax-h3-attention-release-contract bench-qwen-parse \
                  regression-matrix-aggregate-failures regression-matrix-concurrency \
                  regression-matrix-family-sizing regression-matrix-source-image \
                  regression-matrix-transient-retry wan-regression-matrix; do
    script="scripts/tests/${contract}.sh"
    if [ -f "$script" ]; then
      step "contracts: ${contract}" bash "$script"
    fi
  done
  step "contracts: CUDA PTX parser" python3 scripts/tests/cuda-ptx-parser-contract.py
  step "contracts: local multi-GPU qualification" \
    bash scripts/tests/local-multi-gpu-qualification-contract.sh
  step "contracts: H3 private-UAT release" \
    bash scripts/tests/minimax-h3-private-uat-release-contract.sh
  for contract in conformance gpu-conformance capture-producer \
                  qwen-layer50-capture visual-vae-capture audio-vae-capture \
                  transformer-capture; do
    script="scripts/tests/minimax-h3-${contract}-contract.py"
    if [ -f "$script" ]; then
      step "contracts: H3 ${contract}" python3 "$script"
    fi
  done
  # CI reaches actionlint through nix; use whichever is available.
  if command -v actionlint >/dev/null 2>&1; then
    step "contracts: workflow syntax" bash -c 'actionlint .github/workflows/*.yml'
  elif command -v nix >/dev/null 2>&1; then
    step "contracts: workflow syntax" \
      bash -c 'nix run nixpkgs#actionlint -- .github/workflows/*.yml'
  else
    skip "contracts: workflow syntax" "neither actionlint nor nix is installed"
  fi
fi

if wants web; then
  step "web: install" bun install --frozen-lockfile
  step "web: frontend architecture" bun run check:architecture
  step "web: dead code (knip)" bun run check:dead-code
  # Two prettier scopes: the root script only checks `studio/`, and CI runs the
  # web workspace's own from `web/`.
  step "web: prettier (studio)" bun run fmt:check
  step "web: prettier (web)" bash -c 'cd web && exec bun run fmt:check'

  step "web: studio tests" bun run test:studio
  step "web: web tests" bash -c 'cd web && exec bun run test'

  step "web: SPA build (vue-tsc)" bash -c 'cd web && exec bun run build'

  step "web: desktop tests" bash -c 'cd desktop && exec bun run test'

fi

if wants docs; then
  step "docs: install" bash -c 'cd website && exec bun install'

  step "docs: prettier" bash -c 'cd website && exec bun run fmt:check'

  step "docs: reference verification" bash -c 'cd website && exec bun run verify'

  step "docs: build" bash -c 'cd website && exec bun run build'

fi

if wants gpu; then
  case "$(uname -s)" in
    Darwin)
      step "gpu: Metal forced-local clippy" \
        cargo clippy -p mold-ai --features metal,preview,expand,tui,webp,mp4,mdns,pulid --all-targets -- -D warnings
      ;;
    *)
      if command -v nvcc >/dev/null 2>&1; then
        step "gpu: CUDA forced-local clippy" \
          cargo clippy -p mold-ai --features h3-cuda,preview,expand,tui,webp,mp4,mdns,pulid --all-targets -- -D warnings
        step "gpu: clippy h3_runtime_qualification_record" \
          cargo clippy -p mold-ai-inference --features dev-bins,h3-cuda \
          --bin h3_runtime_qualification_record -- -D warnings
        step "gpu: CUDA private H3 server bridge" \
          cargo clippy -p mold-ai-server --features h3-private-uat --all-targets -- -D warnings
        # The capture adapters compile only under `cuda`, so the CPU rust suite
        # above never sees them.
        for bin in h3_qwen_layer50_capture h3_visual_vae_capture \
                   h3_audio_vae_capture h3_transformer_capture; do
          step "gpu: CUDA clippy $bin" \
            cargo clippy -p mold-ai-inference --features dev-bins,h3-private-uat,cuda --bin "$bin" -- -D warnings
        done
      else
        skip "gpu: CUDA forced-local clippy" "nvcc not on PATH"
      fi
      ;;
  esac
fi

if wants nix; then
  if command -v nix >/dev/null 2>&1; then
    step "nix: mold-web sandbox build" nix build .#mold-web --print-build-logs
  else
    skip "nix: mold-web sandbox build" "nix not installed"
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [ "$list_only" -eq 1 ]; then
  exit 0
fi

printf '\n\033[1m==> summary\033[0m\n'
for line in "${results[@]}"; do
  case "$line" in
    FAIL*) printf '\033[31m%s\033[0m\n' "$line" ;;
    SKIP*) printf '\033[33m%s\033[0m\n' "$line" ;;
    *) printf '%s\n' "$line" ;;
  esac
done

if [ "$failed" -gt 0 ]; then
  printf '\n\033[31m%d step(s) failed\033[0m%s\n' "$failed" "${skipped:+, $skipped skipped}"
  exit 1
fi
printf '\n\033[32mall steps passed\033[0m%s\n' "${skipped:+ ($skipped skipped)}"
