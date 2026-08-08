#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ci="$repo_root/.github/workflows/ci.yml"
desktop="$repo_root/.github/workflows/desktop.yml"
ios="$repo_root/.github/workflows/ios.yml"
release_workflow="$repo_root/.github/workflows/release.yml"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

require_text() {
  local file=$1
  local text=$2
  local message=$3
  grep -Fq -- "$text" "$file" || fail "$message"
}

extract_job() {
  local file=$1
  local job=$2
  awk -v key="  $job:" '
    $0 == key { selected = 1 }
    selected && $0 ~ /^  [[:alnum:]_-][[:alnum:]_-]*:$/ && $0 != key { exit }
    selected { print }
  ' "$file"
}

extract_filter() {
  local file=$1
  local filter=$2
  awk -v key="            $filter:" '
    $0 == key { selected = 1 }
    selected && $0 ~ /^            [[:alnum:]_-][[:alnum:]_-]*:$/ && $0 != key { exit }
    selected { print }
  ' "$file"
}

require_text "$ci" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "root CI does not cancel superseded PR runs"
require_text "$ci" 'CARGO_INCREMENTAL: "0"' \
  "root CI enables incremental compilation even though sccache rejects it"
if grep -Fq 'CARGO_INCREMENTAL: "1"' "$release_workflow"; then
  fail "release builds enable incremental compilation while using sccache"
fi
[[ "$(grep -Fc 'CARGO_INCREMENTAL: "0"' "$release_workflow")" -eq 5 ]] \
  || fail "not every sccache-backed release build disables incremental compilation"

# Assert the complete trusted-app predicates, then exercise the security truth
# table. Fragment checks would pass if an operator were accidentally inverted.
python3 - "$ci" "$desktop" "$ios" <<'PY' || exit 1
import re
import sys
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(value.split())


ci_text, desktop_text, ios_text = (Path(path).read_text() for path in sys.argv[1:])
trusted = (
    "github.event_name == 'pull_request' && "
    "github.actor == 'release-plz-mold[bot]' && "
    "startsWith(github.head_ref, 'release-plz-') && "
    "github.event.pull_request.head.repo.full_name == github.repository"
)
match = re.search(r"TRUSTED_RELEASE_PR:\s*\$\{\{\s*(.*?)\s*\}\}", ci_text, re.S)
if match is None or normalize(match.group(1)) != trusted:
    raise SystemExit("FAIL: root trusted release predicate does not match the complete policy")

untrusted = (
    "github.event_name != 'pull_request' || "
    "github.actor != 'release-plz-mold[bot]' || "
    "startsWith(github.head_ref, 'release-plz-') == false || "
    "github.event.pull_request.head.repo.full_name != github.repository"
)
for label, text in (("desktop", desktop_text), ("iOS", ios_text)):
    block = re.search(r"(?ms)^  changes:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n)", text)
    condition = re.search(r"(?ms)^    if: >-\n(.*?)(?=^    [A-Za-z0-9_-]+:)", block.group(1) if block else "")
    if condition is None or normalize(condition.group(1)) != untrusted:
        raise SystemExit(f"FAIL: {label} trusted release predicate does not match the complete policy")


def is_trusted(event: str, actor: str, branch: str, head_repo: str, repo: str) -> bool:
    return (
        event == "pull_request"
        and actor == "release-plz-mold[bot]"
        and branch.startswith("release-plz-")
        and head_repo == repo
    )


cases = [
    (True, "pull_request", "release-plz-mold[bot]", "release-plz-2026", "utensils/mold", "utensils/mold"),
    (False, "push", "release-plz-mold[bot]", "release-plz-2026", "utensils/mold", "utensils/mold"),
    (False, "pull_request", "someone", "release-plz-2026", "utensils/mold", "utensils/mold"),
    (False, "pull_request", "release-plz-mold[bot]", "feat/release-plz-2026", "utensils/mold", "utensils/mold"),
    (False, "pull_request", "release-plz-mold[bot]", "release-plz-2026", "fork/mold", "utensils/mold"),
]
for expected, *inputs in cases:
    if is_trusted(*inputs) is not expected:
        raise SystemExit(f"FAIL: trusted release truth table mismatch for {inputs!r}")

rust = re.search(r"(?ms)^  rust:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n)", ci_text)
if rust is None:
    raise SystemExit("FAIL: protected Rust job is missing")
rust_text = rust.group(1)

run_suite = (
    "needs.changes.outputs.trusted_release_pr != 'true' && "
    "(needs.changes.outputs.rust == 'true' || "
    "(github.event_name == 'push' && needs.changes.outputs.workflow == 'true'))"
)
match = re.search(r"RUN_RUST_SUITE:\s*\$\{\{\s*(.*?)\s*\}\}", rust_text, re.S)
if match is None or normalize(match.group(1)) != run_suite:
    raise SystemExit("FAIL: protected Rust suite selector does not match the complete policy")


def step_condition(name: str) -> str:
    step = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n(.*?)(?=^      - (?:name:|uses:))",
        rust_text,
    )
    condition = re.search(r"(?m)^        if:\s*(.+)$", step.group(1) if step else "")
    if condition is None:
        raise SystemExit(f"FAIL: protected step has no simple condition: {name}")
    return normalize(condition.group(1))


expected_step_conditions = {
    "Require trusted release routing": (
        "needs.changes.outputs.trusted_release_pr == 'true' && "
        "needs.changes.outputs.release != 'true'"
    ),
    "Validate trusted release-plz file scope": (
        "needs.changes.outputs.trusted_release_pr == 'true'"
    ),
    "Validate all locked Cargo graphs": "needs.changes.outputs.release == 'true'",
    "Run protected release contracts": "needs.changes.outputs.release == 'true'",
    "Test MiniMax H3 private-UAT release contract": (
        "needs.changes.outputs.release == 'true'"
    ),
    "Clippy the private MiniMax H3 artifact qualifier": (
        "env.RUN_RUST_SUITE == 'true'"
    ),
    "Test private MiniMax H3 foundations": "env.RUN_RUST_SUITE == 'true'",
}
for name, expected in expected_step_conditions.items():
    actual = step_condition(name)
    if actual != expected:
        raise SystemExit(f"FAIL: protected step condition mismatch for {name}: {actual}")

heavy_start = rust_text.find("      - uses: dtolnay/rust-toolchain@stable")
heavy_end = rust_text.find("      - name: Record protected gate success")
if heavy_start < 0 or heavy_end <= heavy_start:
    raise SystemExit("FAIL: protected Rust heavy-step boundary is missing")
heavy_steps = re.split(
    r"(?m)(?=^      - (?:name:|uses:))",
    rust_text[heavy_start:heavy_end],
)
for step in (candidate for candidate in heavy_steps if candidate.strip()):
    label = step.splitlines()[0].strip()
    condition = re.search(r"(?m)^        if:\s*(.+)$", step)
    if condition is None:
        raise SystemExit(f"FAIL: heavy Rust step is not suite-gated: {label}")
    actual = normalize(condition.group(1))
    if not (
        actual == "env.RUN_RUST_SUITE == 'true'"
        or actual.startswith("env.RUN_RUST_SUITE == 'true' && ")
    ):
        raise SystemExit(f"FAIL: heavy Rust step bypasses suite routing: {label}: {actual}")
PY

release_job="$(extract_job "$ci" rust)"
# The dollar-prefixed names are literal workflow source under test.
# shellcheck disable=SC2016
for contract in \
  'Validate trusted release-plz file scope' \
  'Unexpected generated release path' \
  'git diff --name-status -z "$BASE_SHA...$HEAD_SHA"' \
  '[[ "$status" == M ]]' \
  'cargo metadata --locked --no-deps --format-version 1' \
  '--manifest-path desktop/src-tauri/Cargo.toml' \
  '--manifest-path apps/mobile/src-tauri/Cargo.toml'; do
  grep -Fq -- "$contract" <<< "$release_job" \
    || fail "release gate is missing: $contract"
done

release_change_allowed() {
  local status=$1
  local path=$2
  [[ "$status" == M ]] || return 1
  case "$path" in
    CHANGELOG.md|Cargo.toml|Cargo.lock|desktop/package.json|desktop/src-tauri/Cargo.toml|desktop/src-tauri/Cargo.lock|apps/mobile/src-tauri/Cargo.toml|apps/mobile/src-tauri/Cargo.lock|apps/mobile/src-tauri/tauri.conf.json)
      return 0
      ;;
    crates/*/Cargo.toml)
      [[ "$path" =~ ^crates/[^/]+/Cargo\.toml$ ]]
      ;;
    *)
      return 1
      ;;
  esac
}
release_change_allowed M Cargo.toml \
  || fail "trusted release policy rejects a normal metadata modification"
release_change_allowed M crates/new/Cargo.toml \
  || fail "trusted release policy rejects a normal crate-manifest modification"
for forbidden_status in A D R100 C100; do
  if release_change_allowed "$forbidden_status" crates/new/Cargo.toml; then
    fail "trusted release policy accepts $forbidden_status changes"
  fi
done
if release_change_allowed M src/production.rs; then
  fail "trusted release policy accepts a production-source modification"
fi

require_text "$ci" \
  "cargo clippy -p mold-ai --features cuda,preview,expand,tui,webp,mp4,mdns --all-targets -- -D warnings" \
  "CUDA-gated production code is not linted"
require_text "$ci" \
  "cargo clippy -p mold-ai --features metal,preview,expand,tui,webp,mp4,mdns --all-targets -- -D warnings" \
  "Metal-gated production code is not linted"
require_text "$ci" \
  "nix run nixpkgs#actionlint -- .github/workflows/*.yml" \
  "workflow-only PRs have no protected actionlint proof"
[[ "$(grep -Fc "needs.changes.outputs.gpu == 'true' ||" "$ci")" -eq 2 ]] \
  || fail "CUDA and Metal are not retained for relevant PRs with workflow-only main proof"
for backend_job in cuda-check metal-check; do
  backend_block="$(extract_job "$ci" "$backend_job")"
  grep -Fq "needs.changes.outputs.trusted_release_pr != 'true'" <<< "$backend_block" \
    || fail "$backend_job still runs on generated release PRs"
done

require_text "$ci" \
  "cargo clippy -p mold-ai --features flash-attn -- -D warnings" \
  "the flash-attn binary wiring is only typechecked"
flash_job="$(extract_job "$ci" flash-attn-check)"
grep -Fq "if: github.event_name == 'push'" <<< "$flash_job" \
  || fail "FlashAttention still consumes the PR critical path"

for required_job in rust coverage docs web; do
  required_block="$(extract_job "$ci" "$required_job")"
  grep -Fq 'always() && !cancelled() &&' <<< "$required_block" \
    || fail "required $required_job job does not fail closed without reviving canceled runs"
  grep -Fq "needs.changes.result != 'success'" <<< "$required_block" \
    || fail "required $required_job job does not propagate classifier failures"
  grep -Fq 'Require successful change classification' <<< "$required_block" \
    || fail "required $required_job job has no classifier-failure sentinel"
  grep -Fq 'exit 1' <<< "$required_block" \
    || fail "required $required_job classifier sentinel does not fail"
done

rust_gate="$(extract_job "$ci" rust)"
grep -Fq 'RUN_RUST_SUITE:' <<< "$rust_gate" \
  || fail "protected Rust status does not select its full suite"
grep -Fq 'Require trusted release routing' <<< "$rust_gate" \
  || fail "protected Rust status does not fail closed for trusted release routing"
grep -Fq 'Run protected release contracts' <<< "$rust_gate" \
  || fail "protected Rust status does not own release-contract failures"

grep -Fq 'Check declared MSRV' <<< "$rust_gate" \
  || fail "PR Rust suite no longer checks the declared MSRV"
grep -Fq 'Clippy the private MiniMax H3 artifact qualifier' <<< "$rust_gate" \
  || fail "PR Rust suite no longer checks the private qualifier"
grep -Fq 'Check with all optional features' <<< "$rust_gate" \
  || fail "PR Rust suite no longer checks the optional feature combination"
grep -Fq -- '--skip catalog_api::catalog_live_test::live_search_free_text_can_find_manual_clip_components' <<< "$rust_gate" \
  || fail "PR Rust suite still runs the flaky external catalog monitor"
grep -Fq 'name: Test (full main suite)' <<< "$rust_gate" \
  || fail "main Rust suite does not retain the complete workspace tests"
if grep -Fq -- '--skip batch_transaction::tests::' <<< "$rust_gate"; then
  fail "PR Rust suite skips the high-signal gallery scale regression"
fi
if grep -Fq 'run: cargo check --workspace' "$ci"; then
  fail "root Rust CI still runs cargo check immediately before all-target Clippy"
fi

gpu_filter="$(extract_filter "$ci" gpu)"
grep -Fq 'crates/mold-candle/**' <<< "$gpu_filter" \
  || fail "GPU routing omits backend-conditional mold-candle sources"
for classifier in rust gpu website web nix; do
  classifier_block="$(extract_filter "$ci" "$classifier")"
  if grep -Fq '.github/workflows/ci.yml' <<< "$classifier_block"; then
    fail "workflow-only edits still force the $classifier dynamic build on PRs"
  fi
done
workflow_filter="$(extract_filter "$ci" workflow)"
grep -Fxq "              - '.github/workflows/ci.yml'" <<< "$workflow_filter" \
  || fail "workflow-only changes have no dedicated post-merge classifier"

release_filter="$(extract_filter "$ci" release)"
for manifest in Cargo.toml Cargo.lock desktop/src-tauri/Cargo.toml desktop/src-tauri/Cargo.lock; do
  grep -Fxq "              - '$manifest'" <<< "$release_filter" \
    || fail "release classifier does not run the desktop Candle lock guard for $manifest changes"
done
require_text "$ci" \
  "bash scripts/tests/desktop-candle-nix-source-hash.sh" \
  "release CI does not verify the Nix hash for the locked desktop Candle source"
grep -Fq "'.github/workflows/**'" <<< "$release_filter" \
  || fail "workflow changes do not reach protected actionlint and routing contracts"

nix_filter="$(extract_filter "$ci" nix)"
if grep -Fq 'desktop/**' <<< "$nix_filter"; then
  fail "desktop-only changes still start the mold-web Nix build"
fi
nix_job="$(extract_job "$ci" nix-web)"
grep -Fq "needs.changes.outputs.nix == 'true' ||" <<< "$nix_job" \
  || fail "the sandboxed Nix web build no longer protects relevant PRs"
grep -Fq "github.event_name == 'push' && needs.changes.outputs.workflow == 'true'" <<< "$nix_job" \
  || fail "workflow-only changes do not exercise the Nix build after merge"

desktop_classifier="$(extract_job "$desktop" changes)"
require_text "$desktop" 'name: Classify desktop changes' \
  "desktop workflow has no path classifier"
require_text "$desktop" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.frontend == 'true'" \
  "desktop frontend job is not path-gated on PRs"
desktop_rust="$(extract_job "$desktop" desktop-rust)"
grep -Fq "needs.changes.outputs.rust == 'true' ||" <<< "$desktop_rust" \
  || fail "the macOS native job is not path-gated on PRs"
grep -Fq "needs.changes.outputs.bundle == 'true'" <<< "$desktop_rust" \
  || fail "macOS bundle-sensitive changes do not reach the native job"
grep -Fq 'name: Bundle macOS packaging inputs (debug .app)' <<< "$desktop_rust" \
  || fail "macOS packaging inputs have no focused bundle proof"
grep -Fq "github.event_name == 'pull_request' && needs.changes.outputs.bundle == 'true'" <<< "$desktop_rust" \
  || fail "macOS debug bundle is not restricted to bundle-sensitive PRs"
bundle_filter="$(extract_filter "$desktop" bundle)"
for path in desktop/src-tauri/tauri.macos.conf.json scripts/fix-desktop-macos-linkage.sh; do
  grep -Fq "$path" <<< "$bundle_filter" \
    || fail "desktop bundle classifier omits $path"
done
desktop_linux="$(extract_job "$desktop" desktop-linux)"
grep -Fq "if: github.event_name != 'pull_request'" <<< "$desktop_linux" \
  || fail "Desktop Linux still runs on pull requests"
if grep -Fq 'bunx vue-tsc -b' "$desktop"; then
  fail "desktop frontend still typechecks once explicitly and again during build"
fi
[[ "$(grep -Fc "if: github.event_name != 'pull_request' || needs.changes.outputs.updater == 'true'" "$desktop")" -eq 2 ]] \
  || fail "updater tooling and contract tests are not restricted to updater changes on PRs"
if grep -Fq 'desktop/**' <<< "$desktop_classifier"; then
  fail "desktop frontend classifier is broad enough to include native/mobile files"
fi
[[ "$(grep -Fc 'desktop/src/components/**' "$desktop")" -eq 3 ]] \
  || fail "desktop frontend trigger/classifier does not track desktop components"
[[ "$(grep -Fc 'desktop/src-tauri/**' "$desktop")" -eq 3 ]] \
  || fail "desktop native trigger/classifier does not track the Tauri crate"
if grep -Fq 'desktop/src/mobile/**' "$desktop"; then
  fail "mobile-only changes are not owned exclusively by the iOS workflow"
fi
[[ "$(grep -Fc 'crates/mold-scheduler/**' "$desktop")" -eq 3 ]] \
  || fail "desktop trigger/classifier does not track its scheduler dependency"

require_text "$ios" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "iOS workflow does not cancel superseded PR runs"
require_text "$ios" 'name: Classify iOS changes' \
  "iOS workflow has no path classifier"
ios_frontend_filter="$(extract_filter "$ios" frontend)"
if grep -Fq "'apps/mobile/**'" <<< "$ios_frontend_filter"; then
  fail "native-only iOS changes still run the mobile frontend suite"
fi
grep -Fq "apps/mobile/src-tauri/gen/apple/Assets.xcassets/**" <<< "$ios_frontend_filter" \
  || fail "iOS frontend classifier omits release icon assets"
require_text "$ios" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.rust == 'true'" \
  "iOS simulator Clippy is not retained for native PR changes"
require_text "$ios" 'run: ../scripts/tests/ios-release-assets.sh' \
  "iOS frontend job does not validate the built mobile entry"
require_text "$ios" 'run: bunx vitest run src/mobile' \
  "iOS frontend job does not retain its mobile-specific tests"
if grep -Fq 'run: cargo check --target aarch64-apple-ios-sim' "$ios"; then
  fail "iOS simulator still runs cargo check immediately before Clippy"
fi

echo "PASS: CI routing contract"
