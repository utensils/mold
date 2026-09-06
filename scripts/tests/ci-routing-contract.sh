#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ci="$repo_root/.github/workflows/ci.yml"
desktop="$repo_root/.github/workflows/desktop.yml"
ios="$repo_root/.github/workflows/ios.yml"
android="$repo_root/.github/workflows/android.yml"
android_gradle="$repo_root/apps/mobile/src-tauri/gen/android/build.gradle.kts"
release_workflow="$repo_root/.github/workflows/release.yml"
testflight="$repo_root/.github/workflows/testflight-ios.yml"

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

rust_job=$(extract_job "$ci" rust)
windows_rust_job=$(extract_job "$ci" windows-rust)
cuda_job=$(extract_job "$ci" cuda-check)
# flash_job=$(extract_job "$ci" flash-attn-check)
# grep -Fq 'needs.changes.outputs.h3_cuda_server' <<< "$flash_job" \
#   || fail "H3 CUDA server tests are not path-gated on PRs"
# grep -Fq 'bash scripts/test-h3-cuda-server.sh' <<< "$flash_job" \
#   || fail "CUDA toolkit job does not run the full hermetic H3 server suite"
producer_command='cargo clippy -p mold-ai-inference --features dev-bins,h3-cuda --bin h3_runtime_qualification_record -- -D warnings'
if grep -Fq -- "$producer_command" <<< "$rust_job"; then
  fail "the CUDA-backed H3 runtime-record producer runs in the CPU Rust job"
fi
[[ -z "$cuda_job" ]] \
  || fail "routine CI still contains the redundant 40-minute CUDA forced-local job"
[[ -n "$windows_rust_job" ]] \
  || fail "PR-visible Windows Rust typecheck is missing"
grep -Fq 'cargo check --locked -p mold-ai-server' <<< "$windows_rust_job" \
  || fail "Windows Rust gate does not compile the server routes used by the nightly CLI"
grep -Fq 'cargo build --release -p mold-ai --features cuda' "$release_workflow" \
  || fail "release CI no longer compiles the shipped CUDA artifacts"

require_text "$ci" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "root CI does not cancel superseded PR runs"
require_text "$ci" 'CARGO_INCREMENTAL: "0"' \
  "root CI enables incremental compilation even though sccache rejects it"

for workflow in "$android" "$release_workflow" "$testflight"; do
  require_text "$workflow" \
    "if ! cargo tauri --version 2>/dev/null | grep -Fq '2.11.4'; then" \
    "Tauri CLI install does not tolerate a binary restored by Rust cache: $workflow"
  require_text "$workflow" \
    "cargo install tauri-cli --version 2.11.4 --locked --force" \
    "Tauri CLI install cannot replace a stale restored binary: $workflow"
done

if grep -Fq 'CARGO_INCREMENTAL: "1"' "$release_workflow"; then
  fail "release builds enable incremental compilation while using sccache"
fi
[[ "$(grep -Fc 'CARGO_INCREMENTAL: "0"' "$release_workflow")" -eq 5 ]] \
  || fail "not every sccache-backed release build disables incremental compilation"

# Assert the complete trusted-app predicates, then exercise the security truth
# table. Fragment checks would pass if an operator were accidentally inverted.
python3 - "$ci" "$desktop" "$ios" "$android" <<'PY' || exit 1
import re
import sys
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(value.split())


ci_text, desktop_text, ios_text, android_text = (Path(path).read_text() for path in sys.argv[1:])
# A late-fragment repair changes only the promoted changelog, but release-plz
# still requires Apple delivery at that exact main SHA before tagging it.
for label, text in (("desktop", desktop_text), ("iOS", ios_text)):
    push = re.search(r"(?ms)^  push:\n(.*?)(?=^  [A-Za-z0-9_-]+:)", text)
    if push is None or '"CHANGELOG.md"' not in push.group(1):
        raise SystemExit(f"FAIL: {label} must run for changelog-only release repair pushes")

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
for label, text in (("desktop", desktop_text), ("iOS", ios_text), ("Android", android_text)):
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
    "needs.changes.outputs.workflow == 'true')"
)
match = re.search(r"RUN_RUST_SUITE:\s*\$\{\{\s*(.*?)\s*\}\}", rust_text, re.S)
if match is None or normalize(match.group(1)) != run_suite:
    raise SystemExit("FAIL: protected Rust suite selector does not match the complete policy")

run_format = (
    "needs.changes.outputs.trusted_release_pr != 'true' && "
    "needs.changes.outputs.rust_format == 'true'"
)
match = re.search(r"RUN_RUST_FORMAT:\s*\$\{\{\s*(.*?)\s*\}\}", rust_text, re.S)
if match is None or normalize(match.group(1)) != run_format:
    raise SystemExit("FAIL: protected Rust formatter selector does not match the complete policy")


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
    "Validate workflow syntax": (
        "github.event_name == 'pull_request' && "
        "needs.changes.outputs.workflow == 'true'"
    ),
    "Validate CI routing policy": (
        "github.event_name == 'pull_request' && "
        "needs.changes.outputs.workflow == 'true'"
    ),
    "Run protected release contracts": (
        "needs.changes.outputs.release == 'true'"
    ),
    "Test MiniMax H3 private-UAT release contract": (
        "needs.changes.outputs.release == 'true'"
    ),
    "Clippy the private MiniMax H3 artifact qualifier": (
        "env.RUN_RUST_SUITE == 'true'"
    ),
    "Test private MiniMax H3 foundations": "env.RUN_RUST_SUITE == 'true'",
}
expected_step_conditions["Validate all locked Cargo graphs"] = (
    "needs.changes.outputs.release == 'true'"
)
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
    if label == "- uses: dtolnay/rust-toolchain@stable":
        allowed = actual in {
            "github.event_name == 'pull_request' && env.RUN_RUST_FORMAT == 'true'",
            "env.RUN_RUST_SUITE == 'true'",
        }
    elif label == "- name: Format":
        allowed = actual == "env.RUN_RUST_FORMAT == 'true'"
    else:
        allowed = (
            actual == "env.RUN_RUST_SUITE == 'true'"
            or actual.startswith("env.RUN_RUST_SUITE == 'true' && ")
        )
    if not allowed:
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
  if [[ "$status" == D ]]; then
    case "$path" in
      changelog.d/README.md) return 1 ;;
      changelog.d/*.md) return 0 ;;
    esac
  fi
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
release_change_allowed D changelog.d/my-note.md \
  || fail "trusted release policy rejects a consumed changelog fragment deletion"
if release_change_allowed D changelog.d/README.md; then
  fail "trusted release policy accepts deleting changelog.d/README.md"
fi
if release_change_allowed A changelog.d/my-note.md; then
  fail "trusted release policy accepts a fragment addition on the release PR"
fi
if release_change_allowed D Cargo.toml; then
  fail "trusted release policy accepts a non-fragment deletion"
fi
if release_change_allowed M src/production.rs; then
  fail "trusted release policy accepts a production-source modification"
fi

require_text "$ci" \
  "cargo clippy -p mold-ai --features h3,metal,preview,expand,tui,webp,mp4,mdns,pulid --all-targets -- -D warnings" \
  "Metal-gated production code is not linted"
require_text "$ci" \
  "cargo check -p mold-ai-server --features h3,metal,expand,mdns,metrics,mp4,pulid,webp" \
  "the reviewed H3 Metal server recipe is not compiled"
require_text "$ci" \
  "nix run nixpkgs#actionlint -- .github/workflows/*.yml" \
  "main release validation has no protected actionlint proof"
metal_block="$(extract_job "$ci" metal-check)"
grep -Fq "github.event_name == 'pull_request'" <<< "$metal_block" \
  || fail "the fast Metal compile does not protect relevant pull requests"
grep -Fq "needs.changes.outputs.trusted_release_pr != 'true'" <<< "$metal_block" \
  || fail "the Metal compile runs on generated release PRs"

require_text "$ci" \
  "cargo clippy -p mold-ai --features flash-attn -- -D warnings" \
  "the flash-attn binary wiring is only typechecked"
# flash_job="$(extract_job "$ci" flash-attn-check)"
# grep -Fq "if: github.event_name == 'push'" <<< "$flash_job" \
#   || fail "FlashAttention still consumes the PR critical path"

for required_job in rust docs web; do
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

for spec in \
  "docs|Verify docs references" \
  "docs|Build docs" \
  "web|Check frontend architecture and dead code" \
  "web|Run shared Studio tests" \
  "web|Run web tests" \
  "web|Build web SPA (includes vue-tsc typecheck)"; do
  job=${spec%%|*}
  step=${spec#*|}
  block="$(extract_job "$ci" "$job")"
  if grep -A1 -F -- "- name: $step" <<< "$block" | grep -Fq "if: github.event_name == 'push'"; then
    fail "$job substantive step does not run on pull requests: $step"
  fi
done

coverage_gate="$(extract_job "$ci" coverage)"
grep -Fq "github.event_name == 'push'" <<< "$coverage_gate" \
  || fail "Codecov coverage still recompiles the workspace on pull requests"
grep -Fq 'uses: codecov/codecov-action@v5' <<< "$coverage_gate" \
  || fail "main no longer uploads coverage to Codecov"

rust_gate="$(extract_job "$ci" rust)"
grep -Fq 'RUN_RUST_SUITE:' <<< "$rust_gate" \
  || fail "protected Rust status does not select its full suite"
grep -Fq 'Require trusted release routing' <<< "$rust_gate" \
  || fail "protected Rust status does not fail closed for trusted release routing"
grep -Fq 'Run protected release contracts' <<< "$rust_gate" \
  || fail "protected Rust status does not own release-contract failures"

grep -Fq 'RUN_RUST_FORMAT:' <<< "$rust_gate" \
  || fail "protected Rust status does not select PR formatting"
grep -Fq "if: env.RUN_RUST_FORMAT == 'true'" <<< "$rust_gate" \
  || fail "PR Rust formatting is not isolated from the main suite"
grep -Fq 'Clippy the private MiniMax H3 artifact qualifier' <<< "$rust_gate" \
  || fail "main Rust suite no longer checks the private qualifier"
grep -Fq 'Check with all optional features' <<< "$rust_gate" \
  || fail "main Rust suite no longer checks the optional feature combination"
grep -Fq 'name: Test (deterministic PR suite)' <<< "$rust_gate" \
  || fail "protected Rust status does not run deterministic workspace tests on pull requests"
grep -Fq -- '--skip catalog_api::catalog_live_test::live_search_free_text_can_find_manual_clip_components' <<< "$rust_gate" \
  || fail "the PR suite includes the flaky external catalog monitor"
grep -Fq 'name: Test (full main suite)' <<< "$rust_gate" \
  || fail "main Rust suite does not retain the complete workspace tests"
grep -Fq 'run: timeout --signal=TERM --kill-after=60s 20m cargo test --workspace' <<< "$rust_gate" \
  || fail "main Rust suite does not bound the complete workspace tests"
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
for path in .github/actionlint.yaml .github/workflows/\*\* scripts/tests/ci-routing-contract.sh; do
  grep -Fxq "              - '$path'" <<< "$workflow_filter" \
    || fail "workflow policy classifier omits $path"
done
require_text "$ci" 'name: Validate workflow syntax' \
  "workflow-only PRs have no lightweight static validation"
require_text "$ci" 'uses: docker://rhysd/actionlint:1.7.12' \
  "workflow-only PRs do not use the pinned lightweight actionlint image"
require_text "$ci" 'name: Validate CI routing policy' \
  "workflow-only PRs do not run the routing contract"

release_filter="$(extract_filter "$ci" release)"
for manifest in Cargo.toml Cargo.lock desktop/src-tauri/Cargo.toml desktop/src-tauri/Cargo.lock; do
  grep -Fxq "              - '$manifest'" <<< "$release_filter" \
    || fail "release classifier does not run the desktop Candle lock guard for $manifest changes"
done
require_text "$ci" \
  "bash scripts/tests/desktop-candle-nix-source-hash.sh" \
  "release CI does not verify the Nix hash for the locked desktop Candle source"
require_text "$ci" \
  "bash scripts/tests/desktop-dmg-packaging.sh" \
  "release CI does not exercise retrying desktop DMG packaging"
grep -Fxq "              - 'scripts/create-desktop-dmg.sh'" <<< "$release_filter" \
  || fail "release classifier omits the desktop DMG packager"
grep -Fxq "              - 'scripts/tests/desktop-dmg-packaging.sh'" <<< "$release_filter" \
  || fail "release classifier omits the desktop DMG packaging contract"
grep -Fq "'.github/workflows/**'" <<< "$release_filter" \
  || fail "workflow changes do not reach protected actionlint and routing contracts"

nix_filter="$(extract_filter "$ci" nix)"
if grep -Fq 'desktop/**' <<< "$nix_filter"; then
  fail "desktop-only changes still start the mold-web Nix build"
fi
nix_job="$(extract_job "$ci" nix-web)"
grep -Fq "github.event_name == 'push' &&" <<< "$nix_job" \
  || fail "the sandboxed Nix web build still runs on pull requests"
grep -Fq "needs.changes.outputs.workflow == 'true'" <<< "$nix_job" \
  || fail "workflow-only changes do not exercise the Nix build after merge"

desktop_classifier="$(extract_job "$desktop" changes)"
require_text "$desktop" 'name: Classify desktop changes' \
  "desktop workflow has no path classifier"
require_text "$desktop" '      - "scripts/create-desktop-dmg.sh"' \
  "desktop workflow does not run when its DMG packager changes"
require_text "$desktop" '      - "scripts/tests/desktop-dmg-packaging.sh"' \
  "desktop workflow does not run when its DMG packaging contract changes"
require_text "$desktop" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.frontend == 'true'" \
  "desktop frontend job is not path-gated on PRs"
desktop_rust="$(extract_job "$desktop" desktop-rust)"
grep -Fq "needs.changes.outputs.rust == 'true'" <<< "$desktop_rust" \
  || fail "desktop Rust tests are not path-gated on PRs"
grep -Fq "runs-on: macos-14" <<< "$desktop_rust" \
  || fail "desktop native PRs do not compile on their target platform"
grep -Fq 'cargo fmt --manifest-path src-tauri/Cargo.toml -- --check' <<< "$desktop_rust" \
  || fail "desktop native PRs have no Rust formatting check"
grep -Fq 'cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings' <<< "$desktop_rust" \
  || fail "desktop native PRs have no Clippy gate"
grep -Fq 'cargo test --manifest-path src-tauri/Cargo.toml' <<< "$desktop_rust" \
  || fail "desktop native PRs have no test gate"
if grep -Fq "if: github.event_name == 'push'" <<< "$desktop_rust"; then
  fail "desktop native compilation remains main-only"
fi
if grep -Fq 'Bundle macOS packaging inputs (debug .app)' <<< "$desktop_rust"; then
  fail "desktop PRs still build a macOS app bundle"
fi
desktop_linux="$(extract_job "$desktop" desktop-linux)"
grep -Fq "if: github.event_name != 'pull_request'" <<< "$desktop_linux" \
  || fail "Desktop Linux still runs on pull requests"
desktop_windows="$(extract_job "$desktop" desktop-windows)"
grep -Fq "if: github.event_name != 'pull_request'" <<< "$desktop_windows" \
  || fail "Desktop Windows still runs on pull requests"
if grep -Fq 'bunx vue-tsc -b' "$desktop"; then
  fail "desktop frontend still typechecks once explicitly and again during build"
fi
desktop_frontend="$(extract_job "$desktop" desktop-frontend)"
grep -Fq 'run: bun run fmt:check' <<< "$desktop_frontend" \
  || fail "desktop frontend PRs have no formatting check"
for command in 'run: bun run test' 'run: bun run build'; do
  grep -Fq "$command" <<< "$desktop_frontend" \
    || fail "desktop frontend PRs omit $command"
done
if grep -Fq 'desktop/**' <<< "$desktop_classifier"; then
  fail "desktop frontend classifier is broad enough to include native/mobile files"
fi
grep -Fq 'desktop/src/components/**' <<< "$desktop_classifier" \
  || fail "desktop frontend classifier does not track desktop components"
grep -Fq 'desktop/src-tauri/**' <<< "$desktop_classifier" \
  || fail "desktop native classifier does not track the Tauri crate"
if grep -Fq 'desktop/src/mobile/**' "$desktop"; then
  fail "mobile-only changes leak into the desktop workflow instead of the mobile workflows"
fi
[[ "$(grep -Fc 'crates/mold-scheduler/**' "$desktop")" -eq 2 ]] \
  || fail "desktop main trigger/classifier does not track its scheduler dependency"

require_text "$ios" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "iOS workflow does not cancel superseded PR runs"
require_text "$ios" 'name: Classify iOS changes' \
  "iOS workflow has no path classifier"
ios_frontend_filter="$(extract_filter "$ios" frontend)"
if grep -Fq "'apps/mobile/**'" <<< "$ios_frontend_filter"; then
  fail "native-only iOS changes still run mobile formatting"
fi
grep -Fq "desktop/src/mobile/**" <<< "$ios_frontend_filter" \
  || fail "iOS frontend classifier omits mobile source formatting"
require_text "$ios" \
  "runs-on: macos-14" \
  "iOS native PRs do not compile on their target platform"
require_text "$ios" 'run: ../scripts/tests/ios-release-assets.sh' \
  "iOS frontend job does not validate the built mobile entry"
require_text "$ios" 'run: bunx vitest run src/mobile' \
  "iOS frontend job does not retain its mobile-specific tests"
ios_frontend="$(extract_job "$ios" frontend)"
if grep -Fq "if: github.event_name == 'push'" <<< "$ios_frontend"; then
  fail "iOS frontend builds and tests remain main-only"
fi
ios_rust="$(extract_job "$ios" simulator-rust)"
grep -Fq 'run: cargo fmt -- --check' <<< "$ios_rust" \
  || fail "iOS native PRs have no Rust formatting check"
grep -Fq 'run: cargo clippy --target aarch64-apple-ios-sim -- -D warnings' <<< "$ios_rust" \
  || fail "iOS simulator compilation does not run on pull requests"
if grep -Fq 'run: cargo check --target aarch64-apple-ios-sim' "$ios"; then
  fail "iOS simulator still runs cargo check immediately before Clippy"
fi

require_text "$android" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "Android workflow does not cancel superseded PR runs"
require_text "$android" 'desktop/src/mobile/**' \
  "Android workflow does not run for shared mobile source changes"
require_text "$android" 'apps/mobile/**' \
  "Android workflow does not run for native mobile changes"
require_text "$android" 'runs-on: ubuntu-latest' \
  "Android builds do not run on the supported Linux host"
require_text "$android" 'name: Classify Android changes' \
  "Android workflow has no path classifier"
android_build_filter="$(extract_filter "$android" build)"
grep -Fq "desktop/src/mobile/**" <<< "$android_build_filter" \
  || fail "Android build classifier omits the shared mobile frontend"
android_native_filter="$(extract_filter "$android" native_tests)"
grep -Fq "apps/mobile/plugins/android/**" <<< "$android_native_filter" \
  || fail "Android native-test classifier omits the Kotlin bridge"
require_text "$android" 'targets: aarch64-linux-android' \
  "Android workflow does not install its Rust target"
require_text "$android" '"ndk;27.0.12077973"' \
  "Android workflow does not pin the repository NDK"
require_text "$android" \
  'run: cargo tauri android build --debug --apk --target aarch64 --ci' \
  "Android workflow does not build the ARM64 APK"
require_text "$release_workflow" 'build-android-apk:' \
  "Release workflow does not build the signed Android APK"
require_text "$release_workflow" 'targets: aarch64-linux-android,armv7-linux-androideabi' \
  "Release workflow does not install both universal APK Rust targets"
require_text "$release_workflow" 'artifacts/Mold-android.apk' \
  "Release workflow does not publish a raw Android APK"
require_text "$release_workflow" 'verify --verbose --print-certs' \
  "Release workflow does not verify the Android APK signature"
require_text "$android" 'KERNEL=="kvm", GROUP="kvm", MODE="0666"' \
  "Android emulator CI does not enable KVM"
require_text "$android" 'api-level: 35' \
  "Android workflow does not exercise a modern emulator"
require_text "$android" \
  './gradlew --no-daemon :tauri-plugin-mold-mobile-native:connectedDebugAndroidTest' \
  "Android workflow does not run native instrumentation tests"
require_text "$android_gradle" 'com.google.mlkit:barcode-scanning:17.3.0' \
  "Android pairing does not bundle its barcode decoder for first-run and offline use"

echo "PASS: CI routing contract"
