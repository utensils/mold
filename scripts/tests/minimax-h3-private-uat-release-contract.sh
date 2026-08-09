#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

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

require_text crates/mold-candle/Cargo.toml \
  'h3-private-uat = []' \
  "mold-candle does not keep the private H3 runtime behind its own feature"
require_text crates/mold-inference/Cargo.toml \
  'h3-private-uat = ["mold-candle/h3-private-uat"]' \
  "mold-inference does not narrowly forward the private H3 runtime feature"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-private-uat"]' \
  "the private H3 artifact qualifier is reachable without both development features"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '"mold.minimax-h3.private-uat-artifact-reader.v1"' \
  "the private H3 artifact reader has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_qwen_support.rs \
  '"mold.minimax-h3.private-uat-qwen-support-loader.v1"' \
  "the private H3 Qwen support loader has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'pub enum Token {}' \
  "private H3 FL2VA composition is not safely uninhabited before scheduler memory binding"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  '_activation: admitted_overlap_seal::Token,' \
  "private H3 FL2VA overlap authority no longer contains its uninhabited token"
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'pub(crate) struct H3PrivateBoundComfyStream' \
  "private H3 checkpoint load lacks a non-cloneable pre-allocation binding"
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'bound: H3PrivateBoundComfyStream,' \
  "private H3 resident loader still accepts unbound checkpoint and attention values"
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'device: Device,' \
  "private H3 bound checkpoint does not retain the validated Candle device"
attempt_owner=$(sed -n \
  '/struct H3PrivateAttemptOwner/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq 'device: Device,' <<<"$attempt_owner"; then
  fail "private H3 singular attempt owner does not retain the validated Candle device"
fi
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'struct H3PrivateBoundExecution<E>' \
  "private H3 route capture returns forgeable lease/device parts"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'fn new(bound: H3PrivateBoundExecution<E>, artifacts: A)' \
  "private H3 attempt construction can mix an execution lease with another device"
execution_projection=$(sed -n \
  '/impl<E, A> H3BackendExecutionLease for H3PrivateExecutionProjection/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq '&self.owner.device' <<<"$execution_projection" \
  || grep -Fq 'execution.device()' <<<"$execution_projection"; then
  fail "private H3 execution projections can repoll a safe lease for a different device"
fi
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  '.verify_current_release_candidate_dispatch(device)' \
  "private H3 binding does not verify the actual compiled attention candidate"
for method in \
  transformer_task \
  transformer_checkpoint_content_sha256 \
  transformer_checkpoint_layout_identity_sha256 \
  transformer_checkpoint_identity_sha256 \
  transformer_policy_identity_sha256; do
  require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
    "fn ${method}(&self)" \
    "private H3 artifact lease omits ${method}"
done
composer=$(sed -n \
  '/fn compose_private_comfy_fl2va_runtime/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
bind_line=$(grep -nF 'let bound_stream = bind_private_comfy_stream(' <<<"$composer" | cut -d: -f1)
attempt_line=$(grep -nF 'let attempt = H3PrivateAttemptAuthority::new' <<<"$composer" | cut -d: -f1)
qwen_line=$(grep -nF 'let qwen = H3PrivateQwenAdapter::load_authorized' <<<"$composer" | cut -d: -f1)
if [[ -z "$bind_line" || -z "$attempt_line" || -z "$qwen_line" ]] \
  || (( bind_line >= attempt_line || bind_line >= qwen_line )); then
  fail "private H3 attention/checkpoint binding no longer precedes attempt and Qwen allocation"
fi
if [[ $(grep -Fc 'capture_private_execution_route(execution_lease' <<<"$composer") -ne 1 ]] \
  || grep -Fq 'execution_lease.device()' <<<"$composer"; then
  fail "private H3 composer must consume and capture the execution lease route exactly once"
fi
if [[ $(grep -Fc 'bound_execution.device()' <<<"$composer") -ne 1 ]] \
  || [[ $(grep -Fc 'H3PrivateAttemptAuthority::new(bound_execution, artifact_lease)' <<<"$composer") -ne 1 ]]; then
  fail "private H3 bound execution token is not shared with attention binding and consumed by the attempt"
fi
capture=$(sed -n \
  '/fn capture_private_execution_route/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if [[ $(grep -Fc 'execution.device()' <<<"$capture") -ne 1 ]] \
  || [[ $(grep -Fc 'lease_id: execution.lease_id().into()' <<<"$capture") -ne 1 ]]; then
  fail "private H3 route capture must bind one device and a concrete lease identity"
fi
loader=$(sed -n \
  '/pub(crate) fn load_and_pair_private_comfy_stream/,/^[[:space:]]*{/p' \
  crates/mold-inference/src/minimax_h3/private_runtime.rs)
if grep -Fq 'device:' <<<"$loader"; then
  fail "private H3 resident loader still accepts an external device after binding"
fi
overlap_seal=$(sed -n '/mod admitted_overlap_seal {/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq '#[cfg(not(test))]' <<<"$overlap_seal"; then
  fail "private H3 FL2VA overlap seal is constructible in non-test builds"
fi
overlap_constructor=$(sed -n \
  '/impl H3PrivateFl2VaMemoryOverlapAuthority {/,/fn validate(&self)/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq '#[cfg(test)]' <<<"$overlap_constructor"; then
  fail "private H3 FL2VA overlap authority exposes a non-test constructor"
fi
if [[ $(grep -Fc 'pub(crate) fn new(' <<<"$overlap_constructor") -ne 1 ]] \
  || [[ $(grep -Fc 'Self {' <<<"$overlap_constructor") -ne 1 ]]; then
  fail "private H3 FL2VA overlap authority gained an unguarded construction path"
fi
if grep -Eq '^[[:space:]]*pub([[:space:]]*\([^)]*\))?[[:space:]]+(type[[:space:]]+H3PrivateComfyPipelineRuntime|fn[[:space:]]+compose_private_comfy_fl2va_runtime)' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs; then
  fail "private H3 FL2VA composition exposes an into-runtime escape"
fi
if grep -Eq '(transmute|MaybeUninit|::zeroed|unsafe[[:space:]]*\{).*(H3PrivateFl2VaMemoryOverlapAuthority|admitted_overlap_seal)' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs; then
  fail "private H3 FL2VA overlap authority has an unsafe fabrication path"
fi
if [[ $(grep -Fxc 'mod vae_free_inner_seal {' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs) -ne 1 ]] \
  || [[ $(grep -Fxc 'pub(crate) mod vae_free_inner_seal {' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs) -ne 1 ]]; then
  fail "private H3 VAE-free seal is not private in production and test-visible only"
fi
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub const H3_PRIVATE_AUTHORIZATION_SCOPE: &str = "private-h3-uat";' \
  "private H3 qualification does not use the generic private authorization scope"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";' \
  "private H3 qualification does not require the reviewed authorization schema"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'const REVIEWED_AUTHORIZATION_EVIDENCE_SHA256: &str =' \
  "private H3 qualification does not pin the accepted authorization evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d' \
  "private H3 qualification pins a different authorization evidence identity"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'if record.source_document_sha256 != reviewed_evidence_sha256 {' \
  "private H3 qualification does not reject self-declared authorization evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'fn production_scope_rejects_unreviewed_evidence() {' \
  "private H3 qualification lacks a regression test for unreviewed evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub authorization_record_sha256: String,' \
  "private H3 qualification report does not bind the authorization record identity"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub authorization_source_document_sha256: String,' \
  "private H3 qualification report does not bind the authorization source identity"
require_text crates/mold-inference/src/bin/h3_artifact_qualification.rs \
  'remove("--authorization-record")' \
  "private H3 qualifier CLI does not require the external authorization record"
if grep -Eq 'H3_PRIVATE_HOST_AUTHORITY_SHA256|H3_PRIVATE_MODELS_ROOT|/etc/hostname|/storage/.*/minimax-h3' \
  crates/mold-inference/src/minimax_h3/private_qualification.rs \
  crates/mold-inference/src/bin/h3_artifact_qualification.rs; then
  fail "private H3 qualification regressed to host-name or hardcoded-path authority"
fi
if grep -Fq -- '--authorization-scope' \
  crates/mold-inference/src/bin/h3_artifact_qualification.rs; then
  fail "private H3 qualifier accepts a caller-asserted authorization scope"
fi

if grep -Eq '^h3-private-uat[[:space:]]*=' crates/mold-cli/Cargo.toml; then
  fail "mold-ai forwards the private H3 UAT feature into a runnable product binary"
fi
server_dev_dependencies=$(sed -n '/^\[dev-dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if ! grep -Eq '^mold-inference[[:space:]]*=.*path[[:space:]]*=[[:space:]]*"\.\./mold-inference".*package[[:space:]]*=[[:space:]]*"mold-ai-inference".*features[[:space:]]*=[[:space:]]*\["h3-private-uat"\]' \
  <<<"$server_dev_dependencies"; then
  fail "mold-server tests do not unify the private H3 loader for bridge verification"
fi
server_dependencies=$(sed -n '/^\[dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if grep -Eq '^mold-inference[[:space:]]*=.*h3-private-uat' <<<"$server_dependencies"; then
  fail "mold-ai-server activates the private H3 UAT path through an ordinary dependency"
fi
if sed -n '/^\[features\]/,/^\[/p' crates/mold-server/Cargo.toml \
  | grep -Eq '^h3-private-uat[[:space:]]*='; then
  fail "mold-ai-server forwards the private H3 UAT feature into a runnable product binary"
fi

release_feature_sources="$({
  grep -E -- '--features|releaseFeatures[[:space:]]*=|buildFeatures[[:space:]]*=' \
    .github/workflows/release.yml \
    .github/workflows/desktop.yml \
    Dockerfile \
    flake.nix \
    packaging/aur/mold-ai/PKGBUILD \
    packaging/aur/mold-ai-git/PKGBUILD
  sed -n '/^[[:space:]]*releaseFeatures =/,/^[[:space:]]*completionFeatures =/p' flake.nix
} || true)"
if LC_ALL=C tr -cs '[:alnum:]_-' '\n' <<<"$release_feature_sources" \
  | grep -Fxq h3-private-uat; then
  fail "a published release feature set compiles the private H3 UAT path"
fi

require_text scripts/verify-h3-release-exclusion.sh \
  "private_uat_marker='mold.minimax-h3.private-uat-artifact-reader.v1'" \
  "published binary verification does not reject the private H3 UAT marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'" \
  "published binary verification does not reject the private H3 Qwen support marker"
require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-inference --features dev-bins,h3-private-uat --bin h3_artifact_qualification -- -D warnings' \
  "CI does not compile the authorization-bound artifact qualifier"
require_text .github/workflows/ci.yml \
  'cargo test -p mold-ai-inference --lib --features h3-private-uat minimax_h3' \
  "CI does not execute the private H3 foundation tests"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/private_*.rs'" \
  "private H3 runtime source changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/mod.rs'" \
  "private H3 module-gating changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  'run: bash scripts/tests/minimax-h3-private-uat-release-contract.sh' \
  "release CI does not run the private H3 exclusion contract"
require_text docs/qualification/minimax-h3.md \
  'h3_artifact_qualification' \
  "the private artifact qualifier has no operator runbook"
require_text docs/qualification/minimax-h3.md \
  '--authorization-record' \
  "the private artifact qualifier runbook omits its external authorization record"

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
ordinary_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
private_marker='mold.minimax-h3.private-uat-artifact-reader.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
printf '%s\n' "$ordinary_marker" >"$scratch_dir/ordinary"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary" >/dev/null
printf '%s\n%s\n' "$ordinary_marker" "$private_marker" >"$scratch_dir/private"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 artifact reader"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_qwen_support_marker" \
  >"$scratch_dir/private-qwen-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-support" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 Qwen support loader"
fi

echo "PASS: MiniMax H3 private-UAT release contract"
