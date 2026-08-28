#!/usr/bin/env bash
# shellcheck disable=SC1003,SC2016 # Literal workflow/Docker/Nix source is asserted below.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  echo "cuda distribution contract: $*" >&2
  exit 1
}

require_text() {
  local file="$1"
  local text="$2"
  grep -Fq -- "$text" "$repo_root/$file" \
    || fail "$file is missing: $text"
}

require_ci_release_path() {
  local path="$1"
  local block

  block="$(
    awk '
      $0 == "            release:" { in_release = 1; next }
      in_release && /^            [[:alnum:]_-]+:$/ { exit }
      in_release { print }
    ' "$repo_root/.github/workflows/ci.yml"
  )"
  [[ -n "$block" ]] || fail ".github/workflows/ci.yml has no release path classifier"
  if [[ "$path" == .github/workflows/* ]] \
    && grep -Fq -- "- '.github/workflows/**'" <<<"$block"; then
    return 0
  fi
  grep -Fq -- "- '${path}'" <<<"$block" \
    || fail ".github/workflows/ci.yml release classifier is missing: $path"
}

require_release_job_text() {
  local job="$1"
  local text="$2"
  local block

  block="$(
    awk -v job="  ${job}:" '
      $0 == job { in_job = 1 }
      in_job && /^  [[:alnum:]_-]+:$/ && $0 != job { exit }
      in_job { print }
    ' "$repo_root/$release"
  )"
  [[ -n "$block" ]] || fail "$release is missing job: $job"
  grep -Fq -- "$text" <<<"$block" \
    || fail "$release job $job is missing: $text"
}

release_job_needs() {
  local job="$1"
  awk -v job="  ${job}:" '
      $0 == job { in_job = 1 }
      in_job && /^  [[:alnum:]_-]+:$/ && $0 != job { exit }
      in_job && /^    needs:/ { in_needs = 1 }
      in_needs && /^    [[:alnum:]_-]+:/ && $0 !~ /^    needs:/ { exit }
      in_needs { print }
    ' "$repo_root/$release"
}

require_release_job_need() {
  local job="$1"
  local dependency="$2"
  local needs

  needs="$(release_job_needs "$job")"
  [[ -n "$needs" ]] || fail "$release job $job has no needs declaration"
  grep -Eq "(^|[][,:[:space:]])${dependency}([],:[:space:]]|$)" <<<"$needs" \
    && return 0
  fail "$release job $job does not depend on $dependency"
}

reject_release_job_need() {
  local job="$1"
  local dependency="$2"
  local needs

  needs="$(release_job_needs "$job")"
  [[ -n "$needs" ]] || fail "$release job $job has no needs declaration"
  if grep -Eq "(^|[][,:[:space:]])${dependency}([],:[:space:]]|$)" <<<"$needs"; then
    fail "$release job $job must not depend on lower-priority $dependency"
  fi
}

release=".github/workflows/release.yml"
cache_workflow=".github/workflows/nix-cache.yml"
require_text "$release" \
  "MOLD_DISTRIBUTION_IMAGE_VERSION: \${{ startsWith(github.ref, 'refs/tags/v') && github.ref_name || 'latest' }}"
require_text ".github/workflows/desktop-distribution.yml" \
  'echo "MOLD_DISTRIBUTION_IMAGE_VERSION=$distribution_image_version"'

# Keep this table-driven: adding a future official CUDA target extends one list,
# while the same assertions apply to any number of release variants.
for target in 86 89 100 120; do
  job="build-linux-sm${target}"
  require_release_job_text "$job" "CUDA_COMPUTE_CAP: \"${target}\""
  require_release_job_text "$job" \
    "mold-x86_64-unknown-linux-gnu-cuda-sm${target}.tar.gz"
  require_release_job_text "$job" \
    "scripts/seal-cuda-ptx-manifest.py target/release/mold ${target} target/release/build"
  require_release_job_text "$job" \
    "scripts/verify-cuda-release-binary.sh target/release/mold ${target}"
done
require_text "scripts/verify-cuda-release-binary.sh" \
  'failed its GPU-less loader smoke'
require_text "scripts/verify-cuda-release-binary.sh" \
  'CUDA release binary does not include NVML loader support'
require_text "scripts/verify-cuda-release-binary.sh" \
  'probe-cuda-embedded-ptx.py'
require_text "scripts/verify-cuda-release-binary.sh" \
  '--extract-only'
require_text "scripts/probe-cuda-embedded-ptx.py" \
  '.mold_cuda_ptx'
require_text "Dockerfile" \
  'scripts/seal-cuda-ptx-manifest.py /build/target/release/mold'
require_text "Dockerfile" \
  'scripts/probe-cuda-embedded-ptx.py /build/target/release/mold'
require_text "Dockerfile" \
  'COPY crates/mold-candle/src/comfy_int8/cuda/int8_linear.cu crates/mold-candle/src/comfy_int8/cuda/int8_linear.cu'
while IFS= read -r workspace_member; do
  require_text "Dockerfile" \
    "COPY ${workspace_member}/Cargo.toml ${workspace_member}/Cargo.toml"
  if [[ -f "$repo_root/${workspace_member}/build.rs" ]]; then
    require_text "Dockerfile" \
      "COPY ${workspace_member}/build.rs ${workspace_member}/build.rs"
  fi
  if [[ -d "$repo_root/${workspace_member}/build_support" ]]; then
    require_text "Dockerfile" \
      "COPY ${workspace_member}/build_support/ ${workspace_member}/build_support/"
  fi
  if [[ -f "$repo_root/${workspace_member}/src/lib.rs" ]]; then
    source_name="lib.rs"
  else
    source_name="main.rs"
  fi
  require_text "Dockerfile" \
    "${workspace_member}/src/${source_name}"
done < <(
  python3 - "$repo_root/Cargo.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as manifest:
    for member in tomllib.load(manifest)["workspace"]["members"]:
        print(member)
PY
)
if awk '
  /# Build dependencies only/ { in_dependency_build = 1 }
  in_dependency_build { print }
  /# Now copy the real source code/ { exit }
' "$repo_root/Dockerfile" | grep -Fq '|| true'; then
  fail "Docker dependency-cache build must not hide workspace resolution failures"
fi
require_text "Dockerfile" \
  'python3 \'
require_text "scripts/probe-cuda-embedded-ptx.py" \
  'CUDA_ERROR_INVALID_PTX'
require_text "scripts/probe-cuda-embedded-ptx.py" \
  '--expect-incompatible'
require_text "scripts/probe-cuda-embedded-ptx.py" \
  '"observed_targets"'
if grep -Eq '^[^#]*cuobjdump[[:space:]]+--' "$repo_root/scripts/verify-cuda-release-binary.sh"; then
  fail "final CUDA binary verification must not require dead-stripped cuobjdump sections"
fi

for target in 86 100; do
  require_text "$release" "cuda_cap: \"${target}\""
  require_text "$release" "suffix: \"-sm${target}\""
  require_text "$release" "artifacts/mold-x86_64-unknown-linux-gnu-cuda-sm${target}.tar.gz"
done

for release_job in release-latest release-version release-native; do
  reject_release_job_need "$release_job" "docker"
  reject_release_job_need "$release_job" "build-nix-distribution"
done
require_release_job_need "release-version" "build-macos"
require_release_job_need "release-version" "build-desktop-dmg"
reject_release_job_need "release-latest" "build-windows"
for target in 86 89 100 120; do
  require_release_job_need "release-native" "build-linux-sm${target}"
done
require_release_job_need "release-native" "release-version"
require_release_job_need "release-native" "build-windows"
require_release_job_need "release-containers" "release-native"
require_release_job_need "release-containers" "docker"
require_release_job_text "release-latest" \
  "Pin rolling tag to this release"
require_release_job_text "release-latest" \
  'contents: write'
require_release_job_text "release-latest" \
  'prerelease: true'
require_release_job_text "release-latest" \
  'make_latest: false'
require_release_job_text "release-latest" \
  'GH_TOKEN: ${{ github.token }}'
require_release_job_text "release-latest" \
  '"repos/${GITHUB_REPOSITORY}/git/refs/tags/latest"'
require_release_job_text "release-latest" \
  '"repos/${GITHUB_REPOSITORY}/git/ref/tags/latest"'
require_release_job_text "release-latest" \
  '--method PATCH'
require_release_job_text "release-latest" \
  '-f "sha=${GITHUB_SHA}"'
require_release_job_text "release-latest" \
  '-F force=true'
require_release_job_text "release-latest" \
  'for attempt in $(seq 1 10); do'
require_release_job_text "release-latest" \
  '[[ "$attempt" -eq 10 ]] || sleep 2'
require_release_job_text "release-latest" \
  'if [[ "$tag_sha" != "$GITHUB_SHA" ]]; then'
main_freshness_guard_count="$(
  awk '
    $0 == "  release-latest:" { in_job = 1 }
    in_job && /^  [[:alnum:]_-]+:$/ && $0 != "  release-latest:" { exit }
    in_job { print }
  ' "$repo_root/$release" |
    grep -Fc 'scripts/check-desktop-nightly-main-head.sh "$GITHUB_SHA"'
)"
[[ "$main_freshness_guard_count" -eq 2 ]] \
  || fail "release-latest must reject stale main before release and tag mutation"

release_guard_lines="$(
  grep -nF 'scripts/check-desktop-nightly-main-head.sh "$GITHUB_SHA"' \
    "$repo_root/$release" |
    cut -d: -f1
)"
first_release_guard_line="$(sed -n '1p' <<<"$release_guard_lines")"
second_release_guard_line="$(sed -n '2p' <<<"$release_guard_lines")"
release_upload_line="$(
  grep -n -m1 'uses: softprops/action-gh-release@v2' "$repo_root/$release" |
    cut -d: -f1
)"
rolling_tag_patch_line="$(
  grep -n -m1 '"repos/${GITHUB_REPOSITORY}/git/refs/tags/latest"' \
    "$repo_root/$release" |
    cut -d: -f1
)"
[[ "$first_release_guard_line" -lt "$release_upload_line" \
  && "$second_release_guard_line" -gt "$release_upload_line" \
  && "$second_release_guard_line" -lt "$rolling_tag_patch_line" ]] \
  || fail "fresh-main guards must surround release mutation and tag movement"
require_text "$release" 'tags: ["v*"]'
require_text "$release" '- cron: "47 12 * * *"'
require_text "$release" \
  'group: release-${{ github.event_name == '\''schedule'\'' && '\''docker-nightly'\'' || github.ref }}'
require_release_job_text "docker" \
  "if: github.event_name == 'schedule' || startsWith(github.ref, 'refs/tags/v')"
require_release_job_text "docker" \
  'max-parallel: ${{ github.event_name == '\''schedule'\'' && 2 || 6 }}'
require_release_job_text "release-latest" \
  "if: github.event_name != 'schedule' && github.ref == 'refs/heads/main'"
require_release_job_text "docker" \
  'type=raw,value=latest${{ matrix.suffix }},enable={{is_default_branch}}'
require_release_job_text "docker" \
  'type=raw,value=sha-${{ github.sha }}${{ matrix.suffix }}'
require_release_job_text "docker" \
  'MOLD_DISTRIBUTION_IMAGE_VERSION=${{ env.MOLD_DISTRIBUTION_IMAGE_VERSION }}'
require_release_job_text "docker" \
  'MOLD_GIT_SHA=${{ github.sha }}'
require_release_job_text "docker" \
  '-e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64'
require_release_job_text "docker" \
  '-e EXPECTED_SOURCE_SHA="$GITHUB_SHA"'
require_release_job_text "docker" \
  '-c '\''grep -aFq "$EXPECTED_SOURCE_SHA" "$(command -v mold)" && exec mold version'\'''
require_release_job_text "docker" \
  'grep -Fq "${GITHUB_SHA:0:7}" <<<"$version"'
require_text "Dockerfile" 'ARG MOLD_GIT_SHA'
require_text "Dockerfile" 'ENV MOLD_GIT_SHA=${MOLD_GIT_SHA}'

docker_python_line="$(grep -n -m1 'python3 \\' "$repo_root/Dockerfile" | cut -d: -f1)"
docker_seal_line="$(
  grep -n -m1 'scripts/seal-cuda-ptx-manifest.py /build/target/release/mold' \
    "$repo_root/Dockerfile" | cut -d: -f1
)"
[[ "$docker_python_line" -lt "$docker_seal_line" ]] \
  || fail "Docker builder must install Python before sealing PTX"
docker_source_sha_env_line="$(
  grep -n -m1 'ENV MOLD_GIT_SHA=${MOLD_GIT_SHA}' "$repo_root/Dockerfile" | cut -d: -f1
)"
grep -Fq 'COPY .cargo/config.toml .cargo/config.toml' "$repo_root/Dockerfile" \
  || fail "Docker H3 runtime identity build omits .cargo/config.toml"
docker_builder_packages="$(sed -n '/apt-get update && apt-get install/,/&& break/p' "$repo_root/Dockerfile")"
for package in clang lld; do
  grep -Eq "^[[:space:]]+${package}[[:space:]]*\\\\$" <<< "$docker_builder_packages" \
    || fail "Docker builder omits ${package} required by .cargo/config.toml"
done
docker_dependency_build_line="$(
  grep -n -m1 -E '^RUN cargo build --release -p mold-ai([[:space:]]|$)' \
    "$repo_root/Dockerfile" | cut -d: -f1
)"
docker_real_build_line="$(
  grep -n -m1 -E '^[[:space:]]+cargo build --release -p mold-ai([[:space:]]|$)' \
    "$repo_root/Dockerfile" | cut -d: -f1
)"
[[ "$docker_source_sha_env_line" -gt "$docker_dependency_build_line" \
  && "$docker_source_sha_env_line" -lt "$docker_real_build_line" ]] \
  || fail "Docker source SHA must not invalidate the dependency-cache build"
docker_runtime_copy_line="$(
  grep -n -m1 'COPY --from=builder /build/target/release/mold' \
    "$repo_root/Dockerfile" | cut -d: -f1
)"
[[ "$docker_seal_line" -lt "$docker_runtime_copy_line" ]] \
  || fail "Docker PTX sealing must precede runtime image publication"
require_release_job_text "docker" \
  'Create once and verify immutable stable tag'
require_release_job_text "docker" \
  '--prefer-index=false'
require_release_job_text "docker" \
  'if [[ -n "$existing" && "$existing" != "$BUILT_DIGEST" ]]'
for target in 80 86 90 100 120; do
  require_release_job_text "docker" "cuda_cap: \"${target}\""
  require_release_job_text "docker" "suffix: \"-sm${target}\""
done
require_release_job_text "docker" 'cuda_cap: "89"'
require_release_job_text "docker" 'suffix: ""'
require_ci_release_path "$cache_workflow"
require_text "$cache_workflow" 'tags: ["v*"]'
if grep -Fq 'branches: [main]' "$repo_root/$cache_workflow"; then
  fail "$cache_workflow must not build Cachix closures on main pushes"
fi
require_text "$cache_workflow" 'group: nix-cache-${{ github.ref }}'
require_text "$cache_workflow" 'cancel-in-progress: false'
require_text "$cache_workflow" 'permissions:'
require_text "$cache_workflow" 'contents: read'
require_text "$cache_workflow" 'continue-on-error: true'
require_text "$cache_workflow" 'uses: cachix/cachix-action@v17'
require_text "$cache_workflow" 'name: mold'
require_text "$cache_workflow" 'authToken: ${{ secrets.CACHIX_AUTH_TOKEN }}'
require_text "$cache_workflow" 'useDaemon: false'
require_text "$cache_workflow" 'pathsToPush: ${{ matrix.out_link }}'
for package in mold mold-sm86 mold-sm100 mold-desktop-sm86; do
  require_text "$cache_workflow" "package: $package"
done
require_text "$cache_workflow" \
  'nix build ".#${{ matrix.package }}" --out-link "${{ matrix.out_link }}"'
if grep -Fq 'cachix/cachix-action' "$repo_root/$release"; then
  fail "$release must not wait for best-effort Cachix publication"
fi
if grep -Fq 'build-nix-distribution:' "$repo_root/$release"; then
  fail "$release must not contain blocking Nix distribution builds"
fi
require_text "flake.nix" 'extra-substituters = [ "https://mold.cachix.org" ];'
require_text "flake.nix" \
  '"mold.cachix.org-1:9HBc/bEXDdpbxMjOwpaIDpjZqBh9JYg0h5Fipm+D8m4="'
require_release_job_need "publish" "release-version"
require_release_job_need "publish-aur" "release-native"
require_release_job_text "release-latest" \
  'Include latest completed Windows artifacts'
require_release_job_text "release-latest" \
  "-printf '%f\\0'"
require_release_job_text "release-latest" \
  'xargs -0 sha256sum -- > SHA256SUMS'
require_release_job_text "release-version" \
  'sha256sum -- *.tar.gz *.dmg *.zip > SHA256SUMS'
for checksum_job in release-native release-containers; do
  require_release_job_text "$checksum_job" \
    'sha256sum -- *.tar.gz *.dmg *.zip *.exe *.cer *.apk > SHA256SUMS'
done
if grep -Fq 'sha256sum ./*' "$repo_root/$release"; then
  fail "$release must write asset names without a ./ prefix for legacy updater compatibility"
fi
require_release_job_text "release-containers" \
  'scripts/create-container-digest-manifest.sh'
require_release_job_text "release-containers" \
  'artifacts/mold-container-digests.json'
require_release_job_text "release-containers" \
  'scripts/create-release-provenance.sh'
require_release_job_text "release-containers" \
  'artifacts/mold-release-provenance.json'
require_release_job_text "release-containers" \
  'sha256sum mold-release-provenance.json >> SHA256SUMS'
require_release_job_text "release-containers" \
  'sha256sum mold-container-digests.json >> SHA256SUMS'
provenance_line="$(grep -n 'scripts/create-release-provenance.sh' "$repo_root/$release" | cut -d: -f1)"
provenance_checksum_line="$(
  grep -n 'sha256sum mold-release-provenance.json >> SHA256SUMS' \
    "$repo_root/$release" | cut -d: -f1
)"
[[ "$provenance_checksum_line" -gt "$provenance_line" ]] \
  || fail "release provenance checksum must be appended after provenance generation"
container_manifest_line="$(
  grep -n 'scripts/create-container-digest-manifest.sh' "$repo_root/$release" | cut -d: -f1
)"
container_checksum_line="$(
  grep -n 'sha256sum mold-container-digests.json >> SHA256SUMS' \
    "$repo_root/$release" | cut -d: -f1
)"
[[ "$container_manifest_line" -lt "$provenance_line" ]] \
  || fail "container digest manifest must be generated before release provenance"
[[ "$container_checksum_line" -gt "$container_manifest_line" \
  && "$container_checksum_line" -lt "$provenance_line" ]] \
  || fail "container digest manifest must be checksum-bound before provenance generation"

for phase_g_path in \
  .github/workflows/desktop-distribution.yml \
  website/deployment/nixos.md \
  crates/mold-cli/src/commands/lambda.rs \
  crates/mold-cli/src/commands/runpod.rs \
  desktop/src-tauri/src/runpod.rs \
  crates/mold-core/src/cuda_distribution.rs \
  crates/mold-server/Cargo.toml \
  scripts/verify-cuda-release-binary.sh \
  scripts/seal-cuda-ptx-manifest.py \
  scripts/probe-cuda-embedded-ptx.py \
  scripts/create-release-provenance.sh \
  scripts/create-container-digest-manifest.sh \
  scripts/qualify-cuda-sm86.sh \
  scripts/validate-cuda-qualification-report.py \
  scripts/verify-png-artifact.py \
  scripts/tests/cuda-distribution-contract.sh \
  scripts/tests/cuda-ptx-parser-contract.py \
  scripts/tests/install-cuda-arch.sh \
  scripts/tests/cuda-qualification-contract.sh; do
  require_ci_release_path "$phase_g_path"
done

require_text "flake.nix" 'mold-sm86 = mkMold "86";'
require_text "flake.nix" 'mold-sm100 = mkMold "100";'
require_text "flake.nix" 'mold-desktop-sm86 = mkMoldDesktop "86";'
require_text "flake.nix" 'moldCudaComputeCapability = computeCap;'
require_text "flake.nix" 'cuda-package-consistency ='
require_text "flake.nix" 'moduleWarnings "ada" (mkMold "86")'
require_text "nix/module.nix" 'packageCudaComputeCapability'
require_text "nix/module.nix" 'ada = "89";'
if grep -Fq 'lib.optionals (cfg.cudaArch ==' "$repo_root/nix/module.nix"; then
  fail "Nix CUDA warnings must compare package metadata, not fire unconditionally"
fi
for mapping in \
  '`"ampere"` → `packages.${system}.mold-sm86`' \
  '`"ada"` → `packages.${system}.mold`' \
  '`"blackwell-datacenter"` → `packages.${system}.mold-sm100`' \
  '`"blackwell"` → `packages.${system}.mold-sm120`'; do
  require_text "website/deployment/nixos.md" "$mapping"
done
if grep -Fq 'mold-desktop-sm100' "$repo_root/flake.nix"; then
  fail "B200 is server-only; flake.nix must not export mold-desktop-sm100"
fi
require_text "crates/mold-cli/src/skill/SKILL.md" \
  'B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported'

require_text "install.sh" 'echo "sm86"'
require_text "install.sh" 'echo "sm100"'
require_text "install.sh" '--query-gpu=index,uuid,compute_cap'
require_text "install.sh" 'nvidia-smi -L'
require_text "install.sh" 'CUDA_VISIBLE_DEVICES'
require_text "install.sh" 'SHA256SUMS'
require_text "install.sh" 'return 44'
require_text ".github/workflows/release.yml" 'MOLD_BUILD_CHANNEL:'
require_text ".github/workflows/release.yml" "'stable' || 'nightly'"
require_text "crates/mold-cli/src/commands/lambda.rs" \
  'resolve_distribution_image_reference'
require_text "crates/mold-cli/src/commands/runpod.rs" \
  'resolve_distribution_image_reference'
require_text "desktop/src-tauri/src/runpod.rs" \
  'resolve_distribution_image_reference'
require_text "crates/mold-core/src/cuda_distribution.rs" \
  'refusing mutable tag fallback'
require_text "crates/mold-core/src/cuda_distribution.rs" \
  'pub struct UnsupportedPublishedImagePlatform'
require_text "crates/mold-core/src/cuda_distribution.rs" \
  'linux/arm64'
require_text "crates/mold-cli/src/commands/runpod.rs" \
  'ensure_published_runpod_gpu_identity(&gpu_name, &gpu_display)'
require_text "crates/mold-cli/src/commands/runpod.rs" \
  'ensure_published_image_platform(gpu_type_id)'
require_text "crates/mold-cli/src/commands/runpod.rs" \
  'ensure_published_image_platform(display_name)'
require_text "Dockerfile" 'ARG MOLD_DISTRIBUTION_IMAGE_VERSION=latest'
require_text "Dockerfile" \
  'ENV MOLD_DISTRIBUTION_IMAGE_VERSION=${MOLD_DISTRIBUTION_IMAGE_VERSION}'
require_text "crates/mold-cli/Cargo.toml" 'nvml = ["mold-server/nvml"]'
require_text "crates/mold-cli/Cargo.toml" '"nvml"'
require_text "crates/mold-cli/Cargo.toml" 'flash-attn = ["mold-inference/flash-attn", "cuda"]'
require_text "crates/mold-server/Cargo.toml" \
  'cuda = ["mold-inference/cuda", "nvml", "dep:cudarc"]'
require_text "desktop/src-tauri/Cargo.toml" 'nvml = ["mold-server/nvml"]'
require_text "desktop/src-tauri/Cargo.toml" '"nvml"'

require_text "packaging/aur/mold-ai-bin/PKGBUILD" 'cuda-sm89.tar.gz'
require_text "packaging/aur/README.md" 'CUDA_COMPUTE_CAP=100'
if find "$repo_root/packaging/aur" -mindepth 1 -maxdepth 1 -type d \
  -name 'mold-ai-bin-sm100' -print -quit | grep -q .; then
  fail "mold-ai-bin-sm100 must not be published before real B200 qualification"
fi
if find "$repo_root/packaging/aur" -mindepth 1 -maxdepth 1 -type d \
  -name 'mold-ai-bin-sm86' -print -quit | grep -q .; then
  fail "mold-ai-bin-sm86 must not be published before the real RTX 3090 gate"
fi

require_text "website/guide/installation.md" \
  'B200 support is simulated, not hardware-qualified'
require_text "website/deployment/docker.md" \
  'B200 support is simulated, not hardware-qualified'
for grace_doc in \
  README.md \
  website/deployment/docker.md \
  website/deployment/lambda-cli.md \
  website/deployment/nixos.md \
  website/deployment/runpod-cli.md \
  website/guide/installation.md; do
  require_text "$grace_doc" \
    'GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported'
done
require_text "$release" \
  'GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.'
grace_release_notice_count="$(
  grep -Fc \
    'GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.' \
    "$repo_root/$release"
)"
[[ "$grace_release_notice_count" -eq 2 ]] \
  || fail "both generated release descriptions must carry the Grace platform warning"
if grep -Eq 'B200[[:space:]]*/[[:space:]]*GB200|B/GB200|B200/GB200' \
  "$repo_root/.github/workflows/release.yml"; then
  fail "generated release descriptions falsely route GB200 to an amd64 artifact"
fi
require_text "scripts/qualify-cuda-sm86.sh" 'sm86_attention_image_smoke'
require_text "scripts/qualify-cuda-sm86.sh" 'sm86_ptx_image_smoke'
require_text "scripts/qualify-cuda-sm86.sh" 'sm86_video_smoke'
require_text "scripts/qualify-cuda-sm86.sh" 'sm86_chained_video_smoke'
require_text "docs/qualification/cuda-sm86-report.schema.json" \
  '"hardware_qualified"'
require_text "docs/qualification/README.md" 'width = 256'
require_text "docs/qualification/README.md" 'height = 256'
[[ -x "$repo_root/scripts/tests/cuda-ptx-parser-contract.py" ]] \
  || fail "embedded PTX parser contract is not executable"

test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT

verifier_bin="$test_root/verifier-bin"
mkdir -p "$verifier_bin"
cat >"$verifier_bin/readelf" <<'EOF'
#!/usr/bin/env bash
echo '  Machine:                           Advanced Micro Devices X86-64'
EOF
cat >"$verifier_bin/ldd" <<'EOF'
#!/usr/bin/env bash
echo 'linux-vdso.so.1'
EOF
chmod +x "$verifier_bin/readelf" "$verifier_bin/ldd"

cat >"$test_root/verified-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
# .target instanceof
[[ "${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_86
.address_size 64
.visible .entry verified_probe() {
  ret;
}
PTX
EOF
cat >"$test_root/native-only-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
[[ "${1:-}" == version ]]
EOF
for ptx_target in 85 89 860; do
  cat >"$test_root/wrong-ptx-${ptx_target}-mold" <<EOF
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
[[ "\${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_${ptx_target}
.address_size 64
.visible .entry wrong_probe() {
  ret;
}
PTX
EOF
done
cat >"$test_root/mixed-ptx-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
[[ "${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_86
.address_size 64
.visible .entry sm86_probe() {
  ret;
}
.version 8.0
.target sm_89
.address_size 64
.visible .entry sm89_probe() {
  ret;
}
PTX
EOF
cat >"$test_root/malformed-ptx-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
[[ "${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_86x
.address_size 64
.visible .entry malformed_probe() {
  ret;
}
PTX
EOF
cat >"$test_root/incomplete-ptx-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
[[ "${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_86
.address_size 64
PTX
EOF
cat >"$test_root/spa-target-only-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
# Embedded SPA/static text, not PTX:
# .target sm_86
[[ "${1:-}" == version ]]
EOF
cat >"$test_root/verified-with-spa-noise-mold" <<'EOF'
#!/usr/bin/env bash
# nvmlDeviceGetCount_v2
# Embedded SPA/static target-like text must not affect the PTX target set:
# .target sm_89
[[ "${1:-}" == version ]] && exit 0
exit 1
: <<'PTX'
.version 8.0
.target sm_86
.address_size 64
.visible .entry verified_probe() {
  ret;
}
PTX
EOF
chmod +x "$test_root/verified-mold" "$test_root/native-only-mold" \
  "$test_root/mixed-ptx-mold" "$test_root/malformed-ptx-mold" \
  "$test_root/incomplete-ptx-mold" "$test_root/spa-target-only-mold" \
  "$test_root/verified-with-spa-noise-mold" \
  "$test_root"/wrong-ptx-*-mold

if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh \
    "$test_root/verified-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted unrelated heredoc PTX without a build-bound manifest"
fi
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh \
    "$test_root/verified-with-spa-noise-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted unsealed PTX mixed with unrelated static text"
fi
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh "$test_root/native-only-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted a binary without embedded target PTX"
fi
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh \
    "$test_root/spa-target-only-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted SPA/static .target text as embedded PTX"
fi
for ptx_target in 85 89 860; do
  if PATH="$verifier_bin:$PATH" \
    scripts/verify-cuda-release-binary.sh \
      "$test_root/wrong-ptx-${ptx_target}-mold" 86 >/dev/null 2>&1; then
    fail "CUDA verifier accepted wrong PTX target sm_${ptx_target} for sm_86"
  fi
done
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh "$test_root/mixed-ptx-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted mixed sm_86/sm_89 embedded PTX targets"
fi
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh "$test_root/malformed-ptx-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted a malformed PTX target directive"
fi
if PATH="$verifier_bin:$PATH" \
  scripts/verify-cuda-release-binary.sh "$test_root/incomplete-ptx-mold" 86 >/dev/null 2>&1; then
  fail "CUDA verifier accepted an incomplete PTX module"
fi

cat >"$test_root/embedded-ptx-fixture" <<'EOF'
fixture prefix
.version 8.0
.target sm_86
.address_size 64
.visible .entry exact_artifact_probe() {
  ret;
}
fixture suffix
EOF
chmod +x "$test_root/embedded-ptx-fixture"
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/embedded-ptx-fixture" 86 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe accepted arbitrary executable bytes without a sealed manifest"
fi
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/embedded-ptx-fixture" 89 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe accepted a target-mismatched fixture"
fi
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/mixed-ptx-mold" 86 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe accepted a mixed-target executable"
fi
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/malformed-ptx-mold" 86 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe accepted a malformed complete module"
fi
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/incomplete-ptx-mold" 86 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe accepted an incomplete target module"
fi
if scripts/probe-cuda-embedded-ptx.py \
  "$test_root/verified-with-spa-noise-mold" 86 --extract-only >/dev/null 2>&1; then
  fail "embedded PTX probe treated unrelated static text as build-bound PTX"
fi

TMPDIR="$test_root" python3 scripts/tests/cuda-ptx-parser-contract.py >/dev/null

if [[ -n "${MOLD_REAL_SM86_BINARY:-}" ]]; then
  PATH="${SYSTEM_PATH:-$PATH}" scripts/verify-cuda-release-binary.sh \
    "$MOLD_REAL_SM86_BINARY" 86 >/dev/null
fi

for target in sm80 sm86 sm89 sm90 sm100 sm120; do
  suffix="-$target"
  [[ "$target" == sm89 ]] && suffix=""
  jq -n \
    --arg target "$target" \
    --arg tag "0.20.2$suffix" \
    --arg digest "sha256:$(printf '%064d' "$((10#${target#sm} % 10))")" \
    '{target: $target, tag: $tag, digest: $digest}' \
    >"$test_root/$target.json"
done
scripts/create-container-digest-manifest.sh \
  "$test_root/container.json" \
  ghcr.io/utensils/mold \
  v0.20.2 \
  0000000000000000000000000000000000000000 \
  "$test_root"/sm*.json
jq -e '
  .schema_version == "mold.container-digests.v1"
  and .release_tag == "v0.20.2"
  and (.targets | keys | length) == 6
  and .targets.sm86.tag == "0.20.2-sm86"
  and (.targets.sm86.digest | startswith("sha256:"))
' "$test_root/container.json" >/dev/null

mkdir -p "$test_root/sm86" "$test_root/sm89"
printf 'distinct sm86 binary fixture\n' >"$test_root/sm86/mold"
printf 'distinct sm89 binary fixture\n' >"$test_root/sm89/mold"
tar -czf "$test_root/mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz" \
  -C "$test_root/sm86" mold
tar -czf "$test_root/mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz" \
  -C "$test_root/sm89" mold
(
  cd "$test_root"
  sha256sum \
    ./mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz \
    ./mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz \
    >SHA256SUMS
)
scripts/create-release-provenance.sh \
  "$test_root/provenance.json" \
  v0.20.2 \
  0000000000000000000000000000000000000000 \
  "$test_root/SHA256SUMS" \
  "$test_root"
jq -e '
  .schema_version == "mold.release.provenance.v1"
  and .artifacts.sm86.cuda_target == "sm_86"
  and .artifacts.sm89.cuda_target == "sm_89"
  and .artifacts.sm86.archive_sha256 != .artifacts.sm89.archive_sha256
  and .artifacts.sm86.binary_sha256 != .artifacts.sm89.binary_sha256
' "$test_root/provenance.json" >/dev/null
cp "$test_root/SHA256SUMS" "$test_root/SHA256SUMS.with-stale-provenance"
printf '%064d  mold-release-provenance.json\n' 0 \
  >>"$test_root/SHA256SUMS.with-stale-provenance"
if scripts/create-release-provenance.sh \
  "$test_root/should-not-exist.json" \
  v0.20.2 \
  0000000000000000000000000000000000000000 \
  "$test_root/SHA256SUMS.with-stale-provenance" \
  "$test_root" >/dev/null 2>&1; then
  fail "release provenance accepted a checksum manifest with stale provenance ordering"
fi

echo "cuda distribution contract: ok"
