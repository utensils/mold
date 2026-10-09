#!/usr/bin/env bash
# shellcheck disable=SC2016 # Literal PKGBUILD/workflow source is asserted below.
# The Linux desktop distribution: the `mold-ai-desktop` (source, CUDA) and
# `mold-ai-desktop-bin` (prebuilt, GPU-free) AUR packages, the shared desktop
# file layout, and the tagged release job that produces the -bin payload.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
release=".github/workflows/release.yml"
source_recipe="packaging/aur/mold-ai-desktop/PKGBUILD"
bin_recipe="packaging/aur/mold-ai-desktop-bin/PKGBUILD"
archive="mold-desktop-x86_64-unknown-linux-gnu-cpu.tar.gz"

fail() {
  echo "aur desktop contract: $*" >&2
  exit 1
}

require_text() {
  grep -Fq -- "$2" "$repo_root/$1" || fail "$1 is missing: $2"
}

reject_text() {
  if grep -Fq -- "$2" "$repo_root/$1"; then fail "$1 must not contain: $2"; fi
}

# The body of one PKGBUILD function, `name() {` through its closing `}`.
pkgbuild_function_body() {
  awk -v header="$2() {" '
    $0 == header { in_function = 1; next }
    in_function && $0 == "}" { exit }
    in_function { print }
  ' "$repo_root/$1"
}

# One PKGBUILD array (`name=(...)`), single- or multi-line, as one line.
pkgbuild_array() {
  awk -v header="$2=(" '
    index($0, header) == 1 { in_array = 1 }
    in_array { printf "%s ", $0 }
    in_array && /\)[[:space:]]*$/ { exit }
  ' "$repo_root/$1"
}

release_job_text() {
  awk -v job="  $1:" '
      $0 == job { in_job = 1 }
      in_job && /^  [[:alnum:]_-]+:$/ && $0 != job { exit }
      in_job { print }
    ' "$repo_root/$release"
}

release_job_needs() {
  release_job_text "$1" | awk '
      /^    needs:/ { in_needs = 1 }
      in_needs && /^    [[:alnum:]_-]+:/ && $0 !~ /^    needs:/ { exit }
      in_needs { print }
    '
}

# Both desktop packages install /usr/bin/mold-desktop and nothing under the
# CLI's paths: they conflict with each other only, never with `mold-ai*` or
# extra/mold (the rui314 linker).
conflicts="$(pkgbuild_array "$source_recipe" conflicts)"
grep -Fq "'mold-ai-desktop-bin'" <<<"$conflicts" || fail "$source_recipe must conflict with mold-ai-desktop-bin"
conflicts="$(pkgbuild_array "$bin_recipe" conflicts)"
grep -Fq "'mold-ai-desktop'" <<<"$conflicts" || fail "$bin_recipe must conflict with mold-ai-desktop"
for recipe in "$source_recipe" "$bin_recipe"; do
  if grep -Eq "'mold(-ai(-bin|-git)?)?'" <<<"$(pkgbuild_array "$recipe" conflicts)"; then
    fail "$recipe conflicts with a CLI package; the desktop app installs alongside it"
  fi
  package_body="$(pkgbuild_function_body "$recipe" package)"
  [[ -n "$package_body" ]] || fail "$recipe has no package() function"
  # package() runs under fakeroot with the builder's loader state (#1742).
  if grep -Eq '(^|[[:space:]/])mold-desktop([[:space:]]|$)|\$\{_binname\}"?[[:space:]]+-' <<<"$package_body"; then
    fail "$recipe executes the desktop binary inside package()"
  fi
done

# The source recipe: the same device feature rule as `mold-ai` (asserted by
# cuda-distribution-contract.sh), the desktop crate's own cargo root, the
# frontend built before cargo embeds it, and the one shared install layout.
require_text "$source_recipe" 'cargo fetch --locked --manifest-path "${_manifest}"'
require_text "$source_recipe" '_manifest=desktop/src-tauri/Cargo.toml'
require_text "$source_recipe" 'TAURI_ENV_PLATFORM=linux bun run build:desktop'
require_text "$source_recipe" '--features "tauri/custom-protocol,${gpu_feature},cudnn,pulid,webp,mesh-texture,mesh-matting,mesh-delight"'
require_text "$source_recipe" 'options=(!lto)'
require_text "$source_recipe" './scripts/install-linux-desktop-files.sh'
require_text "$source_recipe" './scripts/verify-h3-release-exclusion.sh'
# B200/B300 is server-only (flake.nix exports no mold-desktop-sm100).
require_text "$source_recipe" 'if [[ "${CUDA_COMPUTE_CAP}" == "100" ]]; then'

# The -bin recipe is GPU-free: no CUDA runtime, no NVIDIA driver.
for array in depends optdepends makedepends; do
  if grep -Ei 'cuda|cudnn|nvidia' <<<"$(pkgbuild_array "$bin_recipe" "$array")"; then
    fail "$bin_recipe $array must not name CUDA or NVIDIA packages"
  fi
done
require_text "$bin_recipe" '_binname=mold-desktop'
require_text "$bin_recipe" '${_binname}-x86_64-unknown-linux-gnu-cpu.tar.gz'
require_text "$bin_recipe" 'provides=("${_pkgname}=${pkgver}")'

# The .desktop entry, metainfo, and icons agree with the install script.
desktop_file="packaging/linux/com.utensils.mold.desktop"
require_text "$desktop_file" 'Exec=mold-desktop %U'
require_text "$desktop_file" 'Icon=com.utensils.mold'
require_text "$desktop_file" 'StartupWMClass=mold-desktop'
require_text "packaging/linux/com.utensils.mold.metainfo.xml" '<id>com.utensils.mold</id>'
require_text "packaging/linux/com.utensils.mold.metainfo.xml" \
  '<launchable type="desktop-id">com.utensils.mold.desktop</launchable>'
require_text "desktop/src-tauri/tauri.conf.json" '"identifier": "com.utensils.mold"'
require_text "scripts/install-linux-desktop-files.sh" 'app_id="com.utensils.mold"'
require_text "scripts/install-linux-desktop-files.sh" '"$prefix/bin/mold-desktop"'
require_text "scripts/package-desktop-linux-archive.sh" 'scripts/install-linux-desktop-files.sh'

# The release job: GPU-free features, verified, packaged through the shared
# layout, proven as an Arch package, and published only on stable tags.
job="$(release_job_text build-linux-desktop-cpu)"
[[ -n "$job" ]] || fail "$release is missing job build-linux-desktop-cpu"
for text in \
  "if: startsWith(github.ref, 'refs/tags/v')" \
  '--features tauri/custom-protocol,pulid,webp' \
  'scripts/verify-h3-release-exclusion.sh' \
  'scripts/package-desktop-linux-archive.sh' \
  "scripts/aur/test-in-docker.sh --archive \"\$PWD/$archive\" mold-ai-desktop-bin" \
  "path: $archive"; do
  grep -Fq -- "$text" <<<"$job" || fail "$release job build-linux-desktop-cpu is missing: $text"
done
if grep -Eq -- '--features[= ][^[:space:]]*(cuda|cudnn|flash-attn|h3)' <<<"$job"; then
  fail "build-linux-desktop-cpu must not compile a GPU feature"
fi
grep -Eq '(^|[][,[:space:]])build-linux-desktop-cpu([],[:space:]]|$)' <<<"$(release_job_needs release-native)" \
  || fail "release-native must wait for build-linux-desktop-cpu"
grep -Fq "artifacts/$archive" <<<"$(release_job_text release-native)" \
  || fail "release-native must upload $archive"
if grep -Fq build-linux-desktop-cpu <<<"$(release_job_needs release-latest)"; then
  fail "the rolling release must not wait for the tag-only desktop job"
fi
publish_aur="$(release_job_text publish-aur)"
for pkg in mold-ai-desktop mold-ai-desktop-bin; do
  grep -Eq "pkgname: \[.*(^|[[:space:],\[])${pkg}([],[:space:]]|$)" <<<"$publish_aur" \
    || fail "publish-aur matrix is missing $pkg"
done

# The AUR tooling knows both packages.
require_text "scripts/aur/update-pkgbuild.sh" "$archive"
require_text "scripts/aur/update-pkgbuild.sh" 'mold-ai|mold-ai-desktop)'
require_text "scripts/aur/test-in-docker.sh" 'mold-ai-bin|mold-ai|mold-ai-git|mold-ai-desktop|mold-ai-desktop-bin)'
require_text "scripts/aur/test-in-docker.sh" 'desktop-file-validate /usr/share/applications/com.utensils.mold.desktop'
require_text "packaging/aur/test/Dockerfile" 'desktop-file-utils'
for doc in packaging/aur/README.md website/guide/installation.md website/guide/desktop.md; do
  require_text "$doc" 'mold-ai-desktop-bin'
done

echo "aur desktop contract: ok"
