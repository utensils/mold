#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT
stub_bin="$test_root/bin"
mkdir -p "$stub_bin"

cat >"$stub_bin/uname" <<'EOF'
#!/bin/sh
case "$1" in
  -s) echo Linux ;;
  -m) echo x86_64 ;;
  *) exit 64 ;;
esac
EOF

cat >"$stub_bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  "-L") printf '%s\n' "${TEST_MIG_LISTING:-}" ;;
  *) printf '%s\n' "${TEST_INVENTORY:?}" ;;
esac
EOF

cat >"$stub_bin/curl" <<'EOF'
#!/bin/sh
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output|--write-out|--retry|--retry-delay) shift 2 ;;
    --*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s\n' "$url" >&2
printf '503'
EOF

chmod +x "$stub_bin/uname" "$stub_bin/nvidia-smi" "$stub_bin/curl"

assert_inventory() {
  local inventory="$1"
  local expected_arch="$2"
  local visible="${3-__unset__}"
  local mig_listing="${4-}"
  local output
  local -a env_args=(
    "PATH=$stub_bin:$PATH"
    "TEST_INVENTORY=$inventory"
    "TEST_MIG_LISTING=$mig_listing"
    "MOLD_VERSION=v-test"
    "MOLD_INSTALL_DIR=$test_root/install"
  )
  if [[ "$visible" != __unset__ ]]; then
    env_args+=("CUDA_VISIBLE_DEVICES=$visible")
  fi

  if output="$(
    env "${env_args[@]}" sh "$repo_root/install.sh" 2>&1
  )"; then
    echo "installer unexpectedly completed with the curl failure stub" >&2
    exit 1
  fi

  grep -Fq "cuda-${expected_arch}.tar.gz" <<<"$output" \
    || {
      echo "inventory did not select $expected_arch" >&2
      echo "$output" >&2
      exit 1
    }
}

assert_inventory_rejected() {
  local inventory="$1"
  local visible="${2-__unset__}"
  local expected_message="${3-Error:}"
  local mig_listing="${4-}"
  local output
  local -a env_args=(
    "PATH=$stub_bin:$PATH"
    "TEST_INVENTORY=$inventory"
    "TEST_MIG_LISTING=$mig_listing"
    "MOLD_VERSION=v-test"
    "MOLD_INSTALL_DIR=$test_root/install"
  )
  if [[ "$visible" != __unset__ ]]; then
    env_args+=("CUDA_VISIBLE_DEVICES=$visible")
  fi

  if output="$(
    env "${env_args[@]}" sh "$repo_root/install.sh" 2>&1
  )"; then
    echo "installer accepted invalid inventory/visibility input" >&2
    exit 1
  fi
  grep -Fq "$expected_message" <<<"$output" \
    || {
      echo "installer rejection omitted expected message: $expected_message" >&2
      echo "$output" >&2
      exit 1
    }
  if grep -Fq 'github.com/utensils/mold/releases/download/' <<<"$output"; then
    echo "installer attempted a release download after invalid GPU input" >&2
    exit 1
  fi
}

row() {
  printf '%s, GPU-%032d, %s' "$1" "$1" "$2"
}

assert_inventory "$(row 0 8.6)" sm86
assert_inventory "$(row 0 8.9)" sm89
assert_inventory "$(row 0 10.0)" sm100
assert_inventory "$(row 0 10.3)" sm100
assert_inventory "$(row 0 12.0)" sm120
assert_inventory "$(row 0 12.1)" sm120
assert_inventory "$(row 0 010.003)" sm100

# Keep the POSIX installer in exact fail-closed parity with the Rust updater:
# capabilities must be a complete numeric major.minor token.
for malformed_cap in \
  10.garbage 12.future 10. 10.0.0 .0 10x0 sm86 +10.0 10.+0 \
  4294967296.0 10.4294967296; do
  assert_inventory_rejected \
    "$(row 0 "$malformed_cap")" \
    __unset__ \
    "no published Mold CUDA archive is proven compatible"
done

# Every non-empty nvidia-smi row is validated before visibility filtering.
for malformed_inventory in \
  '0,GPU-only-two-columns' \
  '0,GPU-too-many,8.6,extra' \
  '0,,8.6' \
  '0,GPU-empty-cap,' \
  '0,GPU-valid,8.6
malformed-hidden-row'; do
  assert_inventory_rejected \
    "$malformed_inventory" \
    __unset__ \
    "malformed nvidia-smi inventory row"
done
assert_inventory_rejected \
  '0,GPU-valid,8.6
malformed-hidden-row' \
  0 \
  "malformed nvidia-smi inventory row"

if output="$(
  PATH="$stub_bin:$PATH" \
    TEST_INVENTORY="$(row 0 8.6)" \
    MOLD_VERSION=v-test \
    MOLD_CUDA_ARCH=sm89 \
    MOLD_INSTALL_DIR="$test_root/install" \
    sh "$repo_root/install.sh" 2>&1
)"; then
  echo "installer unexpectedly completed with the curl failure stub" >&2
  exit 1
fi
grep -Fq 'MOLD_CUDA_ARCH=sm89 is not compatible' <<<"$output" \
  || { echo "installer accepted sm89 on an sm86 fleet" >&2; exit 1; }

for unsupported_cap in 8.0 9.0; do
  if output="$(
    PATH="$stub_bin:$PATH" \
      TEST_INVENTORY="$(row 0 "$unsupported_cap")" \
      MOLD_VERSION=v-test \
      MOLD_INSTALL_DIR="$test_root/install" \
      sh "$repo_root/install.sh" 2>&1
  )"; then
    echo "installer accepted unsupported compute capability $unsupported_cap" >&2
    exit 1
  fi
  grep -Fq 'no published Mold CUDA archive is proven compatible' <<<"$output" \
    || { echo "installer did not fail closed for $unsupported_cap" >&2; exit 1; }
done

for hidden_visibility in '' '   ' -1 ' -1 '; do
  if output="$(
    PATH="$stub_bin:$PATH" \
      TEST_INVENTORY="$(row 0 8.6)" \
      CUDA_VISIBLE_DEVICES="$hidden_visibility" \
      MOLD_VERSION=v-test \
      MOLD_CUDA_ARCH=sm120 \
      MOLD_INSTALL_DIR="$test_root/install" \
      sh "$repo_root/install.sh" 2>&1
  )"; then
    echo "installer unexpectedly completed with the curl failure stub" >&2
    exit 1
  fi
  grep -Fq 'cuda-sm120.tar.gz' <<<"$output" \
    || {
      echo "explicit cross-machine install failed with all GPUs hidden" >&2
      exit 1
    }
done

mixed_pre_blackwell="$(row 0 8.6)
$(row 1 8.9)"
mixed_pre_blackwell_reversed="$(row 0 8.9)
$(row 1 8.6)"
assert_inventory "$mixed_pre_blackwell" sm86
assert_inventory "$mixed_pre_blackwell_reversed" sm86
assert_inventory "$mixed_pre_blackwell" sm86 0
assert_inventory "$mixed_pre_blackwell" sm89 1
assert_inventory "$mixed_pre_blackwell" sm86 "0,0"
assert_inventory "$mixed_pre_blackwell" sm86 " GPU-00000000000000000000000000000000 , 0 "

for empty_selector_list in ',' ',0' '0,' '0,,1'; do
  assert_inventory_rejected \
    "$mixed_pre_blackwell" \
    "$empty_selector_list" \
    "CUDA_VISIBLE_DEVICES contains an empty selector"
done
assert_inventory_rejected \
  "$mixed_pre_blackwell" \
  2 \
  "matched 0 CUDA devices"
assert_inventory_rejected \
  "$mixed_pre_blackwell" \
  GPU-unknown \
  "matched 0 CUDA devices"

mig_inventory="$(row 0 10.0)
$(row 1 10.0)"
mig_listing='GPU 0: NVIDIA B200 (UUID: GPU-00000000000000000000000000000000)
  MIG 1g.23gb Device 0: (UUID: MIG-11111111-1111-1111-1111-111111111111)
  MIG 1g.23gb Device 1: (UUID: MIG-12222222-2222-2222-2222-222222222222)
GPU 1: NVIDIA B200 (UUID: GPU-00000000000000000000000000000001)
  MIG 1g.23gb Device 0: (UUID: MIG-GPU-00000000000000000000000000000001/1/0)'
assert_inventory \
  "$mig_inventory" sm100 \
  MIG-GPU-00000000000000000000000000000001/1/0 "$mig_listing"
assert_inventory "$mig_inventory" sm100 \
  MIG-11111111-1111-1111-1111-111111111111 "$mig_listing"
assert_inventory "$mig_inventory" sm100 MIG-12 "$mig_listing"
assert_inventory_rejected "$mig_inventory" MIG-1 \
  "matched 2 CUDA devices" "$mig_listing"
assert_inventory_rejected "$mig_inventory" MIG-unknown \
  "matched 0 CUDA devices" "$mig_listing"

# Exercise the helper directly so duplicate selectors must collapse to one
# capability row, not merely happen to choose the same archive.
visible_helper="$(
  awk '
    /^visible_compute_caps\(\) \{/ { capture = 1 }
    capture { print }
    capture && /^}/ { exit }
  ' "$repo_root/install.sh"
)"
if grep -Fq 'printf '\''%s\n'\'' "${SELECTORS}" | while' "$repo_root/install.sh"; then
  echo "visible selector dedupe still relies on pipeline-subshell mutation" >&2
  exit 1
fi
eval "$visible_helper"
duplicate_physical_caps="$(
  CUDA_VISIBLE_DEVICES="0,GPU-00000000000000000000000000000000,0" \
    visible_compute_caps "$mig_inventory" "$mig_listing"
)"
[[ "$duplicate_physical_caps" == "10.0" ]] \
  || {
    echo "duplicate physical selectors were not collapsed: $duplicate_physical_caps" >&2
    exit 1
  }
duplicate_mig_parent_caps="$(
  CUDA_VISIBLE_DEVICES="MIG-11111111-1111-1111-1111-111111111111,0" \
    visible_compute_caps "$mig_inventory" "$mig_listing"
)"
[[ "$duplicate_mig_parent_caps" == "10.0" ]] \
  || {
    echo "duplicate MIG/physical parent selectors were not collapsed: $duplicate_mig_parent_caps" >&2
    exit 1
  }

homogeneous_64=""
for index in $(seq 0 63); do
  homogeneous_64="${homogeneous_64}${homogeneous_64:+
}$(row "$index" 10.0)"
done
assert_inventory "$homogeneous_64" sm100

for mixed in \
  "$(row 0 8.6)
$(row 1 10.0)" \
  "$(row 0 10.0)
$(row 1 8.6)" \
  "$(row 0 10.0)
$(row 1 12.0)"; do
  if output="$(
    PATH="$stub_bin:$PATH" \
      TEST_INVENTORY="$mixed" \
      MOLD_VERSION=v-test \
      MOLD_INSTALL_DIR="$test_root/install" \
      sh "$repo_root/install.sh" 2>&1
  )"; then
    echo "installer accepted an incompatible mixed fleet" >&2
    exit 1
  fi
  grep -Fq 'source build' <<<"$output" \
    || { echo "mixed-fleet error omitted source-build guidance" >&2; exit 1; }
done

if output="$(
  PATH="$stub_bin:$PATH" \
    TEST_INVENTORY="$(row 0 8.9)" \
    MOLD_VERSION=v-test \
    MOLD_CUDA_ARCH=sm999 \
    MOLD_INSTALL_DIR="$test_root/install" \
    sh "$repo_root/install.sh" 2>&1
)"; then
  echo "installer accepted an unsupported MOLD_CUDA_ARCH" >&2
  exit 1
fi
grep -Fq 'Expected one of: sm86, sm89, sm100, sm120' <<<"$output" \
  || { echo "installer did not explain the supported overrides" >&2; exit 1; }

fallback_root="$test_root/fallback"
mkdir -p "$fallback_root/bin" "$fallback_root/payload"
cp "$stub_bin/uname" "$stub_bin/nvidia-smi" "$fallback_root/bin/"
cat >"$fallback_root/payload/mold" <<'EOF'
#!/bin/sh
echo "fixture mold"
EOF
chmod +x "$fallback_root/payload/mold"
tar -czf "$fallback_root/mold.tar.gz" -C "$fallback_root/payload" mold
cat >"$fallback_root/bin/curl" <<'EOF'
#!/bin/sh
output=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output)
      output="$2"
      shift 2
      ;;
    --write-out|--retry|--retry-delay)
      shift 2
      ;;
    -*|--*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done
printf '%s\n' "$url" >>"${TEST_CURL_LOG:?}"
case "$url" in
  */mold-x86_64-unknown-linux-gnu-cuda-sm100.tar.gz)
    printf '404'
    ;;
  */mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz)
    if [ "${TEST_SM89_404:-0}" -eq 1 ]; then
      printf '404'
    else
      cp "${TEST_ARCHIVE:?}" "$output"
      printf '200'
    fi
    ;;
  */mold-x86_64-unknown-linux-gnu-cuda.tar.gz)
    cp "${TEST_ARCHIVE:?}" "$output"
    printf '200'
    ;;
  */SHA256SUMS)
    cp "${TEST_SUMS:?}" "$output"
    printf '200'
    ;;
  *)
    printf '500'
    ;;
esac
EOF
chmod +x "$fallback_root/bin/curl"
archive_sha="$(sha256sum "$fallback_root/mold.tar.gz" | awk '{print $1}')"
printf '%s  %s\n' \
  "$archive_sha" \
  "mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz" \
  >"$fallback_root/SHA256SUMS"

if PATH="$fallback_root/bin:$PATH" \
  TEST_INVENTORY="$(row 0 10.0)" \
  TEST_ARCHIVE="$fallback_root/mold.tar.gz" \
  TEST_SUMS="$fallback_root/SHA256SUMS" \
  TEST_CURL_LOG="$fallback_root/curl.log" \
  MOLD_VERSION=v-pre-phase-g \
  MOLD_INSTALL_DIR="$fallback_root/install" \
  sh "$repo_root/install.sh" >/dev/null 2>&1; then
  echo "installer substituted sm89 for a missing sm100 native artifact" >&2
  exit 1
fi

expected_urls="mold-x86_64-unknown-linux-gnu-cuda-sm100.tar.gz"
actual_urls="$(sed 's#^.*/##' "$fallback_root/curl.log")"
[[ "$actual_urls" == "$expected_urls" ]] \
  || {
    echo "installer did not fail closed after the native artifact 404" >&2
    printf 'expected:\n%s\nactual:\n%s\n' "$expected_urls" "$actual_urls" >&2
    exit 1
  }
if grep -Fv '/releases/download/v-pre-phase-g/' "$fallback_root/curl.log" | grep -q .; then
  echo "installer fallback escaped the pinned MOLD_VERSION" >&2
  exit 1
fi
if grep -Eq 'cuda-sm(89|120)\.tar\.gz|SHA256SUMS' "$fallback_root/curl.log"; then
  echo "installer attempted a compatibility substitute after native artifact 404" >&2
  exit 1
fi

: >"$fallback_root/curl.log"
printf '%s  %s\n' \
  "$archive_sha" \
  "mold-x86_64-unknown-linux-gnu-cuda.tar.gz" \
  >"$fallback_root/SHA256SUMS"
PATH="$fallback_root/bin:$PATH" \
  TEST_INVENTORY="$(row 0 8.9)" \
  TEST_ARCHIVE="$fallback_root/mold.tar.gz" \
  TEST_SUMS="$fallback_root/SHA256SUMS" \
  TEST_CURL_LOG="$fallback_root/curl.log" \
  TEST_SM89_404=1 \
  MOLD_VERSION=v-ancient \
  MOLD_INSTALL_DIR="$fallback_root/legacy-install" \
  sh "$repo_root/install.sh" >/dev/null
expected_urls="mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz
mold-x86_64-unknown-linux-gnu-cuda.tar.gz
SHA256SUMS"
actual_urls="$(sed 's#^.*/##' "$fallback_root/curl.log")"
[[ "$actual_urls" == "$expected_urls" ]] \
  || {
    echo "installer did not limit legacy fallback to the historical sm89 filename" >&2
    printf 'expected:\n%s\nactual:\n%s\n' "$expected_urls" "$actual_urls" >&2
    exit 1
  }

transient_root="$test_root/transient"
mkdir -p "$transient_root/bin"
cp "$stub_bin/uname" "$stub_bin/nvidia-smi" "$transient_root/bin/"
cat >"$transient_root/bin/curl" <<'EOF'
#!/bin/sh
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output|--write-out|--retry|--retry-delay) shift 2 ;;
    --*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s\n' "$url" >>"${TEST_CURL_LOG:?}"
exit 7
EOF
chmod +x "$transient_root/bin/curl"
if output="$(
  PATH="$transient_root/bin:$PATH" \
    TEST_INVENTORY="$(row 0 10.0)" \
    TEST_CURL_LOG="$transient_root/curl.log" \
    MOLD_VERSION=v-transient \
    MOLD_INSTALL_DIR="$transient_root/install" \
    sh "$repo_root/install.sh" 2>&1
)"; then
  echo "installer accepted a transient download failure" >&2
  exit 1
fi
[[ "$(wc -l <"$transient_root/curl.log")" -eq 1 ]] \
  || { echo "installer attempted fallback after a transient failure" >&2; exit 1; }
grep -Fq 'no compatibility fallback was attempted' <<<"$output" \
  || { echo "installer did not explain transient failure policy" >&2; exit 1; }

echo "install CUDA architecture selection: ok"
