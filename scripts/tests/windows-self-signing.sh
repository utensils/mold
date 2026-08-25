#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workflow="$root/.github/workflows/desktop.yml"
release="$root/.github/workflows/release.yml"
nightly="$root/.github/workflows/windows-nightly.yml"
config="$root/desktop/src-tauri/tauri.windows-self-signed.conf.json"
docs="$root/website/guide/desktop.md"
home="$root/website/index.md"
importer="$root/scripts/import-windows-signing-certificate.ps1"
verifier="$root/scripts/verify-windows-signatures.ps1"
verifier_test="$root/scripts/tests/windows-signature-verifier.ps1"

nightly_job_text() {
  local job="$1"
  awk -v job="$job" '
    $0 == "  " job ":" { in_job = 1 }
    in_job && /^  [[:alnum:]_-]+:$/ && $0 != "  " job ":" { exit }
    in_job { print }
  ' "$nightly"
}

thumbprint=$(jq -r '.bundle.windows.certificateThumbprint' "$config")
[[ "$thumbprint" =~ ^[A-F0-9]{40}$ ]] || {
  echo "self-signing thumbprint must be a 40-character uppercase SHA-1 hash" >&2
  exit 1
}

jq -e '.bundle.windows.digestAlgorithm == "sha256"' "$config" >/dev/null
jq -e '.bundle.windows.timestampUrl | startswith("http")' "$config" >/dev/null

# These are literal GitHub Actions expressions, not shell substitutions.
# shellcheck disable=SC2016
grep -Fq 'WINDOWS_CERTIFICATE: ${{ secrets.WINDOWS_CERTIFICATE }}' "$workflow"
# shellcheck disable=SC2016
grep -Fq 'WINDOWS_CERTIFICATE_PASSWORD: ${{ secrets.WINDOWS_CERTIFICATE_PASSWORD }}' "$workflow"
grep -Fq 'import-windows-signing-certificate.ps1' "$workflow"
grep -Fq 'tauri.windows-self-signed.conf.json' "$workflow"
grep -Fq 'verify-windows-signatures.ps1' "$workflow"
grep -Fq 'mold-desktop-windows-x64-self-signed' "$workflow"
grep -Fq 'build-windows:' "$release"
grep -Fq "if: startsWith(github.ref, 'refs/tags/v')" "$release"
grep -Fq 'reject_release_job_need "release-latest" "build-windows"' \
  "$root/scripts/tests/cuda-distribution-contract.sh"
grep -Fq 'require_release_job_need "release-native" "build-windows"' \
  "$root/scripts/tests/cuda-distribution-contract.sh"
grep -Fq 'mold-x86_64-pc-windows-msvc-cpu.zip' "$release"
grep -Fq 'Mold-windows-x64-self-signed.exe' "$release"
grep -Fq 'sign-windows-binary.ps1' "$release"
grep -Fq 'Import-PfxCertificate' "$importer"
grep -Fq 'Get-AuthenticodeSignature' "$verifier"
grep -Fq 'windows-signature-verifier.ps1' "$workflow"
grep -Fq 'New-SelfSignedCertificate' "$verifier_test"
grep -Fq 'group: windows-nightly' "$nightly"
grep -Fq 'cancel-in-progress: false' "$nightly"
grep -Fq 'workflow_dispatch:' "$nightly"
grep -Fq 'group: rolling-native-publication' "$nightly"
grep -Fq 'Mold-windows-x64-self-signed.exe' "$nightly"
grep -Fq 'mold-x86_64-pc-windows-msvc-cpu.zip' "$nightly"
grep -Fq 'mold-windows-nightly.json' "$nightly"
grep -Fq 'mold.windows-nightly.v1' "$nightly"
grep -Fq 'Re-sign the final standalone desktop executable' "$nightly"
grep -Fq 'Re-sign the final standalone desktop executable' "$workflow"
grep -Fq 'Re-sign the final standalone desktop executable' "$release"

build_job="$(nightly_job_text build-windows)"
publish_job="$(nightly_job_text publish-windows)"
[[ "$build_job" == *"if: github.ref == 'refs/heads/main'"* ]] || {
  echo "Windows Nightly must not expose signing secrets to non-main dispatches" >&2
  exit 1
}
[[ "$publish_job" == *"if: github.ref == 'refs/heads/main'"* ]] || {
  echo "Windows Nightly must not publish non-main dispatches" >&2
  exit 1
}
[[ "$publish_job" == *'needs: build-windows'* ]] || {
  echo "Windows Nightly publication must depend on its exact build" >&2
  exit 1
}
# shellcheck disable=SC2016
[[ "$publish_job" == *'git merge-base --is-ancestor "$GITHUB_SHA" FETCH_HEAD'* ]] || {
  echo "Windows Nightly must reject sources outside main history" >&2
  exit 1
}

ancestry_line="$(
  # shellcheck disable=SC2016
  grep -nF 'git merge-base --is-ancestor "$GITHUB_SHA" FETCH_HEAD' "$nightly" |
    cut -d: -f1
)"
release_create_line="$(
  grep -n -m1 'gh release create latest' "$nightly" |
    cut -d: -f1
)"
first_upload_line="$(
  grep -n -m1 'gh release upload latest' "$nightly" |
    cut -d: -f1
)"
checksum_upload_line="$(
  grep -n 'gh release upload latest artifacts/SHA256SUMS' "$nightly" |
    cut -d: -f1
)"
[[ "$ancestry_line" -lt "$release_create_line" \
  && "$ancestry_line" -lt "$first_upload_line" ]] || {
  echo "Windows Nightly ancestry guard must precede release mutation" >&2
  exit 1
}
[[ "$checksum_upload_line" -gt "$first_upload_line" ]] || {
  echo "Windows Nightly must publish SHA256SUMS after its Windows assets" >&2
  exit 1
}

# The verifier must never write a certificate store. Importing the self-signed
# certificate into Cert:\CurrentUser\Root to force a `Valid` status needs
# interactive trust confirmation: under -NonInteractive it fails outright, and
# GitHub runs pwsh WITHOUT -NonInteractive, so CryptoAPI instead raises a modal
# dialog on a desktop nobody can see and the step blocks until the job's
# six-hour ceiling. That stalled every Windows artifact, and because `publish`
# needs `build-windows`, the Linux, container, and AUR assets with them.
# Pin the thumbprint against the signature instead of trusting the chain.
if grep -Eq 'Import-Certificate|CertStoreLocation' "$verifier"; then
  echo "verify-windows-signatures.ps1 must not write a certificate store:" >&2
  echo "  importing into Cert:\CurrentUser\Root blocks on a trust dialog in CI" >&2
  exit 1
fi
grep -Fq 'ExpectedThumbprint' "$verifier"
grep -Fq 'CERT_E_UNTRUSTEDROOT' "$verifier"
grep -Fq 'TrustedPublisher' "$docs"
grep -Fq '/icons/windows.svg' "$home"
grep -Fq 'Windows CLI instructions' "$home"

echo "Windows self-signing contract is wired"
