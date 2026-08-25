#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workflow="$root/.github/workflows/desktop.yml"
release="$root/.github/workflows/release.yml"
config="$root/desktop/src-tauri/tauri.windows-self-signed.conf.json"
docs="$root/website/guide/desktop.md"
home="$root/website/index.md"
importer="$root/scripts/import-windows-signing-certificate.ps1"
verifier="$root/scripts/verify-windows-signatures.ps1"
verifier_test="$root/scripts/tests/windows-signature-verifier.ps1"

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
grep -Fq 'require_release_job_need "release-latest" "build-windows"' \
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
