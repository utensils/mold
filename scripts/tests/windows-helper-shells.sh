#!/usr/bin/env bash
# `scripts/windows.ps1` must give the SAME answers under Windows PowerShell 5.1
# and pwsh 7. It is a `.ps1`, so a plain `.\scripts\windows.ps1` runs under
# whichever edition the user's shell is — and the two do not agree about this
# machine unless the script is careful.
#
# Measured on one ARM64 Surface, same moment, both editions:
#   PROCESSOR_ARCHITECTURE                    AMD64  / AMD64   (emulated shell)
#   RuntimeInformation::OSArchitecture        X64    / Arm64   (they disagree!)
#   Win32_Processor.Architecture              12     / 12      (ARM64, correct)
# and on some .NET Framework builds `OSArchitecture` is absent altogether,
# which under `Set-StrictMode` is a terminating error rather than a null.
#
# A wrong answer here is not cosmetic: `Test-IsX64Host` gates whether the
# `cuda` feature enters the build recipe, so an ARM64 laptop reporting x64
# would try to build a CUDA desktop app.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/windows.ps1"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

if [[ "${OS:-}" != "Windows_NT" ]]; then
  echo "skip: windows-only helper test"
  exit 0
fi

[[ -f "$script" ]] || fail "missing $script"

# Parse cleanly under whichever editions exist. `powershell.exe` is Windows
# PowerShell 5.1 and is what a bare `.\scripts\windows.ps1` gets by default;
# `pwsh` is PowerShell 7.
shells=()
if command -v powershell.exe >/dev/null 2>&1; then shells+=("powershell.exe"); fi
if command -v pwsh >/dev/null 2>&1; then shells+=("pwsh"); fi
[[ ${#shells[@]} -gt 0 ]] || fail "no PowerShell edition found"

# shellcheck disable=SC2016  # PowerShell's own $variables must reach it unexpanded.
parse_check='$e = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile($env:MOLD_PS1, [ref]$null, [ref]$e); if ($e) { $e | ForEach-Object { $_.Message }; exit 1 }; "PARSE OK"'

declare -a reported_arch=()

for shell in "${shells[@]}"; do
  MOLD_PS1="$script" "$shell" -NoProfile -Command "$parse_check" >/dev/null \
    || fail "$shell could not parse windows.ps1"

  # `doctor` exercises every probe, including the architecture and toolchain
  # ones. It must exit 0 on a machine whose toolchain is complete, and must
  # never throw — a PropertyNotFoundStrict here is the regression this guards.
  output="$("$shell" -NoProfile -File "$script" doctor 2>&1)" || {
    echo "$output" >&2
    fail "$shell: doctor exited non-zero"
  }

  if ! grep -qi 'Mold Windows toolchain' <<<"$output"; then
    fail "$shell: doctor produced no banner"
  fi
  # Exiting 0 is not enough on its own: a guarded probe can swallow the error
  # and still print a banner, and the reported crash was exactly a strict-mode
  # property miss.
  if grep -qiE 'cannot be found|PropertyNotFound|Unable to find type' <<<"$output"; then
    echo "$output" >&2
    fail "$shell: doctor reported a property/type resolution error"
  fi

  arch="$(sed -n 's/.*Mold Windows toolchain (\([^)]*\)).*/\1/p' <<<"$output" | head -1)"
  [[ -n "$arch" ]] || fail "$shell: doctor did not report an architecture"
  [[ "$arch" != "unknown" ]] || fail "$shell: architecture resolved to 'unknown'"
  reported_arch+=("$arch")

  # `features` is the machine-readable recipe every caller splats.
  "$shell" -NoProfile -File "$script" features >/dev/null \
    || fail "$shell: features exited non-zero"
done

# The point of the test: both editions describe the same machine.
if [[ ${#reported_arch[@]} -gt 1 ]]; then
  for arch in "${reported_arch[@]}"; do
    [[ "$arch" == "${reported_arch[0]}" ]] || fail \
      "PowerShell editions disagree about this machine: ${reported_arch[*]}"
  done
fi

echo "ok: windows.ps1 agrees across ${#shells[@]} PowerShell edition(s) (${reported_arch[0]})"
