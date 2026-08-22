#!/usr/bin/env pwsh
# Mold on Windows — the peer of scripts/ios.sh and scripts/android.sh.
#
# The Nix devshell (`desktop-dev`, `desktop-build`, `desktop-check`,
# `desktop-test`, `desktop-ui`) is the macOS/Linux entry point and does not run
# on Windows; this script is that same command set with the Windows toolchain
# resolved natively.
#
#   scripts\windows.ps1 doctor   # verify the toolchain, name what is missing
#   scripts\windows.ps1 setup    # install what `doctor` can install
#   scripts\windows.ps1 dev      # Tauri dev with hot reload (Vite on :1430)
#   scripts\windows.ps1 ui       # frontend-only Vite server
#   scripts\windows.ps1 check    # rustfmt + clippy -D warnings + vue-tsc + prettier
#   scripts\windows.ps1 test     # cargo test (CPU) + vitest
#   scripts\windows.ps1 build    # NSIS installer + the standalone Mold.exe
#   scripts\windows.ps1 clean    # drop the desktop build outputs
#
# Feature recipe: CPU-only by default, because the reference Windows dev box is
# an ARM64 Surface with no NVIDIA GPU. `cuda` is added when a CUDA toolkit is
# present on an x64 host, and `pulid` when protoc is on PATH (candle-onnx's
# prost-build needs it). MOLD_WINDOWS_FEATURES overrides the whole recipe;
# MOLD_WINDOWS_CUDA=1 / MOLD_WINDOWS_NO_PULID=1 move one axis each.

[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('doctor', 'setup', 'dev', 'ui', 'check', 'test', 'build', 'bundle', 'clean', 'features')]
  [string]$Action = 'dev',

  [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
  [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Desktop = Join-Path $Root 'desktop'
$Manifest = Join-Path $Desktop 'src-tauri\Cargo.toml'
$MSRV = [version]'1.93'
$VitePort = 1430

function Write-Step { param([string]$Text) Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Good { param([string]$Text) Write-Host "  ok   $Text" -ForegroundColor Green }
function Write-Note { param([string]$Text) Write-Host "  warn $Text" -ForegroundColor Yellow }
function Write-Miss { param([string]$Text) Write-Host "  MISS $Text" -ForegroundColor Red }

function Test-Tool { param([string]$Name) return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# The machine's architecture, asked of the MACHINE rather than of this process.
#
# Every convenient source lies on an ARM64 Windows box, because the shell is
# usually x64-emulated. Measured on one Surface, same machine, same moment:
# `PROCESSOR_ARCHITECTURE` says AMD64 in both shells;
# `RuntimeInformation::OSArchitecture` says Arm64 under pwsh 7 and **X64**
# under Windows PowerShell 5.1; `Win32_Processor.Architecture` says 12 (ARM64)
# in both. On some .NET Framework builds that property is missing entirely,
# which under `Set-StrictMode` is a terminating PropertyNotFoundStrict rather
# than a null — the crash that sent us looking. So: WMI first, and every
# fallback guarded, because a wrong answer here is worse than no answer.
function Get-OSArch {
  try {
    $processor = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
    switch ([int]$processor.Architecture) {
      0 { return 'x86' }
      5 { return 'arm' }
      9 { return 'x64' }
      12 { return 'arm64' }
    }
  }
  catch { }
  try {
    $reported = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($reported) { return $reported.ToString().ToLowerInvariant() }
  }
  catch { }
  # Set only when a 32-bit process runs under WOW64, but correct when present.
  if ($env:PROCESSOR_ARCHITEW6432) { return $env:PROCESSOR_ARCHITEW6432.ToLowerInvariant() }
  if ($env:PROCESSOR_ARCHITECTURE) { return $env:PROCESSOR_ARCHITECTURE.ToLowerInvariant() }
  return 'unknown'
}

function Get-RustHostTriple {
  if (-not (Test-Tool 'rustup')) { return $null }
  $shown = & rustup show 2>$null | Select-String 'Default host:\s*(\S+)' | Select-Object -First 1
  if ($shown) { return $shown.Matches[0].Groups[1].Value }
  return $null
}

# The real question behind the CUDA decision is "will cargo produce an x86_64
# binary?", not "what CPU is this". Ask the Rust host triple, which answers it
# directly and — unlike every architecture probe above — reads the same in
# Windows PowerShell and pwsh. OS architecture is only the fallback for a
# machine with no rustup, where nothing is going to be built anyway.
function Test-IsX64Host {
  $triple = Get-RustHostTriple
  if ($triple) { return $triple.StartsWith('x86_64') }
  return (Get-OSArch) -eq 'x64'
}

function Get-VsInstallPath {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return $null }
  $found = & $vswhere -latest -products * -property installationPath 2>$null
  if ($found) { return ($found | Select-Object -First 1) }
  return $null
}

function Get-WebView2Version {
  $keys = @(
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
    'HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
    'HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
  )
  foreach ($key in $keys) {
    try {
      $pv = (Get-ItemProperty -Path $key -Name pv -ErrorAction Stop).pv
      if ($pv -and $pv -ne '0.0.0.0') { return $pv }
    }
    catch { }
  }
  return $null
}

function Get-RustcVersion {
  if (-not (Test-Tool 'rustc')) { return $null }
  $line = & rustc --version 2>$null
  if ($line -match 'rustc (\d+\.\d+\.\d+)') { return [version]$matches[1] }
  return $null
}

function Test-HasCuda {
  if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) { return $true }
  return (Test-Tool 'nvcc')
}

function Resolve-Features {
  if ($env:MOLD_WINDOWS_FEATURES) { return $env:MOLD_WINDOWS_FEATURES }
  $features = @()
  $wantCuda = ($env:MOLD_WINDOWS_CUDA -eq '1') -or ((Test-IsX64Host) -and (Test-HasCuda))
  if ($wantCuda) { $features += 'cuda' }
  if ($env:MOLD_WINDOWS_NO_PULID -ne '1' -and (Test-Tool 'protoc')) { $features += 'pulid' }
  return ($features -join ',')
}

# `cargo tauri` rejects an empty --features value, so callers splat this array.
function Get-FeatureArgs {
  $features = Resolve-Features
  if ([string]::IsNullOrWhiteSpace($features)) { return @() }
  return @('--features', $features)
}

function Assert-Prereqs {
  $missing = @()
  if (-not (Test-Tool 'cargo')) { $missing += 'cargo (https://rustup.rs)' }
  if (-not (Test-Tool 'bun')) { $missing += 'bun (winget install Oven-sh.Bun)' }
  if (-not (Get-VsInstallPath)) { $missing += 'Visual Studio Build Tools with the C++ workload' }
  if (-not (Get-WebView2Version)) { $missing += 'Microsoft Edge WebView2 Runtime' }
  if ($missing.Count -gt 0) {
    Write-Host 'Windows toolchain is incomplete:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host 'Run: scripts\windows.ps1 doctor' -ForegroundColor Yellow
    exit 1
  }
  $rustc = Get-RustcVersion
  if ($rustc -and $rustc -lt $MSRV) {
    Write-Host "rustc $rustc is below the workspace MSRV $MSRV - run: rustup update stable" -ForegroundColor Red
    exit 1
  }
}

function Install-TauriCli {
  if (Test-Tool 'cargo-tauri') { return }
  Write-Step 'installing tauri-cli'
  & cargo install tauri-cli --locked
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Killing a previous `dev` mid-build orphans its Vite child, which keeps :1430
# bound and fails the next run with "Port 1430 is already in use". This is the
# Windows half of the devshell's lsof reaping.
function Clear-VitePort {
  $owners = @()
  try {
    $owners = @(Get-NetTCPConnection -LocalPort $VitePort -State Listen -ErrorAction Stop |
      Select-Object -ExpandProperty OwningProcess -Unique)
  }
  catch { return }
  foreach ($owner in $owners) {
    if ($owner -le 4) { continue }
    Write-Note "killing stale listener on :$VitePort (pid $owner)"
    Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
  }
  if ($owners.Count -gt 0) { Start-Sleep -Milliseconds 500 }
}

function Invoke-Checked {
  param([string]$Exe, [string[]]$Arguments, [string]$WorkingDirectory = $Root)
  Push-Location $WorkingDirectory
  try {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  finally { Pop-Location }
}

function Invoke-BunInstall {
  Invoke-Checked 'bun' @('install', '--frozen-lockfile')
}

function Invoke-Doctor {
  Write-Step "Mold Windows toolchain ($(Get-OSArch))"
  $fatal = 0

  $vs = Get-VsInstallPath
  if ($vs) { Write-Good "MSVC build tools: $vs" }
  else {
    Write-Miss 'MSVC build tools not found'
    Write-Host '       winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    $fatal++
  }

  $rustc = Get-RustcVersion
  if (-not $rustc) { Write-Miss 'rustc not found - install from https://rustup.rs'; $fatal++ }
  elseif ($rustc -lt $MSRV) { Write-Miss "rustc $rustc is below MSRV $MSRV - run: rustup update stable"; $fatal++ }
  else { Write-Good "rustc $rustc (MSRV $MSRV)" }

  $triple = Get-RustHostTriple
  if ($triple) { Write-Good "rust host: $triple" }
  else { Write-Note 'rustup not found - cannot confirm the build target triple' }

  $wv2 = Get-WebView2Version
  if ($wv2) { Write-Good "WebView2 runtime $wv2" }
  else {
    Write-Miss 'WebView2 runtime not found'
    Write-Host '       winget install Microsoft.EdgeWebView2Runtime'
    $fatal++
  }

  if (Test-Tool 'bun') { Write-Good "bun $(& bun --version)" }
  else { Write-Miss 'bun not found - winget install Oven-sh.Bun'; $fatal++ }

  if (Test-Tool 'cargo-tauri') { Write-Good "$(((& cargo tauri --version) -join ' ').Trim())" }
  else { Write-Note 'tauri-cli not installed - scripts\windows.ps1 setup will add it' }

  if (Test-Tool 'protoc') { Write-Good "protoc $((& protoc --version) -replace 'libprotoc ', '') (pulid available)" }
  else { Write-Note 'protoc not found - face identity (pulid) stays off; winget install Google.Protobuf' }

  if (Test-HasCuda) {
    if (Test-IsX64Host) { Write-Good 'CUDA toolkit found' }
    else { Write-Note 'CUDA toolkit found but this is an ARM64 host - CUDA stays off' }
  }
  else {
    Write-Note 'no CUDA toolkit - the embedded engine runs on CPU'
  }

  # AAC muxing is off on Windows: fdk-aac-sys does not compile under MSVC
  # (FDK_archdef.h falls through to a #warning, which is fatal error C1021).
  Write-Note 'AAC audio muxing (mp4 feature) is unavailable on Windows - video still renders'

  $long = $null
  try {
    $long = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
  }
  catch { }
  if ($long -eq 1) { Write-Good 'long paths enabled' }
  else { Write-Note 'long paths disabled - deep cargo target paths can fail; set LongPathsEnabled=1' }

  Write-Host ''
  $features = Resolve-Features
  if ($features) { Write-Host "feature recipe: $features" -ForegroundColor Magenta }
  else { Write-Host 'feature recipe: (none - CPU only)' -ForegroundColor Magenta }

  if ($fatal -gt 0) {
    Write-Host ''
    Write-Host "$fatal blocking item(s)." -ForegroundColor Red
    exit 1
  }
  Write-Host ''
  Write-Host 'Toolchain looks good.' -ForegroundColor Green
}

function Invoke-Setup {
  Write-Step 'installing what is missing'
  if (-not (Test-Tool 'winget')) { Write-Note 'winget unavailable - install prerequisites manually' }
  else {
    if (-not (Get-WebView2Version)) { & winget install --id Microsoft.EdgeWebView2Runtime -e --accept-source-agreements --accept-package-agreements }
    if (-not (Test-Tool 'bun')) { & winget install --id Oven-sh.Bun -e --accept-source-agreements --accept-package-agreements }
    if (-not (Test-Tool 'protoc')) { & winget install --id Google.Protobuf -e --accept-source-agreements --accept-package-agreements }
  }
  Install-TauriCli
  Invoke-BunInstall
  Invoke-Doctor
}

switch ($Action) {
  'doctor' { Invoke-Doctor }

  'setup' { Invoke-Setup }

  'features' { Resolve-Features }

  'dev' {
    Assert-Prereqs
    Install-TauriCli
    Clear-VitePort
    Invoke-BunInstall
    $call = @('tauri', 'dev') + (Get-FeatureArgs) + $Rest
    Write-Step "cargo $($call -join ' ')"
    Invoke-Checked 'cargo' $call $Desktop
  }

  'ui' {
    Invoke-BunInstall
    Clear-VitePort
    Invoke-Checked 'bun' (@('run', 'dev') + $Rest) $Desktop
  }

  'check' {
    Assert-Prereqs
    Invoke-Checked 'cargo' @('fmt', '--manifest-path', $Manifest, '--', '--check')
    Invoke-Checked 'cargo' @('clippy', '--manifest-path', $Manifest, '--all-targets', '--', '-D', 'warnings')
    if (Test-Tool 'protoc') {
      Invoke-Checked 'cargo' @('clippy', '--manifest-path', $Manifest, '--all-targets', '--features', 'pulid', '--', '-D', 'warnings')
    }
    else {
      Write-Note 'skipping the pulid clippy pass - protoc is not installed'
    }
    Invoke-BunInstall
    Invoke-Checked 'bunx' @('vue-tsc', '-b') $Desktop
    Invoke-Checked 'bun' @('run', 'fmt:check') $Desktop
  }

  'test' {
    Assert-Prereqs
    Invoke-Checked 'cargo' (@('test', '--manifest-path', $Manifest) + $Rest)
    Invoke-BunInstall
    Invoke-Checked 'bun' @('run', 'test') $Desktop
  }

  { $_ -in 'build', 'bundle' } {
    Assert-Prereqs
    Install-TauriCli
    Invoke-BunInstall
    $call = @('tauri', 'build') + (Get-FeatureArgs) + @('--bundles', 'nsis') + $Rest
    Write-Step "cargo $($call -join ' ')"
    Invoke-Checked 'cargo' $call $Desktop
    $bundle = Join-Path $Desktop 'src-tauri\target\release\bundle\nsis'
    if (Test-Path $bundle) {
      Write-Host ''
      Get-ChildItem $bundle -Filter *.exe | ForEach-Object { Write-Good $_.FullName }
    }
  }

  'clean' {
    Write-Step 'removing desktop build outputs'
    foreach ($path in @((Join-Path $Desktop 'src-tauri\target'), (Join-Path $Desktop 'dist'))) {
      if (Test-Path $path) { Remove-Item -Recurse -Force $path; Write-Good "removed $path" }
    }
  }
}
