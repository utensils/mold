[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ConfigPath,

  [Parameter(Mandatory = $true)]
  [string]$FilePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$windows = $config.bundle.windows
$kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
$signTool = Get-ChildItem $kits -Filter signtool.exe -Recurse -ErrorAction Stop |
  Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
  Sort-Object FullName -Descending |
  Select-Object -First 1
if (-not $signTool) {
  throw "signtool.exe was not found under $kits"
}

& $signTool.FullName sign `
  /fd $windows.digestAlgorithm `
  /tr $windows.timestampUrl `
  /td SHA256 `
  /sha1 $windows.certificateThumbprint `
  $FilePath
if ($LASTEXITCODE -ne 0) {
  throw "signtool failed for $FilePath with exit code $LASTEXITCODE"
}
