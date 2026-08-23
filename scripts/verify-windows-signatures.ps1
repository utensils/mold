[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$CertificatePath,

  [Parameter(Mandatory = $true)]
  [string]$ExpectedThumbprint,

  [Parameter(Mandatory = $true)]
  [string[]]$FilePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Import-Certificate `
  -FilePath $CertificatePath `
  -CertStoreLocation Cert:\CurrentUser\Root | Out-Null

foreach ($path in $FilePath) {
  if (-not (Test-Path $path -PathType Leaf)) {
    throw "signed artifact is missing: $path"
  }
  $signature = Get-AuthenticodeSignature $path
  if ($signature.Status -ne 'Valid') {
    throw "$path has invalid Authenticode status $($signature.Status)"
  }
  if ($signature.SignerCertificate.Thumbprint -ne $ExpectedThumbprint) {
    throw "$path was signed by an unexpected certificate"
  }
  Write-Output "verified Authenticode: $path"
}
