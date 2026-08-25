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

# The signing certificate is self-signed, so no stock machine will ever chain it
# to a trusted root and Get-AuthenticodeSignature reports UnknownError rather
# than Valid. Do NOT import it into Cert:\CurrentUser\Root to force a Valid
# status: writing the Root store raises a CryptoAPI trust-confirmation dialog,
# and GitHub invokes pwsh WITHOUT -NonInteractive, so the call does not fail —
# it blocks on a window nobody can see until the job hits its six-hour ceiling.
# That stalled every Windows artifact, and `publish` needs `build-windows`, so
# the Linux, container, and AUR assets stalled with them.
#
# Trust is not what this check is for. It proves each artifact carries an intact
# Authenticode signature made by exactly the expected certificate.

$expected = $ExpectedThumbprint.Replace(' ', '').ToUpperInvariant()
if ($expected -notmatch '^[A-F0-9]{40}$') {
  throw "expected thumbprint is not a SHA-1 hash: $ExpectedThumbprint"
}

$pinned = [Security.Cryptography.X509Certificates.X509Certificate2]::new($CertificatePath)
if ($pinned.Thumbprint.ToUpperInvariant() -ne $expected) {
  throw "$CertificatePath does not hold the expected certificate $expected"
}

foreach ($path in $FilePath) {
  if (-not (Test-Path $path -PathType Leaf)) {
    throw "signed artifact is missing: $path"
  }

  $signature = Get-AuthenticodeSignature $path

  # PowerShell collapses every WinVerifyTrust HRESULT it does not recognize
  # into UnknownError. Accept only CERT_E_UNTRUSTEDROOT, which is the expected
  # result for our pinned self-signed certificate; timestamp, provider, and
  # every other unmapped failure must remain fatal. Constructing the message on
  # this machine keeps the comparison correct under localized Windows runners.
  $untrustedRootMessage = [ComponentModel.Win32Exception]::new(-2146762487).Message
  $hasExpectedUntrustedRoot =
    $signature.Status -eq 'UnknownError' -and
    $signature.StatusMessage -eq $untrustedRootMessage
  if ($signature.Status -ne 'Valid' -and -not $hasExpectedUntrustedRoot) {
    throw "$path has invalid Authenticode status $($signature.Status)"
  }
  if (-not $signature.SignerCertificate) {
    throw "$path carries no Authenticode signature"
  }
  if ($signature.SignerCertificate.Thumbprint.ToUpperInvariant() -ne $expected) {
    throw "$path was signed by an unexpected certificate"
  }

  # Everything except the untrusted root must still be sound: an expired or
  # malformed signer is a real failure.
  $chain = [Security.Cryptography.X509Certificates.X509Chain]::new()
  $chain.ChainPolicy.RevocationMode = 'NoCheck'
  $chain.ChainPolicy.VerificationFlags = 'AllowUnknownCertificateAuthority'
  if (-not $chain.Build($signature.SignerCertificate)) {
    $flags = ($chain.ChainStatus | ForEach-Object { $_.Status }) -join ', '
    throw "$path signer certificate failed chain validation: $flags"
  }

  Write-Output "verified Authenticode: $path ($($signature.Status))"
}
