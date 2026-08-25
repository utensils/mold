[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$verifier = Join-Path $PSScriptRoot '..\verify-windows-signatures.ps1'
$work = Join-Path ([IO.Path]::GetTempPath()) "mold-signature-test-$([guid]::NewGuid())"
$certificates = [Collections.Generic.List[Security.Cryptography.X509Certificates.X509Certificate2]]::new()

function New-TestCertificate {
  param([Parameter(Mandatory = $true)][string]$Name)

  $certificate = New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject "CN=$Name" `
    -CertStoreLocation Cert:\CurrentUser\My `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddDays(1)
  $certificates.Add($certificate)
  return $certificate
}

function New-SignedFixture {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)]$Certificate
  )

  Set-Content -Path $Path -Value "Write-Output 'signed fixture'" -Encoding utf8
  $result = Set-AuthenticodeSignature `
    -FilePath $Path `
    -Certificate $Certificate `
    -HashAlgorithm SHA256
  if (-not $result.SignerCertificate) {
    throw "failed to sign test fixture $Path`: $($result.Status)"
  }
}

function Assert-VerifierFails {
  param(
    [Parameter(Mandatory = $true)][scriptblock]$Action,
    [Parameter(Mandatory = $true)][string]$Case
  )

  try {
    & $Action
  }
  catch {
    Write-Output "rejected $Case"
    return
  }
  throw "signature verifier accepted $Case"
}

New-Item -ItemType Directory -Path $work | Out-Null

try {
  $expected = New-TestCertificate -Name 'Mold verifier expected signer'
  $other = New-TestCertificate -Name 'Mold verifier wrong signer'
  $publicCertificate = Join-Path $work 'expected.cer'
  Export-Certificate -Cert $expected -FilePath $publicCertificate -Type CERT | Out-Null

  $valid = Join-Path $work 'valid.ps1'
  New-SignedFixture -Path $valid -Certificate $expected
  & $verifier `
    -CertificatePath $publicCertificate `
    -ExpectedThumbprint $expected.Thumbprint `
    -FilePath $valid

  $tampered = Join-Path $work 'tampered.ps1'
  Copy-Item $valid $tampered
  Add-Content -Path $tampered -Value "Write-Output 'tampered'"
  Assert-VerifierFails -Case 'a tampered signed file' -Action {
    & $verifier `
      -CertificatePath $publicCertificate `
      -ExpectedThumbprint $expected.Thumbprint `
      -FilePath $tampered
  }

  $unsigned = Join-Path $work 'unsigned.ps1'
  Set-Content -Path $unsigned -Value "Write-Output 'unsigned'" -Encoding utf8
  Assert-VerifierFails -Case 'an unsigned file' -Action {
    & $verifier `
      -CertificatePath $publicCertificate `
      -ExpectedThumbprint $expected.Thumbprint `
      -FilePath $unsigned
  }

  $wrongSigner = Join-Path $work 'wrong-signer.ps1'
  New-SignedFixture -Path $wrongSigner -Certificate $other
  Assert-VerifierFails -Case 'a file signed by the wrong certificate' -Action {
    & $verifier `
      -CertificatePath $publicCertificate `
      -ExpectedThumbprint $expected.Thumbprint `
      -FilePath $wrongSigner
  }

  Write-Output 'Windows Authenticode verifier integration test passed'
}
finally {
  foreach ($certificate in $certificates) {
    Remove-Item "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -Force -ErrorAction SilentlyContinue
    $certificate.Dispose()
  }
  Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}
