[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ConfigPath,

  [Parameter(Mandatory = $true)]
  [string]$PublicCertificatePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

foreach ($name in @('WINDOWS_CERTIFICATE', 'WINDOWS_CERTIFICATE_PASSWORD')) {
  $item = Get-Item "Env:$name" -ErrorAction SilentlyContinue
  $value = if ($item) { $item.Value } else { '' }
  if ([string]::IsNullOrWhiteSpace($value)) {
    throw "required repository secret $name is missing"
  }
}

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$expected = $config.bundle.windows.certificateThumbprint
$pfx = Join-Path ([IO.Path]::GetTempPath()) "mold-windows-signing-$([guid]::NewGuid()).pfx"

try {
  [IO.File]::WriteAllBytes(
    $pfx,
    [Convert]::FromBase64String($env:WINDOWS_CERTIFICATE)
  )
  $password = ConvertTo-SecureString `
    -String $env:WINDOWS_CERTIFICATE_PASSWORD `
    -AsPlainText `
    -Force
  $imported = Import-PfxCertificate `
    -FilePath $pfx `
    -CertStoreLocation Cert:\CurrentUser\My `
    -Password $password `
    -Exportable:$false
  if (-not $imported -or $imported.Thumbprint -ne $expected) {
    throw "the imported certificate does not match configured thumbprint $expected"
  }
  Export-Certificate `
    -Cert $imported `
    -FilePath $PublicCertificatePath `
    -Type CERT | Out-Null
  Write-Output $imported.Thumbprint
}
finally {
  Remove-Item $pfx -Force -ErrorAction SilentlyContinue
}
