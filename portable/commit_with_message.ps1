Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$message = Read-Host "Commit message (required)"
if ([string]::IsNullOrWhiteSpace($message)) {
    Write-Host "[ERROR] Commit message cannot be empty. No commit was created."
    exit 2
}

$saveScript = Join-Path $PSScriptRoot "save_version.ps1"
& $saveScript -Message $message
exit $LASTEXITCODE
