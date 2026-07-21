Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$toolsRoot = Join-Path $root "TOOLS"
$runtimeRoot = Join-Path $root ".runtime"
$downloads = Join-Path $runtimeRoot "git-downloads"
$gitTarget = Join-Path $toolsRoot "Git"
$ghTarget = Join-Path $toolsRoot "GitHubCLI"

$gitVersion = "2.55.0.3"
$gitTag = "v2.55.0.windows.3"
$gitArchiveName = "MinGit-$gitVersion-64-bit.zip"
$gitUrl = "https://github.com/git-for-windows/git/releases/download/$gitTag/$gitArchiveName"
$gitSha256 = "F48E2D2DC74A24454ADC6D8FD0AC25BF9C2386F19CFB06202B9465AAAD4F9F05"

$ghVersion = "2.96.0"
$ghArchiveName = "gh_${ghVersion}_windows_amd64.zip"
$ghUrl = "https://github.com/cli/cli/releases/download/v$ghVersion/$ghArchiveName"
$ghChecksumsUrl = "https://github.com/cli/cli/releases/download/v$ghVersion/gh_${ghVersion}_checksums.txt"

New-Item -ItemType Directory -Path $toolsRoot, $downloads -Force | Out-Null

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $existingHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($existingHash -eq $ExpectedSha256) {
            Write-Host "[OK] Cached download verified: $([IO.Path]::GetFileName($Destination))"
            return
        }
    }
    $partial = "$Destination.partial"
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    Write-Host "Downloading $Url"
    & curl.exe --fail --location --retry 3 --retry-delay 2 --output $partial $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actual -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "SHA-256 verification failed for $Url"
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

$existingGit = Join-Path $gitTarget "cmd\git.exe"
if (Test-Path -LiteralPath $existingGit -PathType Leaf) {
    Write-Host "[OK] Portable Git already exists."
}
else {
    $gitArchive = Join-Path $downloads $gitArchiveName
    Get-VerifiedDownload -Url $gitUrl -Destination $gitArchive -ExpectedSha256 $gitSha256
    $gitStage = Join-Path $runtimeRoot "git-install-stage"
    if (Test-Path -LiteralPath $gitStage) { Remove-Item -LiteralPath $gitStage -Recurse -Force }
    New-Item -ItemType Directory -Path $gitStage -Force | Out-Null
    Expand-Archive -LiteralPath $gitArchive -DestinationPath $gitStage -Force
    if (-not (Test-Path -LiteralPath (Join-Path $gitStage "cmd\git.exe") -PathType Leaf)) {
        throw "The MinGit archive did not contain cmd\git.exe."
    }
    if (Test-Path -LiteralPath $gitTarget) { Remove-Item -LiteralPath $gitTarget -Recurse -Force }
    Move-Item -LiteralPath $gitStage -Destination $gitTarget
}

$ghArchive = Join-Path $downloads $ghArchiveName
$ghChecksums = Join-Path $downloads "gh_${ghVersion}_checksums.txt"
$existingGh = Join-Path $ghTarget "bin\gh.exe"
if (Test-Path -LiteralPath $existingGh -PathType Leaf) {
    Write-Host "[OK] Portable GitHub CLI already exists."
}
else {
    $checksumsPartial = "$ghChecksums.partial"
    if (Test-Path -LiteralPath $checksumsPartial) { Remove-Item -LiteralPath $checksumsPartial -Force }
    Write-Host "Downloading GitHub CLI checksums..."
    & curl.exe --fail --location --retry 3 --retry-delay 2 --output $checksumsPartial $ghChecksumsUrl
    if ($LASTEXITCODE -ne 0) { throw "GitHub CLI checksum download failed." }
    Move-Item -LiteralPath $checksumsPartial -Destination $ghChecksums -Force
    $checksumLine = Get-Content -LiteralPath $ghChecksums | Where-Object { $_ -match "\s+$([regex]::Escape($ghArchiveName))$" } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($checksumLine)) {
        throw "The official checksum file does not contain $ghArchiveName."
    }
    $ghSha256 = (($checksumLine -split "\s+")[0]).ToUpperInvariant()
    Get-VerifiedDownload -Url $ghUrl -Destination $ghArchive -ExpectedSha256 $ghSha256
    $ghStage = Join-Path $runtimeRoot "gh-install-stage"
    if (Test-Path -LiteralPath $ghStage) { Remove-Item -LiteralPath $ghStage -Recurse -Force }
    New-Item -ItemType Directory -Path $ghStage -Force | Out-Null
    Expand-Archive -LiteralPath $ghArchive -DestinationPath $ghStage -Force
    $payload = $null
    if (Test-Path -LiteralPath (Join-Path $ghStage "bin\gh.exe") -PathType Leaf) {
        # Current GitHub CLI archives extract their payload directly at the root.
        $payload = $ghStage
    }
    else {
        # Retain compatibility with releases that wrap the payload in one directory.
        $inner = Get-ChildItem -LiteralPath $ghStage -Directory | Select-Object -First 1
        if ($null -ne $inner -and (Test-Path -LiteralPath (Join-Path $inner.FullName "bin\gh.exe") -PathType Leaf)) {
            $payload = $inner.FullName
        }
    }
    if ($null -eq $payload) {
        throw "The GitHub CLI archive did not contain bin\gh.exe."
    }
    if (Test-Path -LiteralPath $ghTarget) { Remove-Item -LiteralPath $ghTarget -Recurse -Force }
    Move-Item -LiteralPath $payload -Destination $ghTarget
    if (Test-Path -LiteralPath $ghStage) { Remove-Item -LiteralPath $ghStage -Recurse -Force }
}

$git = Find-GitExecutable
$gh = Find-GhExecutable
if ($null -eq $git -or $null -eq $gh) { throw "Portable tools were not installed correctly." }
& $git --version
if ($LASTEXITCODE -ne 0) { throw "git.exe self-test failed." }
& $gh --version
if ($LASTEXITCODE -ne 0) { throw "gh.exe self-test failed." }
Write-Host "[OK] Portable Git and GitHub CLI passed self-test."
