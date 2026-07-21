Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$git = Ensure-GitExecutable

if (-not (Test-GitRepository -Git $git)) {
    throw "This workspace is not a Git repository."
}

$status = @(& $git -C $root status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect the working tree." }
if ($status.Count -gt 0) {
    Write-Host "[BLOCKED] The working tree contains uncommitted changes."
    Write-Host "Save or discard those changes before reverting a commit."
    & $git -C $root status --short
    exit 2
}

if (-not (Test-NativeCommand -Executable $git -Arguments @("-C", $root, "rev-parse", "--verify", "HEAD~1"))) {
    throw "The repository does not have a previous commit to return to."
}

$parentLine = ((& $git -C $root rev-list --parents -n 1 HEAD) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect the latest commit." }
if (($parentLine -split "\s+").Count -gt 2) {
    throw "The latest commit is a merge commit. It requires a manual mainline choice and was not changed."
}

$currentHash = ((& $git -C $root rev-parse HEAD) | Out-String).Trim()
$shortHash = ((& $git -C $root rev-parse --short=8 HEAD) | Out-String).Trim()
$summary = ((& $git --no-pager -C $root log -1 --date=local --pretty=format:"%h  %ad  %s") | Out-String).Trim()

Write-Host ""
Write-Host "Latest commit: $summary"
Write-Host ""
Write-Host "This creates a NEW commit that reverses the latest commit."
Write-Host "The original commit remains in history. Reverting this new revert later restores it."
Write-Host ""

$first = Read-Host "Confirmation 1/2 - type REVERT"
if ($first -cne "REVERT") {
    Write-Host "[CANCELLED] Nothing was changed."
    exit 3
}
$second = Read-Host "Confirmation 2/2 - type current commit ID $shortHash"
if ($second.Trim() -cne $shortHash) {
    Write-Host "[CANCELLED] Commit ID did not match. Nothing was changed."
    exit 3
}

Invoke-Checked -Executable $git -Arguments @("-C", $root, "revert", "--no-edit", $currentHash)
Write-Host "[OK] A new revert commit was created locally."

$gh = Find-GhExecutable
if ($null -ne $gh) {
    Enable-PortableGitHubAuth -Git $git -Gh $gh
}
if ($null -ne $gh -and (Test-NativeCommand -Executable $gh -Arguments @("auth", "status", "--hostname", "github.com"))) {
    $gitAuth = @(Get-PortableGitHubConfigArguments -Gh $gh)
    Ensure-OriginRemote -Git $git
    Invoke-Checked -Executable $git -Arguments @($gitAuth + @("-C", $root, "push", "origin", "main"))
    Write-Host "[OK] The revert commit was synchronized to GitHub."
}
else {
    Write-Host "[INFO] GitHub is not authenticated. The revert commit is safe locally."
    Write-Host "Run SYNC_GITHUB.cmd later to upload it."
}
