Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$tools = Ensure-GitTools
$git = $tools.Git
$gh = $tools.Gh

if (-not (Test-GitRepository -Git $git)) {
    throw "This workspace is not a Git repository."
}

$branch = ((& $git -C $root branch --show-current) | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
    throw "Hard deletion is allowed only while the main branch is checked out."
}

$status = @(& $git -C $root status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect the working tree." }
if ($status.Count -gt 0) {
    Write-Host "[BLOCKED] The working tree contains uncommitted changes."
    Write-Host "A hard reset would destroy them, so no action was taken."
    & $git -C $root status --short
    exit 2
}

if (-not (Test-NativeCommand -Executable $git -Arguments @("-C", $root, "rev-parse", "--verify", "HEAD~1"))) {
    throw "The repository does not have a previous commit to reset to."
}

$currentHash = ((& $git -C $root rev-parse HEAD) | Out-String).Trim()
$shortHash = ((& $git -C $root rev-parse --short=8 HEAD) | Out-String).Trim()
$targetHash = ((& $git -C $root rev-parse HEAD~1) | Out-String).Trim()
$targetShort = ((& $git -C $root rev-parse --short=8 HEAD~1) | Out-String).Trim()
$summary = ((& $git --no-pager -C $root log -1 --date=local --pretty=format:"%h  %ad  %s") | Out-String).Trim()

Enable-PortableGitHubAuth -Git $git -Gh $gh
Ensure-OriginRemote -Git $git
if (-not (Test-NativeCommand -Executable $gh -Arguments @("auth", "status", "--hostname", "github.com"))) {
    throw "GitHub is not authenticated. Run SYNC_GITHUB.cmd first; no commit was deleted."
}
$gitAuth = @(Get-PortableGitHubConfigArguments -Gh $gh)
$remoteLines = @(& $git @gitAuth -C $root ls-remote --heads origin refs/heads/main)
if ($LASTEXITCODE -ne 0) {
    throw "The GitHub branch could not be checked. No commit was deleted."
}
$remoteHash = ""
if ($remoteLines.Count -gt 0) {
    $remoteHash = (($remoteLines[0] -split "\s+")[0]).Trim()
    if ($remoteHash -ne $currentHash -and $remoteHash -ne $targetHash) {
        throw "GitHub main does not match the current or target commit. Synchronize and inspect it first."
    }
}

Write-Host ""
Write-Host "LATEST COMMIT TO DELETE: $summary"
Write-Host "NEW main HEAD:          $targetShort"
Write-Host ""
Write-Host "DANGER: This removes the latest commit from local history."
if ($remoteHash -eq $currentHash) {
    Write-Host "DANGER: GitHub main will also be rewritten with force-with-lease."
}
Write-Host "The working tree is clean; no uncommitted source changes were detected."
Write-Host ""

$first = Read-Host "Confirmation 1/2 - type DELETE"
if ($first -cne "DELETE") {
    Write-Host "[CANCELLED] Nothing was changed."
    exit 3
}
$second = Read-Host "Confirmation 2/2 - type commit ID $shortHash"
if ($second.Trim() -cne $shortHash) {
    Write-Host "[CANCELLED] Commit ID did not match. Nothing was changed."
    exit 3
}

if ($remoteHash -eq $currentHash) {
    $lease = "--force-with-lease=refs/heads/main:$currentHash"
    $refspec = "${targetHash}:refs/heads/main"
    Invoke-Checked -Executable $git -Arguments @($gitAuth + @("-C", $root, "push", $lease, "origin", $refspec))
    Write-Host "[OK] GitHub main was moved back with a precise lease."
}

Invoke-Checked -Executable $git -Arguments @("-C", $root, "reset", "--hard", $targetHash)
Write-Host "[OK] The latest commit was removed. main now points to $targetShort."
Write-Host "[INFO] Git may retain the removed commit temporarily in reflog, but do not rely on that as a backup."
