Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$tools = Ensure-GitTools
$git = $tools.Git
$gh = $tools.Gh

if (-not (Test-GitRepository -Git $git)) {
    & (Join-Path $PSScriptRoot "enable_git.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Git initialization failed." }
}

& (Join-Path $PSScriptRoot "save_version.ps1") -NoPush
if ($LASTEXITCODE -ne 0) { throw "Local version save failed." }
Ensure-OriginRemote -Git $git

if (-not (Test-NativeCommand -Executable $gh -Arguments @("auth", "status", "--hostname", "github.com"))) {
    Write-Host "A browser will open for one-time GitHub authorization."
    Write-Host "The token is stored by Windows on this computer, never on the removable drive."
    Invoke-Checked -Executable $gh -Arguments @("auth", "login", "--hostname", "github.com", "--git-protocol", "https", "--web")
}
Invoke-Checked -Executable $gh -Arguments @("auth", "setup-git")

Invoke-Checked -Executable $git -Arguments @("-C", $root, "fetch", "origin", "main")

if (-not (Test-NativeCommand -Executable $git -Arguments @("-C", $root, "merge-base", "HEAD", "origin/main"))) {
    Invoke-Checked -Executable $git -Arguments @("-C", $root, "merge", "origin/main", "--allow-unrelated-histories", "-X", "ours", "-m", "Merge GitHub repository initialization")
}
else {
    & $git -C $root merge-base --is-ancestor origin/main HEAD
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked -Executable $git -Arguments @("-C", $root, "pull", "--rebase", "--autostash", "origin", "main")
    }
}

Invoke-Checked -Executable $git -Arguments @("-C", $root, "push", "--set-upstream", "origin", "main")
Write-Host "[OK] Private GitHub repository synchronized."
Write-Host "https://github.com/Pitohui-Dichrous/SentinelVision-Workspace"
