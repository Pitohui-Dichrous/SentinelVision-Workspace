Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$tools = Ensure-GitTools
$git = $tools.Git

if (-not (Test-GitRepository -Git $git)) {
    Invoke-Checked -Executable $git -Arguments @("-C", $root, "init", "-b", "main")
}

Invoke-Checked -Executable $git -Arguments @("-C", $root, "config", "user.name", "Pitohui-Dichrous")
Invoke-Checked -Executable $git -Arguments @("-C", $root, "config", "user.email", "160625893+Pitohui-Dichrous@users.noreply.github.com")
Invoke-Checked -Executable $git -Arguments @("-C", $root, "config", "core.autocrlf", "false")
Invoke-Checked -Executable $git -Arguments @("-C", $root, "config", "core.longpaths", "true")
Invoke-Checked -Executable $git -Arguments @("-C", $root, "config", "fetch.prune", "true")
Ensure-OriginRemote -Git $git

Invoke-Checked -Executable $git -Arguments @("-C", $root, "add", "-A")
Assert-SafeStagedFiles -Git $git

$hasHead = Test-NativeCommand -Executable $git -Arguments @("-C", $root, "rev-parse", "--verify", "HEAD")
& $git -C $root diff --cached --quiet
$hasChanges = ($LASTEXITCODE -eq 1)
if (-not $hasHead -and -not $hasChanges) {
    throw "No source files were selected for the initial commit. Check .gitignore."
}
if ($hasChanges) {
    $message = if ($hasHead) { "Configure portable Git workflow" } else { "Initial SentinelVision portable workspace" }
    Invoke-Checked -Executable $git -Arguments @("-C", $root, "commit", "-m", $message)
}
else {
    Write-Host "[OK] Git repository already has no pending staged changes."
}

Write-Host "[OK] Local Git repository is ready."
& $git -C $root status --short --branch
