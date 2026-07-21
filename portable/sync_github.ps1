Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "git_common.ps1")

$root = Get-ProjectRoot
$tools = Ensure-GitTools
$git = $tools.Git
$gh = $tools.Gh
$expectedLogin = "Pitohui-Dichrous"
$repository = "Pitohui-Dichrous/SentinelVision-Workspace"

# GitHub CLI may invoke `git` during `gh auth login` (for example while
# configuring the HTTPS credential helper).  Add both portable tool folders
# before any gh command, so a clean public computer does not need Git installed
# globally or have the removable-drive Git directory in its permanent PATH.
Enable-PortableGitHubAuth -Git $git -Gh $gh

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
$gitAuth = @(Get-PortableGitHubConfigArguments -Gh $gh)

$actualLogin = ((& $gh api user --jq ".login") | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $actualLogin -ne $expectedLogin) {
    throw "GitHub account mismatch. Expected $expectedLogin but received $actualLogin."
}

if (-not (Test-NativeCommand -Executable $gh -Arguments @("repo", "view", $repository))) {
    Invoke-Checked -Executable $gh -Arguments @(
        "repo", "create", $repository, "--private",
        "--description", "Portable SentinelVision YOLOv5 training and detection workspace"
    )
}
$repositoryJson = ((& $gh repo view $repository --json "nameWithOwner,visibility") | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to verify the GitHub repository." }
$repositoryInfo = $repositoryJson | ConvertFrom-Json
if ($repositoryInfo.nameWithOwner -ne $repository -or $repositoryInfo.visibility -ne "PRIVATE") {
    throw "Refusing to synchronize because the target repository is not the expected private repository."
}

$remoteMain = @(& $git @gitAuth -C $root ls-remote --heads origin refs/heads/main)
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect the private GitHub repository." }

if ($remoteMain.Count -gt 0) {
    Invoke-Checked -Executable $git -Arguments @($gitAuth + @("-C", $root, "fetch", "origin", "main"))
    if (-not (Test-NativeCommand -Executable $git -Arguments @("-C", $root, "merge-base", "HEAD", "origin/main"))) {
        Invoke-Checked -Executable $git -Arguments @("-C", $root, "merge", "origin/main", "--allow-unrelated-histories", "-X", "ours", "-m", "Merge GitHub repository initialization")
    }
    else {
        & $git -C $root merge-base --is-ancestor origin/main HEAD
        if ($LASTEXITCODE -ne 0) {
            Invoke-Checked -Executable $git -Arguments @($gitAuth + @("-C", $root, "pull", "--rebase", "--autostash", "origin", "main"))
        }
    }
}

Invoke-Checked -Executable $git -Arguments @($gitAuth + @("-C", $root, "push", "--set-upstream", "origin", "main"))
Write-Host "[OK] Private GitHub repository synchronized."
Write-Host "https://github.com/$repository"
