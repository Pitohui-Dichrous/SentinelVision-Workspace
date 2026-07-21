Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script:RemoteUrl = "https://github.com/Pitohui-Dichrous/SentinelVision-Workspace.git"

# Trust only this resolved workspace for the current process. This avoids
# removable-drive ownership warnings without changing the host computer's
# global Git configuration.
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = $script:ProjectRoot

function Get-ProjectRoot {
    return $script:ProjectRoot
}

function Find-GitExecutable {
    $portable = Join-Path $script:ProjectRoot "TOOLS\Git\cmd\git.exe"
    if (Test-Path -LiteralPath $portable -PathType Leaf) { return $portable }
    $command = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    return $null
}

function Find-GhExecutable {
    $portable = Join-Path $script:ProjectRoot "TOOLS\GitHubCLI\bin\gh.exe"
    if (Test-Path -LiteralPath $portable -PathType Leaf) { return $portable }
    $flatPortable = Join-Path $script:ProjectRoot "TOOLS\GitHubCLI\gh.exe"
    if (Test-Path -LiteralPath $flatPortable -PathType Leaf) { return $flatPortable }
    $command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    return $null
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

function Test-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    # Windows PowerShell 5.1 can promote native stderr to a terminating error
    # while ErrorActionPreference is Stop. Probes intentionally expect some
    # non-zero exit codes, so silence them and return a Boolean result.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments *> $null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Ensure-GitTools {
    $git = Ensure-GitExecutable
    $gh = Ensure-GhExecutable
    return [pscustomobject]@{ Git = $git; Gh = $gh }
}

function Ensure-GitExecutable {
    $git = Find-GitExecutable
    if ($null -eq $git) {
        & (Join-Path $PSScriptRoot "install_git_tools.ps1")
        if ($LASTEXITCODE -ne 0) { throw "Portable Git tool installation failed." }
        $git = Find-GitExecutable
    }
    if ($null -eq $git) { throw "git.exe is unavailable." }
    return $git
}

function Ensure-GhExecutable {
    $gh = Find-GhExecutable
    if ($null -eq $gh) {
        & (Join-Path $PSScriptRoot "install_git_tools.ps1")
        if ($LASTEXITCODE -ne 0) { throw "Portable GitHub CLI installation failed." }
        $gh = Find-GhExecutable
    }
    if ($null -eq $gh) { throw "gh.exe is unavailable." }
    return $gh
}

function Enable-PortableGitHubAuth {
    param(
        [Parameter(Mandatory = $true)][string]$Git,
        [Parameter(Mandatory = $true)][string]$Gh
    )
    # gh occasionally invokes git internally, so expose only the portable tools
    # to this process. Nothing is added to the host computer's permanent PATH.
    $gitDirectory = Split-Path -Parent $Git
    $ghDirectory = Split-Path -Parent $Gh
    $env:PATH = "$gitDirectory;$ghDirectory;$env:PATH"

}

function Get-PortableGitHubConfigArguments {
    param([Parameter(Mandatory = $true)][string]$Gh)
    # These -c values apply to one git process only. The first clears inherited
    # helpers; the second points Git at the gh executable on the drive's current
    # letter. No credential or absolute drive path is persisted in Git config.
    $ghForGit = $Gh.Replace("\", "/")
    return @(
        "-c", "credential.helper=",
        "-c", "credential.https://github.com.helper=!`"$ghForGit`" auth git-credential"
    )
}

function Ensure-OriginRemote {
    param([Parameter(Mandatory = $true)][string]$Git)
    if (Test-NativeCommand -Executable $Git -Arguments @("-C", $script:ProjectRoot, "remote", "get-url", "origin")) {
        Invoke-Checked -Executable $Git -Arguments @("-C", $script:ProjectRoot, "remote", "set-url", "origin", $script:RemoteUrl)
    }
    else {
        Invoke-Checked -Executable $Git -Arguments @("-C", $script:ProjectRoot, "remote", "add", "origin", $script:RemoteUrl)
    }
}

function Test-GitRepository {
    param([Parameter(Mandatory = $true)][string]$Git)
    return (Test-NativeCommand -Executable $Git -Arguments @("-C", $script:ProjectRoot, "rev-parse", "--is-inside-work-tree"))
}

function Assert-SafeStagedFiles {
    param([Parameter(Mandatory = $true)][string]$Git)
    $limit = 95MB
    $stagedOutput = @(& $Git -C $script:ProjectRoot -c core.quotepath=false diff --cached --name-only --diff-filter=ACMR -z)
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect staged files." }
    $staged = @(([string]::Join("`n", $stagedOutput)) -split "`0" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($relative in $staged) {
        $normalized = $relative.Replace("\", "/")
        $lower = $normalized.ToLowerInvariant()
        $blockedRoot = @(
            ".runtime/", "runtime/", "tools/", "datasets/", "runs/",
            "training_outputs/", "model_archive/"
        ) | Where-Object { $lower.StartsWith($_) } | Select-Object -First 1
        $blockedDataSubdirectory = (
            $lower -match '^data/[^/]+/' -and
            $lower -notmatch '^data/(hyps|images|scripts)/'
        )
        $blockedResultsModel = ($lower -match '^results/.+/')
        $blockedModelBinary = ($lower -match '\.(pt|pth|onnx|engine|torchscript|tflite|mlmodel|pb|weights|ckpt|safetensors)$')
        $blockedCredential = (
            $lower -match '(^|/)\.env($|\.)' -or
            $lower -match '\.(pem|pfx|p12)$' -or
            $lower -match '(^|/)(auth|[^/]*credentials[^/]*|[^/]*token[^/]*)\.json$'
        )
        $blockedRuntimeReport = ($lower -eq "alerts.csv" -or $lower -eq "integrity/workspace_manifest.json")
        if ($null -ne $blockedRoot -or $blockedDataSubdirectory -or $blockedResultsModel -or $blockedModelBinary -or $blockedCredential -or $blockedRuntimeReport) {
            throw "Protected workspace content cannot be committed: $normalized"
        }

        $path = Join-Path $script:ProjectRoot ($normalized.Replace("/", "\"))
        if ((Test-Path -LiteralPath $path -PathType Leaf) -and (Get-Item -LiteralPath $path).Length -gt $limit) {
            throw "GitHub blocks files above 100 MB. Remove this staged file: $relative"
        }
    }
}
