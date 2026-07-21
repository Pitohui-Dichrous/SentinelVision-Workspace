Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$target = Join-Path $root "PRETRAINED_WEIGHTS"
New-Item -ItemType Directory -Path $target -Force | Out-Null

$weights = @(
    @{ Name="yolov5n.pt";  Sha="4F180CF23BA0717ADA0BADD6C685026D73D48F184D00FC159C2641284B2AC0A3" },
    @{ Name="yolov5s.pt";  Sha="8B3B748C1E592DDD8868022E8732FDE20025197328490623CC16C6F24D0782EE" },
    @{ Name="yolov5m.pt";  Sha="61D933360BA5A7733A36764996C800287D973889D875227F5BEEDD2473A97A56" },
    @{ Name="yolov5l.pt";  Sha="2F603B7354C25454D1270663A14D8DDC1EEA98E5EEBC1D84CE0C6E3150FA155F" },
    @{ Name="yolov5x.pt";  Sha="9F27A794FA0308E2606F90565164571D6F0A0BA18A3CE2E5E5D323B71C157859" },
    @{ Name="yolov5n6.pt"; Sha="496C05A2B991DA15ACC6C4408C63DA287F2513A46C6756618776D4DD71170781" },
    @{ Name="yolov5s6.pt"; Sha="95BDA9019ADA63F37338308D0C81A66047A6FBABA061ABAFF271BA9334CF2A6F" },
    @{ Name="yolov5m6.pt"; Sha="7AFE7FA0F29A8351467200A2A5A33C7996D1155ED6636C7EDD8CA70B14951C64" },
    @{ Name="yolov5l6.pt"; Sha="B357B77F646B0190912985937C44E2923441369D5FDA69BA86DA2AA919212618" },
    @{ Name="yolov5x6.pt"; Sha="257C991B216D625427DC4D9C7C76D6C6D93CF3732797538DA34BDC02F97BFEF0" }
)

foreach ($weight in $weights) {
    $path = Join-Path $target $weight.Name
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $existing = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($existing -eq $weight.Sha) {
            Write-Host "[OK] $($weight.Name)"
            continue
        }
    }
    $partial = "$path.partial"
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    $url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/$($weight.Name)"
    Write-Host "Downloading $($weight.Name)..."
    & curl.exe --fail --location --retry 3 --retry-delay 2 --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($weight.Name)" }
    $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actual -ne $weight.Sha) {
        Remove-Item -LiteralPath $partial -Force
        throw "SHA-256 verification failed: $($weight.Name)"
    }
    Move-Item -LiteralPath $partial -Destination $path -Force
    Write-Host "[OK] $($weight.Name)"
}
Write-Host "[OK] All 10 official YOLOv5 v7.0 detection checkpoints are ready."

