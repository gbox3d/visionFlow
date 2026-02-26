param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$OutDir = ".\dist_whl"

if ($Clean -and (Test-Path $OutDir)) {
    Remove-Item $OutDir -Recurse -Force
}

Write-Host "[build] wheel only (separate output): $OutDir"
uv build --wheel --out-dir $OutDir --clear
if ($LASTEXITCODE -ne 0) {
    throw "uv wheel build failed with exit code $LASTEXITCODE"
}

$wheel = Get-ChildItem $OutDir -Filter *.whl -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $wheel) {
    throw "Wheel build reported success but no .whl file was found in $OutDir"
}

Write-Host ""
Write-Host "[done] $($wheel.FullName)"

