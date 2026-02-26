param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path ".\build\mic_level_monitor") { Remove-Item ".\build\mic_level_monitor" -Recurse -Force }
    if (Test-Path ".\dist\mic_level_monitor") { Remove-Item ".\dist\mic_level_monitor" -Recurse -Force }
}

Write-Host "[build] mic_level_monitor - PyInstaller spec build"
uv run --group dev pyinstaller --noconfirm --clean .\mic_level_monitor.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\mic_level_monitor\mic_level_monitor.exe")) {
    throw "Build reported success but dist\\mic_level_monitor\\mic_level_monitor.exe was not found"
}
if (Test-Path ".\.env") {
    Copy-Item ".\.env" ".\dist\mic_level_monitor\.env" -Force
}

Write-Host ""
Write-Host "[done] dist\mic_level_monitor\mic_level_monitor.exe"
