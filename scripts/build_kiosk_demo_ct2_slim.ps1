param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
    if (Test-Path ".\dist\kiosk_demo_ct2_slim") { Remove-Item ".\dist\kiosk_demo_ct2_slim" -Recurse -Force }
}

Write-Host "[build] kiosk_demo_ct2_slim - PyInstaller spec build"
uv run --group dev pyinstaller --noconfirm --clean .\kiosk_demo_ct2_slim.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\kiosk_demo_ct2_slim\kiosk_demo_ct2_slim.exe")) {
    throw "Build reported success but dist\\kiosk_demo_ct2_slim\\kiosk_demo_ct2_slim.exe was not found"
}
if (Test-Path ".\.env") {
    Copy-Item ".\.env" ".\dist\kiosk_demo_ct2_slim\.env" -Force
}

Write-Host ""
Write-Host "[done] dist\kiosk_demo_ct2_slim\kiosk_demo_ct2_slim.exe"

