param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path ".\build\device_lister_ui") { Remove-Item ".\build\device_lister_ui" -Recurse -Force }
    if (Test-Path ".\dist\device_lister_ui") { Remove-Item ".\dist\device_lister_ui" -Recurse -Force }
}

Write-Host "[build] device_lister_ui - PyInstaller spec build"
uv run --group dev pyinstaller --noconfirm --clean .\device_lister_ui.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\device_lister_ui\device_lister_ui.exe")) {
    throw "Build reported success but dist\\device_lister_ui\\device_lister_ui.exe was not found"
}

Write-Host ""
Write-Host "[done] dist\device_lister_ui\device_lister_ui.exe"

