param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path ".\build\device_lister") { Remove-Item ".\build\device_lister" -Recurse -Force }
    if (Test-Path ".\dist\device_lister") { Remove-Item ".\dist\device_lister" -Recurse -Force }
}

Write-Host "[build] device_lister - PyInstaller spec build"
uv run --group dev pyinstaller --noconfirm --clean .\device_lister.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\device_lister\device_lister.exe")) {
    throw "Build reported success but dist\\device_lister\\device_lister.exe was not found"
}

Write-Host ""
Write-Host "[done] dist\device_lister\device_lister.exe"

