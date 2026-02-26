param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Clean) {
    if (Test-Path ".\build\deviceMngUI") { Remove-Item ".\build\deviceMngUI" -Recurse -Force }
    if (Test-Path ".\dist\deviceMngUI") { Remove-Item ".\dist\deviceMngUI" -Recurse -Force }
}

Write-Host "[build] deviceMngUI - PyInstaller spec build"
uv run --group dev pyinstaller --noconfirm --clean .\deviceMngUI.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path ".\dist\deviceMngUI\deviceMngUI.exe")) {
    throw "Build reported success but dist\\deviceMngUI\\deviceMngUI.exe was not found"
}
if (Test-Path ".\.env") {
    Copy-Item ".\.env" ".\dist\deviceMngUI\.env" -Force
}

Write-Host ""
Write-Host "[done] dist\deviceMngUI\deviceMngUI.exe"

