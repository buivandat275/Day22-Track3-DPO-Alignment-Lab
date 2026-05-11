$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = "C:\Users\buiva\AppData\Local\Programs\Python\Python312\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Python not found at $python"
    Write-Host "Install Python or update scripts/verify-windows.ps1 to point at your python.exe."
    exit 1
}

Push-Location $repo
try {
    $env:PYTHONIOENCODING = "utf-8"
    & $python scripts\verify.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
