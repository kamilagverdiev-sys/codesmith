param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\uvicorn.exe")) {
    throw "uvicorn is missing. Run .\scripts\bootstrap.ps1 first."
}

$args = @("codesmith.api.main:app", "--host", $Host, "--port", "$Port")
if ($Reload) {
    $args += "--reload"
}

& .\.venv\Scripts\uvicorn.exe @args
