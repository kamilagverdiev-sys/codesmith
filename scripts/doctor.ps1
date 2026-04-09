param(
    [switch]$RunTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Check([string]$Name, [scriptblock]$Probe) {
    try {
        $value = & $Probe
        [pscustomobject]@{
            Name = $Name
            Status = "OK"
            Details = if ($null -eq $value) { "" } else { [string]$value }
        }
    } catch {
        [pscustomobject]@{
            Name = $Name
            Status = "MISS"
            Details = $_.Exception.Message
        }
    }
}

function Get-OllamaExecutable() {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $fallback = "C:\Users\DlgFresh\AppData\Local\Programs\Ollama\ollama.exe"
    if (Test-Path $fallback) {
        return $fallback
    }

    throw "ollama executable not found"
}

$checks = @(
    (Get-Check "Git" { (Get-Command git -ErrorAction Stop).Source }),
    (Get-Check "uv" { (Get-Command uv -ErrorAction Stop).Source }),
    (Get-Check "Python 3.11" { & "C:\p311\python.exe" --version }),
    (Get-Check ".venv" { Resolve-Path ".\.venv\Scripts\python.exe" }),
    (Get-Check "config.yaml" { Resolve-Path ".\config.yaml" }),
    (Get-Check ".env" { Resolve-Path ".\.env" }),
    (Get-Check "Docker CLI" {
        $dockerCli = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        if (-not (Test-Path $dockerCli)) {
            throw "docker.exe not found"
        }
        & $dockerCli --version
    }),
    (Get-Check "Docker daemon" {
        $dockerCli = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        & $dockerCli info --format "{{.ServerVersion}}"
    }),
    (Get-Check "Sandbox image" {
        $dockerCli = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        & $dockerCli image inspect codesmith-sandbox:latest --format "{{.Id}}"
    }),
    (Get-Check "Ollama" { Get-OllamaExecutable }),
    (Get-Check "Ollama API" {
        (Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing).StatusCode
    }),
    (Get-Check "Ollama models" {
        $payload = (Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing).Content | ConvertFrom-Json
        (($payload.models | ForEach-Object { $_.name }) -join ", ")
    }),
    (Get-Check "ANTHROPIC_API_KEY" {
        if (-not $env:ANTHROPIC_API_KEY) { throw "not set" }
        "SET"
    }),
    (Get-Check "OPENAI_API_KEY" {
        if (-not $env:OPENAI_API_KEY) { throw "not set" }
        "SET"
    })
)

$checks | Format-Table -AutoSize

if ($RunTests) {
    if (-not (Test-Path ".\.venv\Scripts\pytest.exe")) {
        throw "pytest is missing from .venv"
    }
    .\.venv\Scripts\pytest.exe tests/test_config.py tests/test_registry.py tests/test_filesystem.py -q
}
