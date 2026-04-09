param(
    [string]$Python = "C:\p311\python.exe",
    [switch]$RunTests,
    [switch]$InstallOllama,
    [switch]$InstallDocker,
    [switch]$BuildSandbox,
    [switch]$PullModel,
    [string]$Model = "qwen3-coder:30b"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Command([string]$Name) {
    try {
        return Get-Command $Name -ErrorAction Stop
    } catch {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Ensure-File([string]$Source, [string]$Target) {
    if (-not (Test-Path $Target)) {
        Copy-Item $Source $Target
        Write-Host "Created $Target from template." -ForegroundColor DarkGray
    }
}

function Ensure-UserDir([string]$PathValue) {
    if (-not (Test-Path $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
    }
}

function Ensure-DockerBinOnPath() {
    $dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
    if (-not (Test-Path $dockerBin)) {
        return
    }

    if (-not ($env:Path -split ";" | Where-Object { $_ -eq $dockerBin })) {
        $env:Path += ";$dockerBin"
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not ($userPath -split ";" | Where-Object { $_ -eq $dockerBin })) {
        $newPath = (($userPath ?? "").TrimEnd(";") + ";$dockerBin").Trim(";")
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    }
}

function Ensure-OllamaBinOnPath() {
    $ollamaBin = "C:\Users\DlgFresh\AppData\Local\Programs\Ollama"
    if (-not (Test-Path $ollamaBin)) {
        return
    }

    if (-not ($env:Path -split ";" | Where-Object { $_ -eq $ollamaBin })) {
        $env:Path += ";$ollamaBin"
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not ($userPath -split ";" | Where-Object { $_ -eq $ollamaBin })) {
        $newPath = (($userPath ?? "").TrimEnd(";") + ";$ollamaBin").Trim(";")
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    }
}

function Repair-EditableInstallPath() {
    $pth = ".\.venv\Lib\site-packages\_codesmith.pth"
    if (-not (Test-Path $pth)) {
        return
    }

    $srcPath = (Resolve-Path ".\src").Path
    [System.IO.File]::WriteAllText(
        (Resolve-Path $pth),
        $srcPath + [Environment]::NewLine,
        [System.Text.Encoding]::GetEncoding(1251)
    )
}

function Install-WingetPackage([string]$PackageId) {
    $winget = Ensure-Command "winget"
    & $winget.Source install --id $PackageId -e --source winget `
        --accept-package-agreements --accept-source-agreements
}

Write-Step "Checking local toolchain"
Ensure-Command "uv" | Out-Null
if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}
Ensure-DockerBinOnPath
Ensure-OllamaBinOnPath

Write-Step "Creating local virtual environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    uv venv --python $Python .venv
}

Write-Step "Installing project dependencies"
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"
Repair-EditableInstallPath

Write-Step "Ensuring local config files exist"
Ensure-File ".\config.example.yaml" ".\config.yaml"
Ensure-File ".\.env.example" ".\.env"

Write-Step "Preparing Codesmith data directories"
$homeCodesmith = Join-Path $HOME ".codesmith"
Ensure-UserDir $homeCodesmith
Ensure-UserDir (Join-Path $homeCodesmith "logs")
Ensure-UserDir (Join-Path $homeCodesmith "workspaces")

if ($InstallOllama) {
    Write-Step "Installing Ollama"
    Install-WingetPackage "Ollama.Ollama"
    Ensure-OllamaBinOnPath
}

if ($InstallDocker) {
    Write-Step "Installing Docker Desktop"
    Install-WingetPackage "Docker.DockerDesktop"
    Ensure-DockerBinOnPath
}

if ($PullModel) {
    Write-Step "Pulling Ollama model $Model"
    $ollama = Get-Command "ollama" -ErrorAction SilentlyContinue
    if (-not $ollama) {
        throw "ollama command not found. Install Ollama first or re-open the shell."
    }
    & $ollama.Source pull $Model
}

if ($RunTests) {
    Write-Step "Running focused test suite"
    .\.venv\Scripts\pytest.exe tests/test_config.py tests/test_registry.py tests/test_filesystem.py -q
}

if ($BuildSandbox) {
    Write-Step "Building sandbox image"
    & ".\scripts\build-sandbox.ps1"
}

Write-Step "Bootstrap complete"
Write-Host "Activate the environment with .\.venv\Scripts\activate" -ForegroundColor Green
Write-Host "Use .\scripts\doctor.ps1 to validate local runtime readiness." -ForegroundColor Green
