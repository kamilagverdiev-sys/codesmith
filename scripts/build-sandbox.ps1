Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if (Test-Path $dockerBin) {
    if (-not ($env:Path -split ";" | Where-Object { $_ -eq $dockerBin })) {
        $env:Path += ";$dockerBin"
    }
}

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    $dockerCli = Join-Path $dockerBin "docker.exe"
    if (-not (Test-Path $dockerCli)) {
        throw "docker.exe not found. Install Docker Desktop first."
    }
    $docker = Get-Command $dockerCli
}

Write-Host "==> Docker daemon" -ForegroundColor Cyan
& $docker.Source info | Out-Host

Write-Host "==> Building codesmith-sandbox:latest" -ForegroundColor Cyan
& $docker.Source build -t codesmith-sandbox:latest -f docker/sandbox.Dockerfile docker/

Write-Host "==> Sandbox image ready" -ForegroundColor Green
