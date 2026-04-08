#requires -version 5
<#
.SYNOPSIS
    Codesmith launcher - interactive menu for the codesmith agent.

.DESCRIPTION
    Top-level entry point invoked by codesmith.bat (or directly from a
    PowerShell prompt). Without arguments, prints a menu and dispatches
    to the chosen action. With an argument, runs that action directly:

        codesmith.ps1 web      - start Web UI
        codesmith.ps1 repl     - interactive chat
        codesmith.ps1 info     - config + health
        codesmith.ps1 doctor   - sandbox / docker / ollama check
        codesmith.ps1 build    - rebuild sandbox image
        codesmith.ps1 install  - bootstrap venv + deps

    Designed for double-click launches: handles missing venv, ensures
    UTF-8 console, and gives friendly error messages instead of stack
    traces when something is not installed.

    NOTE: this file is kept ASCII-only on purpose. Windows PowerShell 5
    reads .ps1 files as the system ANSI codepage, not UTF-8, so any
    non-ASCII source character would corrupt the parser.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('menu','web','repl','info','doctor','build','install','chat','solve','help')]
    [string] $Action = 'menu',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Force UTF-8 so the agent's Rich/box-drawing output renders correctly.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
chcp 65001 > $null

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Codesmith = Join-Path $RepoRoot ".venv\Scripts\codesmith.exe"

function Write-Title {
    param([string] $Text)
    Write-Host ""
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "  $('=' * $Text.Length)" -ForegroundColor DarkCyan
    Write-Host ""
}

function Test-Venv {
    if (-not (Test-Path $Python)) {
        Write-Host ""
        Write-Host "  ! venv not found at .venv\" -ForegroundColor Yellow
        Write-Host "  ! run option [7] Install (or scripts\bootstrap.ps1) first." -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
    return $true
}

function Invoke-Codesmith {
    param([string[]] $CmdArgs)
    if (-not (Test-Venv)) {
        $script:LastActionExit = 1
        return
    }
    if (-not (Test-Path $Codesmith)) {
        & $Python -m codesmith.cli @CmdArgs
    } else {
        & $Codesmith @CmdArgs
    }
    $script:LastActionExit = $LASTEXITCODE
}

function Show-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "  +======================================+" -ForegroundColor Cyan
    Write-Host "  |           " -ForegroundColor Cyan -NoNewline
    Write-Host "C O D E S M I T H" -ForegroundColor White -NoNewline
    Write-Host "          |" -ForegroundColor Cyan
    Write-Host "  |   " -ForegroundColor Cyan -NoNewline
    Write-Host "hybrid AI coder + sandbox" -ForegroundColor DarkGray -NoNewline
    Write-Host "      |" -ForegroundColor Cyan
    Write-Host "  +======================================+" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    [1] " -ForegroundColor Yellow -NoNewline
    Write-Host "Web UI       " -NoNewline
    Write-Host "  - browser chat with live SSE stream" -ForegroundColor DarkGray
    Write-Host "    [2] " -ForegroundColor Yellow -NoNewline
    Write-Host "REPL         " -NoNewline
    Write-Host "  - interactive multi-turn chat" -ForegroundColor DarkGray
    Write-Host "    [3] " -ForegroundColor Yellow -NoNewline
    Write-Host "Solve task   " -NoNewline
    Write-Host "  - run a task with self-repair loop" -ForegroundColor DarkGray
    Write-Host "    [4] " -ForegroundColor Yellow -NoNewline
    Write-Host "Info         " -NoNewline
    Write-Host "  - config + live health checks" -ForegroundColor DarkGray
    Write-Host "    [5] " -ForegroundColor Yellow -NoNewline
    Write-Host "Doctor       " -NoNewline
    Write-Host "  - docker / sandbox / ollama probes" -ForegroundColor DarkGray
    Write-Host "    [6] " -ForegroundColor Yellow -NoNewline
    Write-Host "Build sandbox" -NoNewline
    Write-Host "  - rebuild docker sandbox image" -ForegroundColor DarkGray
    Write-Host "    [7] " -ForegroundColor Yellow -NoNewline
    Write-Host "Install      " -NoNewline
    Write-Host "  - bootstrap venv + dependencies" -ForegroundColor DarkGray
    Write-Host "    [Q] " -ForegroundColor DarkYellow -NoNewline
    Write-Host "Quit"
    Write-Host ""
    $choice = Read-Host "  choose"
    return $choice.Trim().ToLower()
}

function Resolve-MenuChoice {
    param([string] $Choice)
    switch ($Choice) {
        '1' { return 'web' }
        '2' { return 'repl' }
        '3' { return 'solve' }
        '4' { return 'info' }
        '5' { return 'doctor' }
        '6' { return 'build' }
        '7' { return 'install' }
        'q' { return 'quit' }
        default { return '' }
    }
}

# Run-Action MUST NOT return a value via the pipeline. PowerShell would
# otherwise capture all native-command stdout (e.g. codesmith info output)
# as part of the return value and swallow it. Instead, every action sets
# $script:LastActionExit and we read that after the call.
function Run-Action {
    param([string] $Name)
    $script:LastActionExit = 0

    switch ($Name) {
        'web'  { Invoke-Codesmith (@('web') + $Rest); return }
        'repl' { Invoke-Codesmith @('repl'); return }
        'info' { Invoke-Codesmith @('info'); return }
        'chat' {
            if (-not $Rest -or $Rest.Length -eq 0) {
                $msg = Read-Host "  message"
                if ([string]::IsNullOrWhiteSpace($msg)) { return }
                Invoke-Codesmith @('chat', $msg)
                return
            }
            Invoke-Codesmith (@('chat') + $Rest)
            return
        }
        'solve' {
            if (-not $Rest -or $Rest.Length -eq 0) {
                $task = Read-Host "  task"
                if ([string]::IsNullOrWhiteSpace($task)) { return }
                Invoke-Codesmith @('solve', $task)
                return
            }
            Invoke-Codesmith (@('solve') + $Rest)
            return
        }
        'doctor' {
            Write-Title "doctor"
            & (Join-Path $PSScriptRoot 'doctor.ps1')
            $script:LastActionExit = $LASTEXITCODE
            return
        }
        'build' {
            Write-Title "build sandbox"
            & (Join-Path $PSScriptRoot 'build-sandbox.ps1')
            $script:LastActionExit = $LASTEXITCODE
            return
        }
        'install' {
            Write-Title "bootstrap"
            & (Join-Path $PSScriptRoot 'bootstrap.ps1')
            $script:LastActionExit = $LASTEXITCODE
            return
        }
        'help' {
            Write-Host ""
            Write-Host "  usage: codesmith.bat [web|repl|info|doctor|build|install|chat|solve]" -ForegroundColor White
            Write-Host ""
            Write-Host "  no argument: opens an interactive menu." -ForegroundColor DarkGray
            return
        }
        default {
            Write-Host "  unknown action: $Name" -ForegroundColor Red
            $script:LastActionExit = 2
            return
        }
    }
}

if ($Action -ne 'menu') {
    Run-Action $Action
    exit $script:LastActionExit
}

while ($true) {
    $choice = Show-Menu
    $name = Resolve-MenuChoice $choice
    if ($name -eq '') {
        Write-Host "  ? '$choice' - try 1..7 or Q" -ForegroundColor Red
        Start-Sleep -Seconds 1
        continue
    }
    if ($name -eq 'quit') {
        exit 0
    }
    Run-Action $name
    Write-Host ""
    Write-Host "  press Enter to return to menu..." -ForegroundColor DarkGray
    [void](Read-Host)
}
