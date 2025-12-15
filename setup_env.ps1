$ErrorActionPreference = "Stop"

# Always work relative to the script location (so you can run it from anywhere)
$root = $PSScriptRoot

$venvPath         = Join-Path $root ".venv"
$requirementsFile = Join-Path $root "_requirements.txt"
$extensionsFile   = Join-Path $root "_extensions.txt"

# Optional: force a specific Python on Windows via the "py" launcher (if available)
$pyLauncherArgs = @("-3.13")   # e.g. "-3.13" or "-3.13-64"

function Resolve-PythonRunner {
    if ($IsWindows -and (Get-Command py -ErrorAction SilentlyContinue)) {
        return @{ Exe = "py"; Args = $pyLauncherArgs }
    }

    foreach ($cmd in @("python3", "python")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            return @{ Exe = $cmd; Args = @() }
        }
    }

    throw "No Python found. On Ubuntu install: sudo apt install python3 python3-venv python3-pip"
}

function Resolve-CodeCli {
    foreach ($cmd in @("code", "code-insiders", "codium", "code-oss")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) { return $cmd }
    }
    return $null
}

Write-Host "=== Creating/checking virtual environment '$venvPath' ==="
$py = Resolve-PythonRunner

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment using: $($py.Exe) $($py.Args -join ' ')"
    & $py.Exe @($py.Args) -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists, skipping creation."
}

Write-Host "=== Activating virtual environment ==="
$activateScript = if ($IsWindows) {
    Join-Path $venvPath "Scripts/Activate.ps1"
} else {
    Join-Path $venvPath "bin/Activate.ps1"
}

if (-not (Test-Path $activateScript)) {
    throw "Activate script not found at '$activateScript'."
}

& $activateScript
Write-Host "Virtual env: $env:VIRTUAL_ENV"

Write-Host "=== Upgrading pip ==="
python -m pip install --upgrade pip

if (Test-Path $requirementsFile) {
    Write-Host "=== Installing Python packages from $requirementsFile ==="
    python -m pip install -r $requirementsFile
} else {
    Write-Warning "Requirements file '$requirementsFile' not found. Skipping Python package install."
}

Write-Host "=== Installing VS Code extensions ==="
if (Test-Path $extensionsFile) {
    $codeCmd = Resolve-CodeCli
    if (-not $codeCmd) {
        Write-Warning "VS Code CLI ('code'/'codium') not found in PATH. Skipping extension install."
    } else {
        Get-Content $extensionsFile | ForEach-Object {
            $ext = $_.Trim()
            if ($ext -and -not $ext.StartsWith("#")) {
                Write-Host "Installing VS Code extension '$ext'..."
                & $codeCmd --install-extension $ext
            }
        }
    }
} else {
    Write-Warning "Extensions file '$extensionsFile' not found. Skipping VS Code extension install."
}

Write-Host "=== Setup finished. Virtual environment is active in this PowerShell session. ==="

