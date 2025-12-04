$ErrorActionPreference = "Stop"

$venvPath        = ".venv"
$requirementsFile = "_requirements.txt"
$extensionsFile   = "_extensions.txt"

Write-Host "=== Creating/checking virtual environment '$venvPath' ==="

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    py -3.13-64 -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists, skipping creation."
}

Write-Host "=== Activating virtual environment ==="
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

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
    if (-not (Get-Command code -ErrorAction SilentlyContinue)) {
        Write-Warning "'code' CLI (VS Code) nicht in PATH gefunden. Extensions werden übersprungen."
    } else {
        Get-Content $extensionsFile | ForEach-Object {
            $ext = $_.Trim()
            if ($ext -and -not $ext.StartsWith("#")) {
                Write-Host "Installing VS Code extension '$ext'..."
                code --install-extension $ext
            }
        }
    }
} else {
    Write-Warning "Extensions file '$extensionsFile' not found. Skipping VS Code extension install."
}

Write-Host "=== Setup finished. Virtual environment is active in this PowerShell session. ==="
