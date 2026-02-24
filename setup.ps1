<# 
.SYNOPSIS
    Setup script for AI PC Training environment

.DESCRIPTION
    Downloads and extracts the OpenVINO GenAI package, then sets up the environment.
    Run this script once after cloning the repository.

.EXAMPLE
    .\setup.ps1
#>

$ErrorActionPreference = "Stop"

$PACKAGE_URL = "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.0/windows/openvino_genai_windows_2026.0.0.0_x86_64.zip"
$PACKAGE_ZIP = "openvino_genai_windows_2026.0.0.0_x86_64.zip"
$PACKAGE_DIR = "openvino_genai_windows_2026.0.0.0_x86_64"

# Set Intel proxy if not already set
if (-not $env:http_proxy) {
    $env:http_proxy = "http://proxy-dmz.intel.com:912"
    $env:https_proxy = "http://proxy-dmz.intel.com:912"
    $env:no_proxy = ".intel.com,intel.com,localhost,127.0.0.1"
    Write-Host "Proxy configured: $env:http_proxy"
}

# Download OpenVINO GenAI package if not already present
if (-not (Test-Path $PACKAGE_DIR)) {
    if (-not (Test-Path $PACKAGE_ZIP)) {
        Write-Host "Downloading OpenVINO GenAI 2026.0.0 package (~223 MB)..."
        Invoke-WebRequest -Uri $PACKAGE_URL -OutFile $PACKAGE_ZIP -Proxy $env:http_proxy
        Write-Host "Download complete."
    }
    
    Write-Host "Extracting package..."
    Expand-Archive -Path $PACKAGE_ZIP -DestinationPath "." -Force
    Remove-Item $PACKAGE_ZIP
    Write-Host "Extraction complete."
} else {
    Write-Host "OpenVINO GenAI package already exists at: $PACKAGE_DIR"
}

# Set up OpenVINO environment
$setupvars = Join-Path $PSScriptRoot "$PACKAGE_DIR\setupvars.ps1"
if (Test-Path $setupvars) {
    Write-Host "`nSetting up OpenVINO environment..."
    & $setupvars
    Write-Host "`nEnvironment ready! You can now build the samples."
    Write-Host "  cd samples\cpp\visual_language_chat"
    Write-Host "  cmake -B build"
    Write-Host "  cmake --build build --config Release"
} else {
    Write-Host "ERROR: setupvars.ps1 not found at: $setupvars" -ForegroundColor Red
    exit 1
}
