param([string]$Python = 'py')
$ErrorActionPreference = 'Stop'
try {
    if ($Python -eq 'py') { & py -3.12 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' }
    else { & $Python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' }
    if ($LASTEXITCODE -ne 0) { throw 'This receiver package requires Python 3.12.' }
    $bundleRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    if (-not (Test-Path -LiteralPath (Join-Path $bundleRoot 'bundle_manifest.json'))) {
        throw 'Run this installer from a generated C5 receiver bundle, not the development checkout.'
    }
    $verifyArgs = @('-X','utf8','-B',(Join-Path $PSScriptRoot 'package_c5_receiver.py'),'--verify',$bundleRoot)
    if ($Python -eq 'py') { & py -3.12 @verifyArgs } else { & $Python @verifyArgs }
    if ($LASTEXITCODE -ne 0) { throw 'Receiver bundle verification failed before environment creation.' }
    $runtimeDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../runtime'))
    $environmentDir = Join-Path $runtimeDir '.venv'
    if (Test-Path -LiteralPath $environmentDir) {
        throw 'The target environment already exists. Check it manually; this installer does not overwrite environments.'
    }
    if ($Python -eq 'py') { & py -3.12 -m venv $environmentDir }
    else { & $Python -m venv $environmentDir }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
    $pythonExe = Join-Path $environmentDir 'Scripts/python.exe'
    & $pythonExe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed; preserve logs and inspect the new environment.' }
    & $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements-c5-receiver-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'GLC dependency installation failed.' }
    & $pythonExe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Installed dependency consistency check failed.' }
    & $pythonExe -X utf8 -B (Join-Path $PSScriptRoot 'c5_receiver.py') --check
    exit $LASTEXITCODE
} catch {
    Write-Host ('Setup failed: ' + $_.Exception.Message)
    exit 1
}
