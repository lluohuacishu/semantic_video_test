param([string]$Config = $env:GLC_CONFIG, [switch]$Check)
$ErrorActionPreference = 'Stop'
try {
    $specifiedConfig = [bool]$Config
    if (-not $Config) { $Config = Join-Path $PSScriptRoot 'runtime.local.json' }
    $explicitConfig = Test-Path -LiteralPath $Config -PathType Leaf
    if ($specifiedConfig -and -not $explicitConfig) { throw 'Configuration file does not exist.' }
    if ($explicitConfig) { $resolvedConfig = (Resolve-Path -LiteralPath $Config).Path }
    $pythonPath = $env:GLC_PYTHON
    if (-not $pythonPath -and $explicitConfig) {
        $resolvedConfig = (Resolve-Path -LiteralPath $Config).Path
        $settings = Get-Content -LiteralPath $resolvedConfig -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($settings.runtime.python) { $pythonPath = [Environment]::ExpandEnvironmentVariables($settings.runtime.python) }
        if ($pythonPath -and -not [IO.Path]::IsPathRooted($pythonPath)) {
            $pythonPath = Join-Path (Split-Path -Parent $resolvedConfig) $pythonPath
        }
    }
    if (-not $pythonPath) {
        $localPython = Join-Path $PSScriptRoot '../../../运行环境/GLC_Python环境/Scripts/python.exe'
        if (Test-Path -LiteralPath $localPython -PathType Leaf) { $pythonPath = $localPython }
        else { $pythonPath = (Get-Command python -ErrorAction Stop).Source }
    }
    $arguments = @('-B', '-u', (Join-Path $PSScriptRoot 'c3_server.py'))
    if ($explicitConfig) { $arguments += @('--config', $resolvedConfig) }
    elseif ($env:GLC_CONFIG) { throw 'GLC_CONFIG file does not exist.' }
    if ($Check) { $arguments += '--check' } else { $arguments += '--open' }
    & $pythonPath @arguments
    exit $LASTEXITCODE
} catch {
    Write-Host ('Launch failed: ' + $_.Exception.Message)
    Write-Host 'Copy runtime.example.json to runtime.local.json and configure your local paths, or set GLC_PYTHON.'
    exit 1
}
