param([ValidateSet('sender','receiver')][string]$Role = 'sender', [string]$Config, [switch]$Check)
$ErrorActionPreference = 'Stop'
try {
    $supplied = [bool]$Config
    if (-not $Config) { $Config = Join-Path $PSScriptRoot ('c5_' + $Role + '.local.json') }
    $present = Test-Path -LiteralPath $Config -PathType Leaf
    if ($supplied -and -not $present) { throw 'Endpoint configuration does not exist.' }
    $pythonPath = $env:GLC_PYTHON
    $runtimeConfig = Join-Path $PSScriptRoot 'runtime.local.json'
    if ($present) {
        $Config = (Resolve-Path -LiteralPath $Config).Path
        $settings = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($settings.base_config) {
            $runtimeConfig = [Environment]::ExpandEnvironmentVariables($settings.base_config)
            if (-not [IO.Path]::IsPathRooted($runtimeConfig)) {
                $runtimeConfig = Join-Path (Split-Path -Parent $Config) $runtimeConfig
            }
        }
        if (-not $pythonPath -and $settings.runtime.python) {
            $pythonPath = [Environment]::ExpandEnvironmentVariables($settings.runtime.python)
            if (-not [IO.Path]::IsPathRooted($pythonPath)) { $pythonPath = Join-Path (Split-Path -Parent $Config) $pythonPath }
        }
    }
    if (-not $pythonPath -and (Test-Path -LiteralPath $runtimeConfig -PathType Leaf)) {
        $runtimeConfig = (Resolve-Path -LiteralPath $runtimeConfig).Path
        $runtime = Get-Content -LiteralPath $runtimeConfig -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($runtime.runtime.python) {
            $pythonPath = [Environment]::ExpandEnvironmentVariables($runtime.runtime.python)
            if (-not [IO.Path]::IsPathRooted($pythonPath)) { $pythonPath = Join-Path (Split-Path -Parent $runtimeConfig) $pythonPath }
        }
    }
    if (-not $pythonPath) {
        $candidate = Join-Path $PSScriptRoot '../../../运行环境/GLC_Python环境/Scripts/python.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $pythonPath = $candidate }
        else { $pythonPath = (Get-Command python -ErrorAction Stop).Source }
    }
    $arguments = @('-X','utf8','-B','-u',(Join-Path $PSScriptRoot ('c5_' + $Role + '.py')))
    if ($present) { $arguments += @('--config',$Config) }
    if ($Check) { $arguments += '--check' } else { $arguments += '--open' }
    & $pythonPath @arguments
    exit $LASTEXITCODE
} catch {
    Write-Host ('Launch failed: ' + $_.Exception.Message)
    Write-Host 'Configure runtime.local.json and c5_sender.local.json / c5_receiver.local.json.'
    exit 1
}
