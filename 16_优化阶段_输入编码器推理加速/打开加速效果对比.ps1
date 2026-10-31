param([int]$Port=8773, [switch]$NoBrowser)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$pythonPath=$env:GLC_PYTHON
$webRoot=Get-ChildItem -LiteralPath $repoRoot -Directory -Filter '08_*' | Select-Object -First 1
if (-not $pythonPath -and $webRoot) {
    $configPath=Join-Path $webRoot.FullName 'runtime.local.json'
    if (Test-Path -LiteralPath $configPath) {
        $config=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $pythonPath=$config.runtime.python
        if ($pythonPath -and -not [IO.Path]::IsPathRooted($pythonPath)) { $pythonPath=Join-Path $webRoot.FullName $pythonPath }
    }
}
if (-not $pythonPath) { $pythonPath=(Get-Command python -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'GLC Python was not found. Check runtime.local.json.' }
$entryArgs=@('-X','utf8',(Join-Path $PSScriptRoot 'open_viewer.py'),'--port',[string]$Port)
if ($NoBrowser) { $entryArgs+='--no-browser' }
& $pythonPath @entryArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
