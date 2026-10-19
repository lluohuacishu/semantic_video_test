# One-time, same-volume workspace migration. All paths come from the reviewed manifest.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PSScriptRoot '工作区整理清单_20261019.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.status -eq 'completed') {
    Write-Output 'This migration is complete. Read the manifest and execution journal; do not repeat it.'
    exit 0
}
$workspace = [IO.Path]::GetFullPath($manifest.workspace).TrimEnd('\')
$prefix = $workspace + '\'
function Checked-Path([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path leaves workspace: $absolute"
    }
    return $absolute
}
function File-Stats([string]$path) {
    $item = Get-Item -LiteralPath $path -Force
    if (-not $item.PSIsContainer) { return @{ files = 1; bytes = $item.Length } }
    $measure = Get-ChildItem -LiteralPath $path -Force -Recurse -File | Measure-Object -Property Length -Sum
    return @{ files = $measure.Count; bytes = [long]$measure.Sum }
}
foreach ($entry in $manifest.entries) {
    $source = Checked-Path $entry.source
    $destination = Checked-Path $entry.destination
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    $sourceItem = Get-Item -LiteralPath $source -Force
    if (Test-Path -LiteralPath $destination) {
        $destinationItem = Get-Item -LiteralPath $destination -Force
        if ($destinationItem.LinkType -eq 'Junction' -and [IO.Path]::GetFullPath($destinationItem.Target) -eq $source) { continue }
    }
    if ($sourceItem.LinkType -eq 'Junction' -and (Test-Path -LiteralPath $destination)) {
        if ([IO.Path]::GetFullPath($sourceItem.Target) -ne $destination) { throw 'Unexpected compatibility target' }
        continue
    }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    if ((Get-Item -LiteralPath $source -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Source is already a compatibility link: $source"
    }
}
if (-not $Apply) {
    Write-Output "Validated $($manifest.entries.Count) moves. Use -Apply to execute."
    exit 0
}
$journalPath = Checked-Path '03_历史归档/管理元数据/整理执行记录_20261019.json'
New-Item -ItemType Directory -Path (Split-Path $journalPath) -Force | Out-Null
$journal = [Collections.Generic.List[object]]::new()
foreach ($entry in $manifest.entries) {
    $source = Checked-Path $entry.source
    $destination = Checked-Path $entry.destination
    $alreadyMoved = (Get-Item -LiteralPath $source -Force).LinkType -eq 'Junction'
    $canonicalLink = (Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination -Force).LinkType -eq 'Junction'
    $placement = if ($canonicalLink) { 'canonical_junction_to_original' } else { 'moved_with_legacy_junction' }
    $before = if ($alreadyMoved) { File-Stats $destination } else { File-Stats $source }
    if (-not $alreadyMoved -and -not $canonicalLink) {
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
        # Atomic directory rename also handles hidden .git metadata without a partial recursive move.
        if ($entry.kind -eq 'directory') {
            try { [IO.Directory]::Move($source, $destination) }
            catch {
                $baseError = $_.Exception.GetBaseException()
                if ($baseError.HResult -notin @(-2147024891, -2147024864) -or -not $entry.compatibility_junction -or (Test-Path -LiteralPath $destination)) { throw }
                # Keep one physical copy when Windows denies a rename; make it available at the canonical path.
                New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
                $canonicalLink = $true
                $placement = 'canonical_junction_to_original'
            }
        }
        else { [IO.File]::Move($source, $destination) }
    }
    $after = File-Stats $destination
    if ($before.files -ne $after.files -or $before.bytes -ne $after.bytes) {
        throw "Inventory changed during move: $source -> $destination"
    }
    if ($entry.compatibility_junction -and -not $alreadyMoved -and -not $canonicalLink) {
        New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
        if ((Get-Item -LiteralPath $source -Force).LinkType -ne 'Junction') {
            throw "Compatibility junction failed: $source"
        }
    }
    $journal.Add(@{source=$entry.source; destination=$entry.destination; before=$before; after=$after; compatibility_junction=$entry.compatibility_junction; already_recovered=$alreadyMoved; placement=$placement; verified=$true})
    ConvertTo-Json -InputObject @($journal.ToArray()) -Depth 10 | Set-Content -LiteralPath $journalPath -Encoding utf8
    Write-Output "Moved and verified: $($entry.source)"
}
# Keep compatibility paths out of the normal top-level view. /L targets the link itself.
$legacyWork = Checked-Path 'work'
(Get-Item -LiteralPath $legacyWork -Force).Attributes = (Get-Item -LiteralPath $legacyWork -Force).Attributes -bor [IO.FileAttributes]::Hidden
foreach ($relative in @('outputs', '语义通信_整合归档_2026-10-17')) {
    & attrib.exe '+H' '/L' (Checked-Path $relative)
    if ($LASTEXITCODE -ne 0) { throw "Failed to hide legacy junction: $relative" }
}
Write-Output "Completed $($journal.Count) verified moves. The new canonical repository is $($manifest.authoritative_repository)."
