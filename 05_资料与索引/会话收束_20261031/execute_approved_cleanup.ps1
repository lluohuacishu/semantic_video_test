# One-time execution of the explicitly approved file list. No recursive deletion.
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\lenovo\Documents\ChatGPT\语义通信'
$resolvedRoot = (Resolve-Path -LiteralPath $taskRoot).ProviderPath.TrimEnd('\')
if (-not [string]::Equals($resolvedRoot, $taskRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Unexpected workspace root'
}
$proposalFile = Join-Path $PSScriptRoot '待确认清理清单.json'
$approvalFile = Join-Path $PSScriptRoot '清理授权.json'
$preflightFile = Join-Path $PSScriptRoot '清理前只读校验.json'
$executionFile = Join-Path $PSScriptRoot '清理执行记录.json'
if (Test-Path -LiteralPath $executionFile) { throw 'An execution record already exists. Do not run twice.' }
$proposal = Get-Content -LiteralPath $proposalFile -Raw -Encoding UTF8 | ConvertFrom-Json
$approval = Get-Content -LiteralPath $approvalFile -Raw -Encoding UTF8 | ConvertFrom-Json
$preflight = Get-Content -LiteralPath $preflightFile -Raw -Encoding UTF8 | ConvertFrom-Json
$proposalHash = (Get-FileHash -LiteralPath $proposalFile -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $approval.approved -or $proposalHash -ne $approval.proposal_sha256 -or
    $proposalHash -ne $preflight.proposal_sha256 -or -not $preflight.passed) {
    throw 'Approval, immutable proposal, or preflight does not match'
}
if (($approval.approved_groups -join ',') -ne 'A1,A2,A3,A4' -or
    ($approval.preserve_groups -join ',') -ne 'B1,B2') { throw 'Only the approved A scope is supported' }
if ([DateTimeOffset]::Now - [DateTimeOffset]::Parse($preflight.checked_at) -gt [TimeSpan]::FromMinutes(5)) {
    throw 'Full read-only preflight is stale; run verify_cleanup_plan.py immediately before cleanup'
}

function Assert-ExactFile([string]$RelativePath) {
    if ([IO.Path]::IsPathRooted($RelativePath) -or $RelativePath.Split('\') -contains '..') {
        throw 'Invalid relative file path'
    }
    $absolute = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $RelativePath))
    if (-not $absolute.StartsWith($resolvedRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $absolute.Split('\') -contains '.git') { throw 'Target outside the named workspace' }
    $cursor = $absolute
    while ($true) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point: $cursor" }
        if ([string]::Equals($cursor, $resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    $file = Get-Item -LiteralPath $absolute -Force
    if ($file.PSIsContainer) { throw 'A proposed target is a directory' }
    return $absolute
}

$targets = @()
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($group in $proposal.groups) {
    if ($group.id -notin $approval.approved_groups) { continue }
    foreach ($entry in $group.files) {
        $absolute = Assert-ExactFile $entry.relative_path
        if (-not $seen.Add($absolute)) { throw 'Duplicate target' }
        if ($entry.relative_path -in $proposal.preserved_required_files) { throw 'Preserved input proposed for deletion' }
        foreach ($preservedRoot in $proposal.preserved_roots) {
            $protected = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $preservedRoot)).TrimEnd('\')
            if ($absolute.StartsWith($protected + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Target overlaps a preserved dependency tree'
            }
        }
        $file = Get-Item -LiteralPath $absolute -Force
        $expectedTicks = [long]([decimal]$entry.mtime_ns / 100) + [long]621355968000000000
        if ($file.Length -ne $entry.bytes -or $file.LastWriteTimeUtc.Ticks -ne $expectedTicks) {
            throw "Target metadata changed: $absolute"
        }
        if ((Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Target SHA changed: $absolute"
        }
        $targets += [PSCustomObject]@{ group=$group.id; relative_path=$entry.relative_path;
            absolute_path=$absolute; bytes=$entry.bytes; sha256=$entry.sha256; write_ticks=$file.LastWriteTimeUtc.Ticks }
    }
}
$logicalBytes = [long](($targets | Measure-Object -Property bytes -Sum).Sum)
if ($targets.Count -ne $approval.expected_file_count -or $logicalBytes -ne $approval.expected_logical_bytes) {
    throw 'Approved count or byte total does not match'
}
$deleted = [Collections.Generic.List[object]]::new()
$receipt = [ordered]@{
    approved=$true; user_answer=$approval.user_answer; approved_groups=$approval.approved_groups;
    proposal_sha256=$proposalHash; preflight_sha256=(Get-FileHash -LiteralPath $preflightFile -Algorithm SHA256).Hash.ToLowerInvariant();
    started_at=[DateTimeOffset]::Now.ToString('o'); finished_at=$null; workspace=$resolvedRoot;
    file_count_expected=$targets.Count; logical_bytes_expected=$logicalBytes;
    disk_free_before=[long](Get-PSDrive -Name C).Free; disk_free_after=$null;
    deleted_files=$deleted; passed=$false; error=$null;
    note='Only exact approved A files; no recursive deletion. B files and all protected dependencies retained. Free-space delta may include unrelated disk activity.'
}
try {
    # All hashes and paths have passed before the first removal.
    foreach ($target in $targets) {
        $absolute = Assert-ExactFile $target.relative_path
        $file = Get-Item -LiteralPath $absolute -Force
        if ($file.Length -ne $target.bytes -or $file.LastWriteTimeUtc.Ticks -ne $target.write_ticks) {
            throw "Target changed immediately before deletion: $absolute"
        }
        Remove-Item -LiteralPath $absolute -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $absolute) { throw "Target still exists: $absolute" }
        $deleted.Add([PSCustomObject]@{group=$target.group; relative_path=$target.relative_path;
            bytes=$target.bytes; sha256=$target.sha256; deleted_at=[DateTimeOffset]::Now.ToString('o')})
    }
    $receipt.passed = $deleted.Count -eq $approval.expected_file_count
} catch {
    $receipt.error = $_.Exception.Message
    throw
} finally {
    $receipt.finished_at = [DateTimeOffset]::Now.ToString('o')
    $receipt.disk_free_after = [long](Get-PSDrive -Name C).Free
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $executionFile -Encoding UTF8
}
Write-Output "Deleted $($deleted.Count) explicitly approved A files; logical bytes $logicalBytes. B retained."
