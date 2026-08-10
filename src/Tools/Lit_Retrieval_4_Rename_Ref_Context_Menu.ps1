# Lit_Retrieval_4_Rename_Ref_Context_Menu.ps1
# Called by Windows Explorer context menu — once per selected file.
# Uses atomic file-creation lock + stability check to batch all
# selections into a single Python invocation in one PowerShell window.

param(
    [Parameter(Position = 0, Mandatory)]
    [string]$FilePath
)

$projectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$queueFile  = Join-Path $env:TEMP "rename_ref_queue.txt"
$lockFile   = Join-Path $env:TEMP "rename_ref_leader.lock"

# ── Step 1: Append this file path to the shared queue ────────────────
[System.IO.File]::AppendAllText($queueFile, "$FilePath`r`n",
    [System.Text.Encoding]::UTF8)

# ── Step 2: Clean up stale lock from a previous crashed run ──────────
if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).CreationTime
    if ($age.TotalSeconds -gt 30) {
        Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
    }
}

# ── Step 3: Try to become the leader (atomic file creation) ──────────
try {
    $fs = [System.IO.File]::Open($lockFile,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
} catch {
    # Lock file already exists — another instance is the leader. Exit.
    return
}

# ── Step 4 (leader only): Wait until the queue is stable ─────────────
# Keep checking until no new files are appended for two consecutive
# intervals, so that even staggered Explorer launches are captured.
try {
    $lastCount = 0
    $stableRounds = 0
    while ($stableRounds -lt 2) {
        Start-Sleep -Milliseconds 400
        $lines = @(
            [System.IO.File]::ReadAllLines($queueFile,
                [System.Text.Encoding]::UTF8) |
            Where-Object { $_.Trim() -ne '' }
        )
        if ($lines.Count -eq $lastCount -and $lastCount -gt 0) {
            $stableRounds++
        } else {
            $stableRounds = 0
        }
        $lastCount = $lines.Count
    }

    # Move queue to a unique file so a new queue can start immediately.
    $batchFile = Join-Path $env:TEMP "rename_ref_batch_$PID.txt"
    Move-Item -LiteralPath $queueFile -Destination $batchFile -Force

    if ($lines.Count -eq 0) {
        Remove-Item -LiteralPath $batchFile -Force -ErrorAction SilentlyContinue
        return
    }

    # Pass the file list via --from-file to avoid all quoting/escaping issues
    # with paths that contain spaces.
    $pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"
    $scriptPy  = Join-Path $projectDir "src\Tools\Lit_Retrieval_4_Rename_Ref.py"
    $cmd = @"
& '$pythonExe' '$scriptPy' --from-file '$batchFile'
Write-Host ''
Write-Host 'Press any key to close ...' -ForegroundColor Cyan
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@

    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd
}
finally {
    $fs.Close()
    Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
}
