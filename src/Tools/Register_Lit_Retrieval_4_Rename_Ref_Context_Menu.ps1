# Register_Lit_Retrieval_4_Rename_Ref_Context_Menu.ps1
# Run as Administrator to register the "Rename Ref" context-menu entry
# for .pdf, .epub, and .djvu files.
#
# Usage:
#   .\Register_Lit_Retrieval_4_Rename_Ref_Context_Menu.ps1            # register
#   .\Register_Lit_Retrieval_4_Rename_Ref_Context_Menu.ps1 -Unregister # remove

param(
    [switch]$Unregister
)

$launcherScript = Join-Path $PSScriptRoot "Lit_Retrieval_4_Rename_Ref_Context_Menu.ps1"
$menuLabel      = "LYH Rename Ref"
$verbName       = "RenameRef"

# File extensions to register the context menu for.
$extensions = @(".pdf", ".epub", ".djvu")

# The command Explorer will execute (once per selected file).
# -WindowStyle Hidden so the collector scripts are invisible; only the
# final PowerShell window (started by the leader) is visible to the user.
$command = "powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcherScript`" `"%1`""

foreach ($ext in $extensions) {
    $keyPath = "HKLM:\SOFTWARE\Classes\SystemFileAssociations\$ext\shell\$verbName"

    if ($Unregister) {
        if (Test-Path $keyPath) {
            Remove-Item -LiteralPath $keyPath -Recurse -Force
            Write-Host "Removed: $keyPath" -ForegroundColor Yellow
        } else {
            Write-Host "Not found (skip): $keyPath" -ForegroundColor DarkGray
        }
        continue
    }

    # ── Create / update the verb key ───────────────────────────────────
    if (-not (Test-Path $keyPath)) {
        New-Item -Path $keyPath -Force | Out-Null
    }
    Set-ItemProperty -LiteralPath $keyPath -Name "(Default)" -Value $menuLabel

    # ── Create / update the command subkey ─────────────────────────────
    $cmdKey = "$keyPath\command"
    if (-not (Test-Path $cmdKey)) {
        New-Item -Path $cmdKey -Force | Out-Null
    }
    Set-ItemProperty -LiteralPath $cmdKey -Name "(Default)" -Value $command

    Write-Host "Registered: $keyPath" -ForegroundColor Green
    Write-Host "  Command : $command"
}

if (-not $Unregister) {
    Write-Host "`nDone. The '$menuLabel' option will appear when you right-click $($extensions -join ', ') files." -ForegroundColor Cyan
} else {
    Write-Host "`nDone. Context menu entries removed." -ForegroundColor Cyan
}
