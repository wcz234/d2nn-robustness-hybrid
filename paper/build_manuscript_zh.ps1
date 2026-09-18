param(
    [string]$Typst = "typst"
)

$ErrorActionPreference = "Stop"
$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PaperDir
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python (Join-Path $PaperDir "build_typst_zh.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Typst compile --root $ProjectRoot `
    (Join-Path $PaperDir "manuscript_zh.typ") `
    (Join-Path $PaperDir "MANUSCRIPT_ZH.pdf")
exit $LASTEXITCODE
