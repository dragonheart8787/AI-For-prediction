# 選配：Optuna + PyTorch（需另裝依賴）
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m pip install -r requirements-core.txt -r requirements-automl.txt
python -m pytest tests/test_automl_torch.py -q
Write-Host "OK: automl/torch smoke passed (skipped if deps missing)."
