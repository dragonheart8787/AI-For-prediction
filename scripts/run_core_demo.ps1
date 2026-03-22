# 乾淨環境最小驗證（於專案根目錄執行）
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m pip install -r requirements-core.txt
python -m pytest tests/test_unified_predict.py -q
python crawler_train_pipeline.py --help
python launch_predict_service.py --help
Write-Host "OK: core demo checks passed."
