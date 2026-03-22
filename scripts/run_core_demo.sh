#!/usr/bin/env bash
# 乾淨環境最小驗證（請在專案根目錄執行）
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements-core.txt
python -m pytest tests/test_unified_predict.py -q
python crawler_train_pipeline.py --help
python launch_predict_service.py --help
echo "OK: core demo checks passed."
