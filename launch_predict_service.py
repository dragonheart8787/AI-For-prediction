#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一鍵啟動預測 HTTP 服務（載入 crawler_train_pipeline 產生的 .pkl）。

範例：
  python crawler_train_pipeline.py stock_price_next --model linear
  python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765

另可設環境變數：UNIFIED_MODEL_PATH
"""
import sys
import os

# 專案根目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from model_serving.unified_http_service import run_server

    run_server()
