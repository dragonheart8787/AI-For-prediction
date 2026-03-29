#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""啟動本機預測操作介面（瀏覽器）。等同 python -m model_serving.unified_http_service"""
from model_serving.unified_http_service import run_server

if __name__ == "__main__":
    run_server()
