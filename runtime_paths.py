#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專案路徑解析：禁止硬編碼本機絕對路徑，統一由環境變數或 CLI 覆寫。
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """此檔所在目錄即專案根（假設 runtime_paths.py 位於 repo 根）。"""
    return Path(__file__).resolve().parent


def resolve_models_dir(cli_override: str | None = None) -> Path:
    """模型目錄：`--models-dir` > `PREDICT_AI_MODELS_DIR` > `models/`。"""
    base = cli_override or os.environ.get("PREDICT_AI_MODELS_DIR") or "models"
    return Path(base).expanduser().resolve()


def resolve_data_dir(cli_override: str | None = None) -> Path:
    """資料根目錄：`--data-dir` > `PREDICT_AI_DATA_DIR` > `data/`。"""
    base = cli_override or os.environ.get("PREDICT_AI_DATA_DIR") or "data"
    return Path(base).expanduser().resolve()


def resolve_relative_to_cwd(path_str: str) -> Path:
    """將設定檔中的相對路徑解析為絕對路徑（相對於 cwd）。"""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()
