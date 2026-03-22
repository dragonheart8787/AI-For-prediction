#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實驗／訓練紀錄（輕量版，無需 MLflow）
- 事件 schema 固定欄位，便於過濾與與 artifacts 對齊
- 單檔過大時自動輪替（rotation）
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# 超過此位元組數則輪替（預設 50MB）
MAX_JSONL_BYTES = int(os.environ.get("PREDICT_AI_EXPERIMENT_LOG_MAX_BYTES", str(50 * 1024 * 1024)))


def _rotate_if_needed(path: str) -> None:
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return
    try:
        if os.path.getsize(ap) < MAX_JSONL_BYTES:
            return
    except OSError:
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated = f"{ap}.{ts}.bak"
    try:
        shutil.move(ap, rotated)
    except OSError:
        pass


def log_experiment(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    path: str = "data/experiment_runs.jsonl",
) -> str:
    """
    追加一筆實驗紀錄（相容舊 API：使用 ``event`` 作為事件名）。
    """
    payload = payload or {}
    rec: Dict[str, Any] = {
        "ts_unix": time.time(),
        "ts_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event,
        **payload,
    }
    return _append_jsonl(rec, path)


def log_training_event(
    *,
    event_type: str,
    run_id: str,
    task_name: str,
    model_requested: str,
    model_trained: str,
    row_count: int,
    feature_count: int,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error_type: Optional[str] = None,
    duration_ms: Optional[int] = None,
    parent_run_id: Optional[str] = None,
    path: str = "data/experiment_runs.jsonl",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    固定 schema 的訓練事件（建議新程式優先使用）。

    必填語意：event_type, run_id, task_name, model_*, row/feature counts, status
    """
    rec: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ts_unix": time.time(),
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "task_name": task_name,
        "model_requested": model_requested,
        "model_trained": model_trained,
        "row_count": int(row_count),
        "feature_count": int(feature_count),
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "status": status,
        "error_type": error_type,
        "duration_ms": duration_ms,
    }
    if extra:
        rec.update(extra)
    return _append_jsonl(rec, path)


def _append_jsonl(rec: Dict[str, Any], path: str) -> str:
    _rotate_if_needed(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ap = os.path.abspath(path)
    with open(ap, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return ap
