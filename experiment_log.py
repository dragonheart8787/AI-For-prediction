#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實驗／訓練紀錄（輕量版，無需 MLflow）。

v0.9.5 新功能：
- log_epoch()     — 逐 epoch 指標紀錄（深度模型訓練進度）
- read_experiments() — 載入並過濾 JSONL 紀錄
- compare_runs()  — 對多個 run_id 指標做差異比較
- summarize_experiments() — 統計各模型平均表現
- tag 支援        — log_experiment / log_training_event 均可附加 tags
- 單檔過大時自動輪替（rotation）
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# 超過此位元組數則輪替（預設 50MB）
MAX_JSONL_BYTES = int(os.environ.get("PREDICT_AI_EXPERIMENT_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
_DEFAULT_LOG_PATH = "data/experiment_runs.jsonl"
_DEFAULT_EPOCH_LOG_PATH = "data/epoch_logs.jsonl"


# ---------------------------------------------------------------------------
# 內部工具
# ---------------------------------------------------------------------------

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


def _append_jsonl(rec: Dict[str, Any], path: str) -> str:
    _rotate_if_needed(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ap = os.path.abspath(path)
    with open(ap, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return ap


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 寫入 API
# ---------------------------------------------------------------------------

def log_experiment(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    path: str = _DEFAULT_LOG_PATH,
    *,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    追加一筆實驗紀錄（相容舊 API：使用 ``event`` 作為事件名）。

    Parameters
    ----------
    tags : 可選標籤，如 {"env": "ci", "version": "0.9.5"}
    """
    payload = payload or {}
    rec: Dict[str, Any] = {
        "ts_unix": time.time(),
        "ts_iso": _now_iso(),
        "event": event,
        **payload,
    }
    if tags:
        rec["tags"] = tags
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
    path: str = _DEFAULT_LOG_PATH,
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    固定 schema 的訓練事件（建議新程式優先使用）。

    必填語意：event_type, run_id, task_name, model_*, row/feature counts, status
    """
    rec: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": _now_iso(),
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
    if tags:
        rec["tags"] = tags
    if extra:
        rec.update(extra)
    return _append_jsonl(rec, path)


def log_epoch(
    *,
    run_id: str,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    lr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    path: str = _DEFAULT_EPOCH_LOG_PATH,
) -> str:
    """
    逐 epoch 指標紀錄（深度模型訓練進度追蹤）。

    Parameters
    ----------
    run_id     : 與 log_training_event 的 run_id 對應
    epoch      : 當前 epoch 數（0-based）
    train_loss : 訓練損失
    val_loss   : 驗證損失（可選）
    lr         : 當前學習率（可選）
    """
    rec: Dict[str, Any] = {
        "event_type": "epoch",
        "ts_unix": time.time(),
        "ts_iso": _now_iso(),
        "run_id": run_id,
        "epoch": epoch,
        "train_loss": float(train_loss),
    }
    if val_loss is not None:
        rec["val_loss"] = float(val_loss)
    if lr is not None:
        rec["lr"] = float(lr)
    if extra:
        rec.update(extra)
    return _append_jsonl(rec, path)


# ---------------------------------------------------------------------------
# 讀取 / 分析 API
# ---------------------------------------------------------------------------

def read_experiments(
    path: str = _DEFAULT_LOG_PATH,
    *,
    event_type: Optional[str] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    model_trained: Optional[str] = None,
    tag_filter: Optional[Dict[str, str]] = None,
    since_ts: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    讀取並過濾 JSONL 實驗紀錄。

    Parameters
    ----------
    tag_filter : {"key": "value"} — 匹配所有指定 tag
    since_ts   : Unix 時間戳（只取此時間之後的紀錄）
    limit      : 最多返回幾筆

    Returns
    -------
    list of dicts，按 ts_unix 升序排列
    """
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return []
    records: List[Dict[str, Any]] = []
    with open(ap, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 過濾條件
            if event_type and rec.get("event_type") != event_type:
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            if status and rec.get("status") != status:
                continue
            if model_trained and rec.get("model_trained") != model_trained:
                continue
            if since_ts and rec.get("ts_unix", 0) < since_ts:
                continue
            if tag_filter:
                rec_tags = rec.get("tags") or {}
                if not all(rec_tags.get(k) == v for k, v in tag_filter.items()):
                    continue
            records.append(rec)

    records.sort(key=lambda r: r.get("ts_unix", 0))
    if limit:
        records = records[-limit:]
    return records


def compare_runs(
    run_ids: List[str],
    path: str = _DEFAULT_LOG_PATH,
    *,
    metrics_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    比較多個 run_id 的訓練指標。

    Returns
    -------
    {
        "runs": {run_id: {...metrics...}},
        "best_run": {"run_id": ..., "metric": ..., "value": ...},
        "comparison_keys": [...],
    }
    """
    default_keys = ["train_rmse", "train_mae", "train_r2", "rmse", "mae", "r2"]
    keys = metrics_keys or default_keys
    run_data: Dict[str, Dict[str, Any]] = {}

    for rid in run_ids:
        recs = read_experiments(path, run_id=rid, status="success")
        if not recs:
            run_data[rid] = {}
            continue
        # 取最後一筆（通常是完成紀錄）
        latest = recs[-1]
        m = latest.get("metrics") or {}
        run_data[rid] = {
            k: m.get(k) for k in keys if k in m
        }
        run_data[rid]["model_trained"] = latest.get("model_trained", "?")
        run_data[rid]["row_count"] = latest.get("row_count", 0)
        run_data[rid]["duration_ms"] = latest.get("duration_ms")

    # 找最佳（以 train_rmse 或 rmse 為主，越小越好）
    best_key = "train_rmse" if "train_rmse" in keys else "rmse"
    best_run_id: Optional[str] = None
    best_val = float("inf")
    for rid, d in run_data.items():
        v = d.get(best_key)
        if v is not None and float(v) < best_val:
            best_val = float(v)
            best_run_id = rid

    return {
        "runs": run_data,
        "best_run": {
            "run_id": best_run_id,
            "metric": best_key,
            "value": best_val if best_run_id else None,
        },
        "comparison_keys": keys,
    }


def summarize_experiments(
    path: str = _DEFAULT_LOG_PATH,
    *,
    group_by: str = "model_trained",
) -> Dict[str, Any]:
    """
    依指定欄位分組，統計各組的平均 / 最佳 RMSE、MAE 等。

    Parameters
    ----------
    group_by : 分組欄位，如 "model_trained" | "task_name"

    Returns
    -------
    {
        "groups": {
            "xgboost": {"count": 5, "mean_rmse": ..., "best_rmse": ...},
            ...
        }
    }
    """
    recs = read_experiments(path, status="success")
    groups: Dict[str, List[float]] = {}
    for rec in recs:
        key = str(rec.get(group_by, "unknown"))
        m = rec.get("metrics") or {}
        rmse = m.get("train_rmse") or m.get("rmse")
        if rmse is not None:
            groups.setdefault(key, []).append(float(rmse))

    result: Dict[str, Any] = {"groups": {}}
    for key, values in groups.items():
        result["groups"][key] = {
            "count": len(values),
            "mean_rmse": float(sum(values) / len(values)),
            "best_rmse": float(min(values)),
            "worst_rmse": float(max(values)),
        }
    return result


def read_epoch_logs(
    run_id: str,
    path: str = _DEFAULT_EPOCH_LOG_PATH,
) -> List[Dict[str, Any]]:
    """
    讀取指定 run_id 的逐 epoch 紀錄，按 epoch 排序。
    """
    recs = read_experiments(path, run_id=run_id)
    recs = [r for r in recs if r.get("event_type") == "epoch"]
    recs.sort(key=lambda r: r.get("epoch", 0))
    return recs
