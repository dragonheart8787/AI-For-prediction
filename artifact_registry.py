#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實驗產物目錄約定：artifacts/{task}/{run_id}/ 下 config、metrics、summary 等。
（與 TrainingDataStore 經驗回放 JSONL 分離）
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def default_artifacts_root() -> str:
    return os.environ.get("PREDICT_AI_ARTIFACTS_ROOT", "artifacts")


def ensure_run_dir(task_name: str, run_id: str, base: Optional[str] = None) -> str:
    root = base or default_artifacts_root()
    safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)[:120]
    path = os.path.join(root, safe_task, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_run_bundle(
    run_dir: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    feature_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """寫入單次訓練建議檔名，回傳路徑映射。"""
    paths: Dict[str, str] = {}
    if config is not None:
        p = os.path.join(run_dir, "config.json")
        write_json(p, config)
        paths["config"] = p
    if metrics is not None:
        p = os.path.join(run_dir, "metrics.json")
        write_json(p, metrics)
        paths["metrics"] = p
    if summary is not None:
        p = os.path.join(run_dir, "summary.json")
        write_json(p, summary)
        paths["summary"] = p
    if feature_manifest is not None:
        p = os.path.join(run_dir, "feature_manifest.json")
        write_json(p, feature_manifest)
        paths["feature_manifest"] = p
    return paths
