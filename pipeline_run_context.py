#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練回合脈絡：run_id、config hash、git commit（可選），供實驗追溯。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RunMetadata:
    run_id: str
    task_name: str
    model_type: str
    feature_count: int
    row_count: int
    train_time_sec: float
    dependency_mode: str  # "core" | "automl"
    data_sources: List[str]
    artifact_paths: Dict[str, str]
    config_hash: str
    git_commit: Optional[str] = None
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def try_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def hash_config_blob(obj: Any) -> str:
    """對設定／任務 dict 做穩定短 hash。"""
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
