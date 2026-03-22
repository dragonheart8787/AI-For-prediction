#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練資料儲存庫：防遺忘機制（經驗回放）

- 儲存每次訓練的 (X, y, task_id, domain, feature_names)
- 增量訓練時，從舊任務採樣混合，避免 catastrophic forgetting
- 支援全任務訓練與通用模型

單次訓練「產物目錄／metrics／config」請優先使用 ``artifact_registry`` +
``artifacts/{task}/{run_id}/``（與本 JSONL 記憶庫分離）。
"""
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = "data/training_memory.jsonl"
DEFAULT_MAX_SAMPLES_PER_TASK = 500
DEFAULT_REPLAY_RATIO = 0.3  # 增量訓練時，30% 來自舊任務


class TrainingDataStore:
    """訓練資料記憶庫，用於經驗回放防遺忘"""

    def __init__(
        self,
        path: str = DEFAULT_STORE_PATH,
        max_samples_per_task: int = DEFAULT_MAX_SAMPLES_PER_TASK,
    ) -> None:
        self.path = path
        self.max_samples_per_task = max_samples_per_task
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """載入已儲存的資料"""
        self._entries = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._entries.append(json.loads(line))
            except Exception as e:
                logger.warning("載入訓練記憶失敗: %s", e)

    def save(self) -> None:
        """儲存到磁碟"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_id: str,
        domain: str,
        feature_names: List[str],
    ) -> None:
        """新增訓練樣本到記憶庫（採樣以控制大小）"""
        n = X.shape[0]
        if n > self.max_samples_per_task:
            idx = random.sample(range(n), self.max_samples_per_task)
            X = X[idx]
            y = y[idx]

        # 移除該任務舊資料，再加入新資料
        self._entries = [e for e in self._entries if e.get("task_id") != task_id]

        entry = {
            "task_id": task_id,
            "domain": domain,
            "feature_names": feature_names,
            "X": X.tolist(),
            "y": y.tolist(),
            "n_samples": len(y),
        }
        self._entries.append(entry)
        self.save()
        logger.info("訓練記憶已更新: %s (%d 樣本)", task_id, len(y))

    def get_replay_samples(
        self,
        exclude_task_id: Optional[str] = None,
        n_samples: int = 100,
        domains: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        取得經驗回放樣本（用於增量訓練時混合，避免遺忘）。
        回傳 (X_replay, y_replay, feature_names)。
        若特徵維度不一致，會嘗試對齊或跳過。
        """
        candidates = [
            e for e in self._entries
            if (exclude_task_id is None or e.get("task_id") != exclude_task_id)
            and (domains is None or e.get("domain") in domains)
        ]
        if not candidates:
            return np.zeros((0, 0)), np.zeros(0), []

        # 隨機選任務，再從中採樣
        all_X, all_y = [], []
        feature_names: List[str] = []
        per_task = max(1, n_samples // len(candidates))

        for e in random.sample(candidates, min(len(candidates), 10)):
            X_arr = np.array(e["X"], dtype=float)
            y_arr = np.array(e["y"], dtype=float)
            if X_arr.size == 0 or y_arr.size == 0:
                continue
            n = min(per_task, X_arr.shape[0])
            idx = random.sample(range(X_arr.shape[0]), n)
            all_X.append(X_arr[idx])
            all_y.append(y_arr[idx])
            if not feature_names and e.get("feature_names"):
                feature_names = e["feature_names"]

        if not all_X:
            return np.zeros((0, 0)), np.zeros(0), []

        # 對齊特徵維度：取最大維度，不足的填 0
        max_cols = max(X.shape[1] for X in all_X)
        aligned = []
        for X in all_X:
            if X.shape[1] < max_cols:
                pad = np.zeros((X.shape[0], max_cols - X.shape[1]))
                X = np.hstack([X, pad])
            aligned.append(X)
        X_replay = np.vstack(aligned)
        y_replay = np.concatenate(all_y)
        return X_replay, y_replay, feature_names

    def get_all_for_universal(
        self,
        canonical_features: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        取得所有任務的資料，對齊到 canonical 特徵空間，用於訓練通用模型。
        """
        if not self._entries:
            return np.zeros((0, 0)), np.zeros(0), canonical_features or []

        canonical = canonical_features or []
        all_X, all_y = [], []

        for e in self._entries:
            X = np.array(e["X"], dtype=float)
            y = np.array(e["y"], dtype=float)
            names = e.get("feature_names", [])
            if canonical:
                # 對齊到 canonical：按 canonical 順序，沒有的填 0
                X_aligned = np.zeros((X.shape[0], len(canonical)))
                for i, cf in enumerate(canonical):
                    if cf in names:
                        j = names.index(cf)
                        X_aligned[:, i] = X[:, j]
                X = X_aligned
            all_X.append(X)
            all_y.append(y)

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        return X_all, y_all, canonical or (self._entries[0].get("feature_names", []))

    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出已儲存的任務"""
        return [
            {"task_id": e["task_id"], "domain": e["domain"], "n_samples": e["n_samples"]}
            for e in self._entries
        ]

    def clear(self) -> None:
        """清空記憶"""
        self._entries = []
        self.save()


def merge_with_replay(
    X_new: np.ndarray,
    y_new: np.ndarray,
    store: TrainingDataStore,
    task_id: str,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    將新資料與回放樣本混合，用於防遺忘訓練。
    回傳 (X_merged, y_merged)。
    """
    n_new = X_new.shape[0]
    n_replay_target = int(n_new * replay_ratio / (1 - replay_ratio)) if replay_ratio < 1 else n_new

    X_replay, y_replay, _ = store.get_replay_samples(exclude_task_id=task_id, n_samples=n_replay_target)
    if X_replay.shape[0] == 0:
        return X_new, y_new

    # 對齊特徵維度
    n_cols_new, n_cols_replay = X_new.shape[1], X_replay.shape[1]
    max_cols = max(n_cols_new, n_cols_replay)
    if n_cols_new < max_cols:
        X_new = np.hstack([X_new, np.zeros((X_new.shape[0], max_cols - n_cols_new))])
    if n_cols_replay < max_cols:
        X_replay = np.hstack([X_replay, np.zeros((X_replay.shape[0], max_cols - n_cols_replay))])

    X_merged = np.vstack([X_new, X_replay])
    y_merged = np.concatenate([y_new, y_replay])
    # 打亂順序
    idx = np.random.permutation(len(y_merged))
    return X_merged[idx], y_merged[idx]
