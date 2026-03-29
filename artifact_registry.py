#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實驗產物目錄 + 模型 Registry。

v0.9.5 新功能：
- ModelRegistry 類別：正式模型版本管理
  - register_model()  — 登記新版本（自動遞增）
  - list_models()     — 列出所有已登記模型（可過濾）
  - load_registered_model() — 按名稱 + 版本載入
  - promote_model()   — 升階（staging → production）
  - deprecate_model() — 標記為 deprecated
  - get_latest()      — 取得某 stage 的最新版本
- 原始檔案寫入 API（write_run_bundle 等）維持不變
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_REGISTRY_INDEX_FILE = "registry.json"


def default_artifacts_root() -> str:
    return os.environ.get("PREDICT_AI_ARTIFACTS_ROOT", "artifacts")


def default_registry_root() -> str:
    return os.environ.get("PREDICT_AI_REGISTRY_ROOT", "model_registry")


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


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    輕量模型 Registry（不依賴 MLflow）。

    目錄結構：
        {registry_root}/
            registry.json          <- 所有版本索引
            {model_name}/
                v1/  model.pkl  meta.json
                v2/  model.pkl  meta.json
                ...

    Stages：
        "development" → "staging" → "production" → "deprecated"
    """

    STAGES = ("development", "staging", "production", "deprecated")

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.abspath(root or default_registry_root())
        self._index_path = os.path.join(self.root, _REGISTRY_INDEX_FILE)
        os.makedirs(self.root, exist_ok=True)
        self._index: Dict[str, List[Dict[str, Any]]] = self._load_index()

    # ------------------------------------------------------------------
    # Index 管理
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, List[Dict[str, Any]]]:
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        write_json(self._index_path, self._index)

    # ------------------------------------------------------------------
    # 登記
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        model: Any,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: str = "development",
        description: str = "",
        run_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        登記模型新版本。

        Parameters
        ----------
        name         : 模型名稱（如 "stock_predictor"）
        model        : 可序列化的模型物件（pkl）
        metrics      : 訓練指標 dict
        stage        : 初始 stage（development / staging / production）
        source_path  : 若指定 .pkl 路徑，直接 copy；否則由 ``model`` 序列化

        Returns
        -------
        version_meta dict
        """
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]
        entries = self._index.setdefault(safe_name, [])
        version = len(entries) + 1
        ver_dir = os.path.join(self.root, safe_name, f"v{version}")
        os.makedirs(ver_dir, exist_ok=True)

        # 儲存模型
        model_path = os.path.join(ver_dir, "model.pkl")
        if source_path and os.path.isfile(source_path):
            shutil.copy2(source_path, model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 儲存 meta
        meta: Dict[str, Any] = {
            "name": safe_name,
            "version": version,
            "stage": stage if stage in self.STAGES else "development",
            "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": metrics or {},
            "params": params or {},
            "feature_names": list(feature_names or []),
            "tags": tags or {},
            "description": description,
            "run_id": run_id,
            "model_path": model_path,
        }
        meta_path = os.path.join(ver_dir, "meta.json")
        write_json(meta_path, meta)

        entries.append(meta)
        self._save_index()
        return meta

    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------

    def list_models(
        self,
        name: Optional[str] = None,
        *,
        stage: Optional[str] = None,
        tag_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        列出已登記模型版本（可按名稱 / stage / tag 過濾）。

        Returns list of version_meta dicts，按登記時間升序。
        """
        result: List[Dict[str, Any]] = []
        names = [name] if name else list(self._index.keys())
        for n in names:
            for entry in self._index.get(n, []):
                if stage and entry.get("stage") != stage:
                    continue
                if tag_filter:
                    t = entry.get("tags") or {}
                    if not all(t.get(k) == v for k, v in tag_filter.items()):
                        continue
                result.append(entry)
        return result

    def get_version(self, name: str, version: int) -> Optional[Dict[str, Any]]:
        """取得特定版本的 meta。"""
        for entry in self._index.get(name, []):
            if entry.get("version") == version:
                return entry
        return None

    def get_latest(
        self, name: str, stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """取得最新版本（可限定 stage）。"""
        entries = [
            e for e in self._index.get(name, [])
            if stage is None or e.get("stage") == stage
        ]
        return entries[-1] if entries else None

    # ------------------------------------------------------------------
    # 載入
    # ------------------------------------------------------------------

    def load_registered_model(
        self, name: str, version: Optional[int] = None, *, stage: Optional[str] = None
    ) -> Any:
        """
        載入已登記的模型（pickle）。

        若 version=None 且 stage=None → 取最新版本。
        若 version=None 且 stage 指定 → 取該 stage 最新版本。
        """
        if version is not None:
            meta = self.get_version(name, version)
        else:
            meta = self.get_latest(name, stage)

        if meta is None:
            raise FileNotFoundError(
                f"找不到模型 {name!r}（version={version}, stage={stage}）"
            )
        model_path = meta.get("model_path", "")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型檔案不存在：{model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    # Stage 管理
    # ------------------------------------------------------------------

    def promote_model(
        self,
        name: str,
        version: int,
        *,
        to_stage: str,
    ) -> Dict[str, Any]:
        """升階模型至指定 stage（同名的舊 production 會自動降為 staging）。"""
        if to_stage not in self.STAGES:
            raise ValueError(f"stage 必須是：{self.STAGES}")

        # 若升至 production，舊的 production 降級
        if to_stage == "production":
            for entry in self._index.get(name, []):
                if entry.get("stage") == "production" and entry.get("version") != version:
                    entry["stage"] = "staging"
                    # 同步更新 meta.json
                    _meta_path = os.path.join(
                        self.root, name, f"v{entry['version']}", "meta.json"
                    )
                    if os.path.isfile(_meta_path):
                        write_json(_meta_path, entry)

        meta = self.get_version(name, version)
        if meta is None:
            raise FileNotFoundError(f"找不到模型 {name!r} v{version}")
        meta["stage"] = to_stage
        meta["promoted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # 更新 meta.json
        _meta_path = os.path.join(self.root, name, f"v{version}", "meta.json")
        if os.path.isfile(_meta_path):
            write_json(_meta_path, meta)
        self._save_index()
        return meta

    def deprecate_model(self, name: str, version: int) -> Dict[str, Any]:
        """標記版本為 deprecated（不可再 load 但保留檔案）。"""
        return self.promote_model(name, version, to_stage="deprecated")

    # ------------------------------------------------------------------
    # 刪除
    # ------------------------------------------------------------------

    def delete_model_version(self, name: str, version: int, *, dry_run: bool = False) -> bool:
        """
        刪除指定版本（檔案 + index）。

        dry_run=True 時只回傳是否可刪除，不實際刪除。
        """
        entries = self._index.get(name, [])
        target = None
        for entry in entries:
            if entry.get("version") == version:
                target = entry
                break
        if target is None:
            return False
        if target.get("stage") == "production":
            raise ValueError("不可直接刪除 production 版本，請先降階再刪除。")
        if dry_run:
            return True
        ver_dir = os.path.join(self.root, name, f"v{version}")
        if os.path.isdir(ver_dir):
            shutil.rmtree(ver_dir, ignore_errors=True)
        self._index[name] = [e for e in entries if e.get("version") != version]
        self._save_index()
        return True

    # ------------------------------------------------------------------
    # 比較版本
    # ------------------------------------------------------------------

    def compare_versions(
        self,
        name: str,
        version_a: int,
        version_b: int,
        *,
        metrics_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """比較兩個版本的指標與設定。"""
        ma = self.get_version(name, version_a)
        mb = self.get_version(name, version_b)
        if ma is None or mb is None:
            raise FileNotFoundError("指定版本不存在。")
        keys = metrics_keys or list(set(list(ma.get("metrics", {}).keys()) + list(mb.get("metrics", {}).keys())))
        diff: Dict[str, Any] = {}
        for k in keys:
            va = ma.get("metrics", {}).get(k)
            vb = mb.get("metrics", {}).get(k)
            diff[k] = {"v_a": va, "v_b": vb, "delta": (vb - va) if (va is not None and vb is not None) else None}
        return {
            "name": name,
            "version_a": version_a,
            "version_b": version_b,
            "metrics_diff": diff,
            "stage_a": ma.get("stage"),
            "stage_b": mb.get("stage"),
        }
