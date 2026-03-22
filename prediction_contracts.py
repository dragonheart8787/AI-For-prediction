#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預測管線契約：設定與統一回傳結構（與 UnifiedPredictor 搭配，避免「萬能神檔」擴散）。
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """訓練／推論相關設定（單一物件傳遞，避免散亂 kwargs 地獄）。"""

    model_type: str = "linear"
    normalize: bool = False
    seed: int = 42
    use_onnx: bool = False
    auto_onnx: bool = False
    cache_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 60
    rate_limit_refill_per_sec: float = 1.0
    rate_limit_mode: str = "raise"
    tasks_path: str = "config/tasks.yaml"


@dataclass
class InferenceConfig:
    """推論預設（HTTP／批次可覆寫）。"""

    domain: str = "financial"
    batch_size_cap: int = 8192


@dataclass
class RuntimeCapabilities:
    """環境能力探測（lazy：不在 import 時載入 torch）。"""

    has_xgboost: bool = False
    has_lightgbm: bool = False
    has_onnxruntime: bool = False
    has_skl2onnx: bool = False
    has_torch: bool = False
    has_optuna: bool = False


def detect_runtime_capabilities() -> RuntimeCapabilities:
    def _has(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    return RuntimeCapabilities(
        has_xgboost=_has("xgboost"),
        has_lightgbm=_has("lightgbm"),
        has_onnxruntime=_has("onnxruntime"),
        has_skl2onnx=_has("skl2onnx"),
        has_torch=_has("torch"),
        has_optuna=_has("optuna"),
    )


@dataclass
class PredictionArtifacts:
    """可序列化、可寫入實驗紀錄的附帶產物摘要。"""

    confidence: Optional[float] = None
    horizons: Optional[List[int]] = None
    n_samples: Optional[int] = None
    domain: Optional[str] = None
    automl_meta: Optional[Dict[str, Any]] = None
    torch_feature_attr: Optional[List[float]] = None


@dataclass
class FitSummary:
    """訓練回合摘要（可掛 RunMetadata）。"""

    model_type: str = ""
    samples: int = 0
    features: int = 0
    train_rmse: float = 0.0
    train_mae: float = 0.0
    train_r2: float = 0.0


def artifacts_to_dict(a: PredictionArtifacts) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if a.confidence is not None:
        d["confidence"] = a.confidence
    if a.horizons is not None:
        d["horizons"] = a.horizons
    if a.n_samples is not None:
        d["n_samples"] = a.n_samples
    if a.domain is not None:
        d["domain"] = a.domain
    if a.automl_meta is not None:
        d["automl_meta"] = a.automl_meta
    if a.torch_feature_attr is not None:
        d["torch_feature_attr"] = a.torch_feature_attr
    return d


def build_fit_result(
    *,
    predictions: Any,
    model_type: str,
    feature_names: Optional[List[str]],
    metrics: Dict[str, Any],
    artifacts: Dict[str, Any],
    legacy_flat_metrics: bool = True,
) -> Dict[str, Any]:
    """
    訓練完成統一回傳。

    - ``metrics``：含 train_rmse, train_mae, train_r2, samples, features, model 等
    - ``legacy_flat_metrics``：為 True 時將 train_* 同步鋪平在頂層（過渡期相容）
    """
    out: Dict[str, Any] = {
        "predictions": predictions,
        "model_type": model_type,
        "feature_names": list(feature_names or []),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts),
    }
    if legacy_flat_metrics:
        for k in ("train_rmse", "train_mae", "train_r2", "samples", "features", "model"):
            if k in metrics:
                out[k] = metrics[k]
    return out


def build_predict_result(
    *,
    predictions: Any,
    model_type: str,
    feature_names: Optional[List[str]],
    metrics: Optional[Dict[str, Any]],
    artifacts: Dict[str, Any],
    legacy: bool = True,
) -> Dict[str, Any]:
    """推論統一回傳；legacy=True 時保留 ``prediction`` / ``model`` 鍵。"""
    out: Dict[str, Any] = {
        "predictions": predictions,
        "model_type": model_type,
        "feature_names": list(feature_names or []),
        "metrics": metrics,
        "artifacts": dict(artifacts),
    }
    if legacy:
        out["prediction"] = predictions
        out["model"] = model_type
        # 過渡期：維持舊版頂層鍵，新程式請讀 artifacts
        if artifacts:
            if "confidence" in artifacts:
                out["confidence"] = artifacts["confidence"]
            if "domain" in artifacts:
                out["domain"] = artifacts["domain"]
            if "horizons" in artifacts:
                out["horizons"] = artifacts["horizons"]
            if "n_samples" in artifacts:
                out["n_samples"] = artifacts["n_samples"]
    return out
