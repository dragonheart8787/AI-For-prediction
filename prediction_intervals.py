#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
近似預測區間（與核心推論分離）：僅依賴已訓練 Predictor 的 train_rmse 等摘要。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def naive_interval_from_predictor(
    predictor: Any,
    X: Any,
    *,
    z_score: float = 1.96,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    以訓練集 **train_rmse** 當殘差標準差，產出 ± z·RMSE 對稱區間（粗估）。
    ``predictor`` 須為已 ``fit`` 的 UnifiedPredictor（或有相同介面者）。
    """
    if getattr(predictor, "model", None) is None:
        raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
    fm = getattr(predictor, "_fit_metrics", None) or {}
    if float(fm.get("train_rmse", -1.0)) < 0:
        raise RuntimeError("無有效 train_rmse，請先完成 fit() 且指標計算成功。")
    sigma = float(fm["train_rmse"])
    X_arr = predictor._to_array(X)
    X_arr = predictor._validate_input(X_arr, context="預測特徵 X")
    if getattr(predictor, "normalize", False) and getattr(predictor, "_scaler", None) is not None:
        X_arr = predictor._scaler.transform(X_arr)
    pred = predictor._predict_array(X_arr)
    lo = pred - z_score * sigma
    hi = pred + z_score * sigma
    h = horizons
    if h is None and hasattr(predictor, "_default_horizons"):
        h = predictor._default_horizons(domain)
    preds_list = pred.tolist()
    lo_l = lo.tolist()
    hi_l = hi.tolist()
    return {
        "predictions": preds_list,
        "prediction": preds_list,
        "model_type": getattr(predictor, "model_name", "unknown"),
        "feature_names": list(getattr(predictor, "_feature_names", None) or []),
        "metrics": None,
        "lower": lo_l,
        "upper": hi_l,
        "artifacts": {
            "domain": domain,
            "horizons": h or [],
            "lower": lo_l,
            "upper": hi_l,
            "interval_sigma_train_rmse": sigma,
            "z_score": float(z_score),
            "note": "naive_interval_from_train_rmse",
        },
    }
