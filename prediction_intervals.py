#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預測區間（Prediction Intervals）模組 — 與核心推論分離。

支援四種方法：
1. naive       — ± z·RMSE（訓練集殘差，粗估，無需校準集）
2. conformal   — Split Conformal Prediction（無分佈假設，保證覆蓋率）
3. bootstrap   — Residual Bootstrap（重複取樣，更寬、較保守）
4. quantile    — Quantile Regression（需 LightGBM；直接估上下分位數）

主入口：compute_prediction_intervals()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Naive（原有，擴充文字與返回鍵）
# ---------------------------------------------------------------------------

def naive_interval_from_predictor(
    predictor: Any,
    X: Any,
    *,
    z_score: float = 1.96,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    以訓練集 train_rmse 為殘差 σ，輸出 ± z·σ 對稱區間（粗估）。
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
    return _build_interval_result(pred, lo, hi, predictor, domain, h, "naive", sigma=sigma, z_score=z_score)


# ---------------------------------------------------------------------------
# 2. Split Conformal Prediction
# ---------------------------------------------------------------------------

def conformal_intervals(
    predictor: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: Any,
    *,
    alpha: float = 0.10,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Split Conformal Prediction（Vovk et al.）。

    Parameters
    ----------
    predictor   : 已 fit 的 UnifiedPredictor
    X_calib     : 校準集特徵（**未被** fit 使用過的時序後段）
    y_calib     : 校準集目標值
    X_test      : 測試集特徵
    alpha       : 錯誤率（1-alpha = 覆蓋率，如 alpha=0.1 → 90% 覆蓋）

    保證：在可交換假設下，覆蓋率 ≥ 1 - alpha。
    """
    if getattr(predictor, "model", None) is None:
        raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

    X_calib = np.asarray(X_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=float)
    if X_calib.shape[0] < 2:
        raise ValueError("conformal 需要至少 2 個校準樣本。")

    # 校準殘差
    X_ca = predictor._to_array(X_calib)
    X_ca = predictor._validate_input(X_ca, "校準集 X")
    if getattr(predictor, "normalize", False) and getattr(predictor, "_scaler", None) is not None:
        X_ca = predictor._scaler.transform(X_ca)
    cal_pred = predictor._predict_array(X_ca).ravel()
    y_cal = y_calib.ravel()[: len(cal_pred)]
    residuals = np.abs(y_cal - cal_pred)

    # 分位數（有限樣本修正）
    n = len(residuals)
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    q_hat = float(np.quantile(residuals, level))
    logger.debug("conformal: n_calib=%d  q_hat=%.4f  alpha=%.3f", n, q_hat, alpha)

    # 測試集預測
    X_te = predictor._to_array(X_test)
    X_te = predictor._validate_input(X_te, "測試集 X")
    if getattr(predictor, "normalize", False) and getattr(predictor, "_scaler", None) is not None:
        X_te = predictor._scaler.transform(X_te)
    pred = predictor._predict_array(X_te)
    lo = pred - q_hat
    hi = pred + q_hat
    h = horizons or (predictor._default_horizons(domain) if hasattr(predictor, "_default_horizons") else [1])
    return _build_interval_result(pred, lo, hi, predictor, domain, h, "conformal",
                                  q_hat=q_hat, alpha=alpha, n_calib=n)


# ---------------------------------------------------------------------------
# 3. Residual Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_intervals(
    predictor: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: Any,
    *,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 42,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Residual Bootstrap 預測區間。

    以校準集殘差為分佈，對每個測試點重複取樣 n_boot 次，
    取 alpha/2 與 1-alpha/2 百分位數為下上界。
    """
    if getattr(predictor, "model", None) is None:
        raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

    X_calib = np.asarray(X_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=float)
    if X_calib.shape[0] < 4:
        raise ValueError("bootstrap 需要至少 4 個校準樣本。")

    X_ca = predictor._to_array(X_calib)
    X_ca = predictor._validate_input(X_ca, "校準集 X")
    if getattr(predictor, "normalize", False) and getattr(predictor, "_scaler", None) is not None:
        X_ca = predictor._scaler.transform(X_ca)
    cal_pred = predictor._predict_array(X_ca).ravel()
    y_cal = y_calib.ravel()[: len(cal_pred)]
    residuals = y_cal - cal_pred  # 有號殘差

    X_te = predictor._to_array(X_test)
    X_te = predictor._validate_input(X_te, "測試集 X")
    if getattr(predictor, "normalize", False) and getattr(predictor, "_scaler", None) is not None:
        X_te = predictor._scaler.transform(X_te)
    pred = predictor._predict_array(X_te)  # (N, 1)

    rng = np.random.default_rng(seed)
    n_test = pred.shape[0]
    boot_preds = np.empty((n_boot, n_test), dtype=float)
    for b in range(n_boot):
        sampled_residuals = rng.choice(residuals, size=n_test, replace=True)
        boot_preds[b] = pred.ravel() + sampled_residuals

    lo = np.percentile(boot_preds, 100 * alpha / 2, axis=0).reshape(-1, 1)
    hi = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0).reshape(-1, 1)
    h = horizons or (predictor._default_horizons(domain) if hasattr(predictor, "_default_horizons") else [1])
    return _build_interval_result(pred, lo, hi, predictor, domain, h, "bootstrap",
                                  n_boot=n_boot, alpha=alpha, n_calib=len(residuals))


# ---------------------------------------------------------------------------
# 4. Quantile Regression（LightGBM pinball loss）
# ---------------------------------------------------------------------------

def quantile_regression_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Any,
    *,
    alpha: float = 0.10,
    n_estimators: int = 300,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    使用 LightGBM Quantile Regression 訓練上下界模型。

    不依賴 predictor，適合在校準集上從頭另訓一對分位數模型。
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("quantile_regression_intervals 需要 lightgbm：pip install lightgbm") from e

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_te = np.asarray(X_test, dtype=float)
    if X_te.ndim == 1:
        X_te = X_te.reshape(1, -1)

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    base_params: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "verbose": -1,
    }

    lo_model = lgb.LGBMRegressor(objective="quantile", alpha=lo_q, **base_params)
    hi_model = lgb.LGBMRegressor(objective="quantile", alpha=hi_q, **base_params)
    lo_model.fit(X_train, y_train)
    hi_model.fit(X_train, y_train)

    mid_model = lgb.LGBMRegressor(objective="regression", **base_params)
    mid_model.fit(X_train, y_train)

    pred_mid = mid_model.predict(X_te).reshape(-1, 1)
    pred_lo = lo_model.predict(X_te).reshape(-1, 1)
    pred_hi = hi_model.predict(X_te).reshape(-1, 1)

    h = horizons or [1]
    return {
        "predictions": pred_mid.tolist(),
        "prediction": pred_mid.tolist(),
        "model_type": "lightgbm_quantile",
        "feature_names": list(feature_names or []),
        "metrics": None,
        "lower": pred_lo.tolist(),
        "upper": pred_hi.tolist(),
        "artifacts": {
            "method": "quantile",
            "alpha": alpha,
            "lo_quantile": lo_q,
            "hi_quantile": hi_q,
            "domain": domain,
            "horizons": h,
        },
    }


# ---------------------------------------------------------------------------
# 統一入口
# ---------------------------------------------------------------------------

def compute_prediction_intervals(
    predictor: Any,
    X_test: Any,
    *,
    method: str = "conformal",
    X_calib: Optional[np.ndarray] = None,
    y_calib: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    z_score: float = 1.96,
    n_boot: int = 200,
    seed: int = 42,
    domain: str = "financial",
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    統一預測區間入口。

    Parameters
    ----------
    predictor : 已 fit 的 UnifiedPredictor
    X_test    : 測試特徵
    method    : "naive" | "conformal" | "bootstrap" | "quantile"
    X_calib, y_calib : 校準集（conformal/bootstrap 必填；quantile 為訓練集）
    alpha     : 錯誤率（1-alpha 為目標覆蓋率）
    z_score   : 僅 naive 使用
    n_boot    : 僅 bootstrap 使用
    """
    m = method.lower()
    if m == "naive":
        return naive_interval_from_predictor(
            predictor, X_test, z_score=z_score, domain=domain, horizons=horizons
        )
    if m in ("conformal", "split_conformal"):
        if X_calib is None or y_calib is None:
            raise ValueError("conformal 方法需要 X_calib 與 y_calib。")
        return conformal_intervals(
            predictor, X_calib, y_calib, X_test, alpha=alpha, domain=domain, horizons=horizons
        )
    if m == "bootstrap":
        if X_calib is None or y_calib is None:
            raise ValueError("bootstrap 方法需要 X_calib 與 y_calib。")
        return bootstrap_intervals(
            predictor, X_calib, y_calib, X_test, n_boot=n_boot, alpha=alpha, seed=seed,
            domain=domain, horizons=horizons,
        )
    if m == "quantile":
        if X_calib is None or y_calib is None:
            raise ValueError("quantile 方法需要 X_calib（訓練集）與 y_calib。")
        X_te = predictor._to_array(X_test) if hasattr(predictor, "_to_array") else np.asarray(X_test, dtype=float)
        fn = getattr(predictor, "_feature_names", None)
        return quantile_regression_intervals(
            X_calib, y_calib, X_te, alpha=alpha, domain=domain, horizons=horizons, feature_names=fn
        )
    raise ValueError(f"未知 method={method!r}，請選：naive, conformal, bootstrap, quantile")


# ---------------------------------------------------------------------------
# 內部工具
# ---------------------------------------------------------------------------

def _build_interval_result(
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    predictor: Any,
    domain: str,
    horizons: Optional[List[int]],
    method: str,
    **extra: Any,
) -> Dict[str, Any]:
    """統一回傳格式（與 prediction_contracts 相容）。"""
    return {
        "predictions": pred.tolist(),
        "prediction": pred.tolist(),
        "model_type": getattr(predictor, "model_name", "unknown"),
        "feature_names": list(getattr(predictor, "_feature_names", None) or []),
        "metrics": None,
        "lower": lo.tolist(),
        "upper": hi.tolist(),
        "artifacts": {
            "method": method,
            "domain": domain,
            "horizons": horizons or [],
            "lower": lo.tolist(),
            "upper": hi.tolist(),
            **extra,
        },
    }


def interval_coverage(
    y_true: Union[np.ndarray, Sequence[float]],
    lower: Union[np.ndarray, Sequence[float]],
    upper: Union[np.ndarray, Sequence[float]],
) -> Dict[str, float]:
    """計算區間評估指標：覆蓋率、平均寬度、PICP、PINAW。"""
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    n = min(len(y), len(lo), len(hi))
    y, lo, hi = y[:n], lo[:n], hi[:n]
    covered = (y >= lo) & (y <= hi)
    picp = float(covered.mean())
    width = hi - lo
    avg_width = float(width.mean())
    y_range = float(y.max() - y.min()) if y.max() != y.min() else 1.0
    pinaw = avg_width / y_range
    return {"picp": picp, "avg_width": avg_width, "pinaw": pinaw, "n": n}
