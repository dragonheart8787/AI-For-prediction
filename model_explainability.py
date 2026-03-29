#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型可解釋性（與推論核心分離）。

支援：
1. Torch 梯度近似特徵重要性
2. 排列重要性（Permutation Importance）— 無需額外依賴
3. 樹模型特徵重要性（XGBoost / LightGBM 內建）
4. SHAP 值（選配：pip install shap）
5. 局部線性近似（LIME 精簡版，無需 lime 套件）
6. 統一 API：explain()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Torch 梯度近似（原有）
# ---------------------------------------------------------------------------

def torch_gradient_feature_vector(
    model: Any,
    X_arr: np.ndarray,
    *,
    max_samples: int = 64,
) -> Optional[np.ndarray]:
    """
    對 ``TorchRegressorWrapper`` 計算輸入梯度近似重要性向量（已正規化合 1）。
    失敗回傳 None。
    """
    if type(model).__name__ != "TorchRegressorWrapper":
        return None
    try:
        from nn_models.torch_regressors import compute_input_gradient_importance
        return compute_input_gradient_importance(model, X_arr, max_samples=max_samples)
    except Exception as ex:
        logger.debug("torch_gradient_feature_vector 失敗：%s", ex)
        return None


# ---------------------------------------------------------------------------
# 2. 排列重要性（Permutation Importance）
# ---------------------------------------------------------------------------

def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    metric: str = "mse",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    排列重要性：對每個特徵隨機打亂後計算指標下降量（↑ 下降越多越重要）。

    Parameters
    ----------
    model        : 有 predict(X) → np.ndarray 的物件
    X            : 特徵矩陣 (N, F)
    y            : 目標值 (N,)
    n_repeats    : 打亂重複次數（越多越穩定）
    metric       : "mse" | "mae" | "rmse" | "r2"
    feature_names: 可選的特徵名稱列表

    Returns
    -------
    {
        "importances_mean": [...],   # 形狀 (F,)
        "importances_std":  [...],
        "importances_raw":  [...],   # 形狀 (F, n_repeats)
        "feature_names":    [...],
        "baseline_score":   float,
    }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, f = X.shape
    rng = np.random.default_rng(seed)

    def _score(preds: np.ndarray, y_true: np.ndarray) -> float:
        p = np.asarray(preds, dtype=float).ravel()[: len(y_true)]
        yt = y_true[: len(p)]
        if metric == "mae":
            return float(np.mean(np.abs(yt - p)))
        if metric in ("rmse",):
            return float(np.sqrt(np.mean((yt - p) ** 2)))
        if metric == "r2":
            ss_res = np.sum((yt - p) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return float(np.mean((yt - p) ** 2))  # mse default

    baseline_preds = np.asarray(model.predict(X), dtype=float).ravel()
    baseline = _score(baseline_preds, y)
    higher_is_better = metric == "r2"

    importances_raw = np.zeros((f, n_repeats), dtype=float)
    for feat_idx in range(f):
        for rep in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_preds = np.asarray(model.predict(X_perm), dtype=float).ravel()
            perm_score = _score(perm_preds, y)
            if higher_is_better:
                importances_raw[feat_idx, rep] = baseline - perm_score
            else:
                importances_raw[feat_idx, rep] = perm_score - baseline

    imp_mean = importances_raw.mean(axis=1)
    imp_std = importances_raw.std(axis=1)

    # 正規化（可選）
    total = float(np.sum(np.abs(imp_mean)))
    imp_normalized = (imp_mean / total).tolist() if total > 0 else (np.ones(f) / f).tolist()

    names = list(feature_names) if feature_names else [f"f{i}" for i in range(f)]
    return {
        "importances_mean": imp_mean.tolist(),
        "importances_std": imp_std.tolist(),
        "importances_normalized": imp_normalized,
        "importances_raw": importances_raw.tolist(),
        "feature_names": names,
        "baseline_score": baseline,
        "metric": metric,
        "n_repeats": n_repeats,
    }


# ---------------------------------------------------------------------------
# 3. 樹模型特徵重要性
# ---------------------------------------------------------------------------

def tree_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    *,
    importance_type: str = "gain",
) -> Optional[Dict[str, Any]]:
    """
    從 XGBoost / LightGBM 模型提取特徵重要性（gain / split / weight）。

    Parameters
    ----------
    model          : xgb.XGBRegressor 或 lgb.LGBMRegressor 實例
    importance_type: xgb 用 "gain" / "weight" / "cover"；lgb 用 "gain" / "split"

    Returns None 若不是受支援的樹模型。
    """
    model_cls = type(model).__name__

    # XGBoost
    if model_cls == "XGBRegressor" or model_cls == "XGBClassifier":
        try:
            scores = model.get_booster().get_score(importance_type=importance_type)
            total = sum(scores.values()) or 1.0
            imp_norm = {k: v / total for k, v in scores.items()}
            names = sorted(scores, key=lambda k: -scores[k])
            return {
                "importances": {k: scores[k] for k in names},
                "importances_normalized": imp_norm,
                "feature_names": names,
                "importance_type": importance_type,
                "model_type": "xgboost",
            }
        except Exception as ex:
            logger.debug("xgboost feature importance 失敗：%s", ex)
            return None

    # LightGBM
    if model_cls == "LGBMRegressor" or model_cls == "LGBMClassifier":
        try:
            imp = model.feature_importances_
            fn = (
                list(feature_names)
                if feature_names
                else getattr(model, "feature_name_", None) or [f"f{i}" for i in range(len(imp))]
            )
            total = float(imp.sum()) or 1.0
            imp_dict = dict(zip(fn, (imp / total).tolist()))
            imp_raw = dict(zip(fn, imp.tolist()))
            names = sorted(imp_raw, key=lambda k: -imp_raw[k])
            return {
                "importances": imp_raw,
                "importances_normalized": imp_dict,
                "feature_names": names,
                "importance_type": "split",
                "model_type": "lightgbm",
            }
        except Exception as ex:
            logger.debug("lightgbm feature importance 失敗：%s", ex)
            return None

    return None


# ---------------------------------------------------------------------------
# 4. SHAP（選配）
# ---------------------------------------------------------------------------

def shap_explanation(
    model: Any,
    X: np.ndarray,
    *,
    max_samples: int = 200,
    feature_names: Optional[List[str]] = None,
    backend: str = "auto",
) -> Optional[Dict[str, Any]]:
    """
    使用 SHAP 計算特徵貢獻值（需 pip install shap）。

    backend: "auto" | "tree" | "linear" | "kernel"
    - "auto": 自動偵測模型類型
    - "tree": TreeExplainer（XGBoost/LightGBM/RandomForest 最快）
    - "linear": LinearExplainer（sklearn LinearRegression）
    - "kernel": KernelExplainer（最慢，萬用型）

    Returns
    -------
    {
        "shap_values": np.ndarray (N, F),
        "shap_mean_abs": [...],
        "feature_names": [...],
        "base_value": float,
    }
    失敗回傳 None（不拋錯）。
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning("SHAP 未安裝：pip install shap")
        return None

    X = np.asarray(X, dtype=float)
    n = min(X.shape[0], max_samples)
    Xb = X[:n]
    fn = list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]

    try:
        model_cls = type(model).__name__
        be = backend.lower()

        if be == "tree" or (be == "auto" and model_cls in ("XGBRegressor", "LGBMRegressor")):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xb)
            base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

        elif be == "linear" or (be == "auto" and hasattr(model, "coef_")):
            background = shap.maskers.Independent(Xb, max_samples=min(50, n))
            explainer = shap.LinearExplainer(model, background)
            sv = explainer.shap_values(Xb)
            base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

        else:
            background = Xb[: min(50, n)]
            explainer = shap.KernelExplainer(
                lambda x: np.asarray(model.predict(x), dtype=float).ravel(), background
            )
            sv = explainer.shap_values(Xb, nsamples=100, silent=True)
            base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        mean_abs = np.abs(sv).mean(axis=0).tolist()
        total = sum(mean_abs) or 1.0
        mean_abs_norm = [v / total for v in mean_abs]

        return {
            "shap_values": sv.tolist(),
            "shap_mean_abs": mean_abs,
            "shap_mean_abs_normalized": mean_abs_norm,
            "feature_names": fn,
            "base_value": base,
            "n_samples": n,
        }
    except Exception as ex:
        logger.warning("SHAP 計算失敗：%s", ex)
        return None


# ---------------------------------------------------------------------------
# 5. 局部線性近似（Lightweight LIME 精簡版）
# ---------------------------------------------------------------------------

def local_linear_explanation(
    model: Any,
    x: np.ndarray,
    X_background: np.ndarray,
    *,
    n_perturb: int = 500,
    kernel_width: float = 0.75,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    局部線性近似（Lightweight LIME 精簡版，無需 lime 套件）。

    以高斯核對 x 鄰域進行加權線性迴歸，輸出每個特徵的局部線性係數。
    ``model`` 須有 predict(X) 方法。
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError as e:
        raise ImportError("local_linear_explanation 需要 scikit-learn：pip install scikit-learn") from e

    x = np.asarray(x, dtype=float).ravel()
    X_bg = np.asarray(X_background, dtype=float)
    f = len(x)
    fn = list(feature_names) if feature_names else [f"f{i}" for i in range(f)]

    rng = np.random.default_rng(seed)
    stds = X_bg.std(axis=0)
    stds[stds == 0] = 1.0

    # 擾動樣本
    noise = rng.normal(0, 1, size=(n_perturb, f)) * stds
    X_perturb = x + noise

    # 核權重：基於距離（標準化）
    dists = np.sqrt(np.sum(((X_perturb - x) / stds) ** 2, axis=1))
    sigma = kernel_width * np.sqrt(f)
    weights = np.exp(-0.5 * (dists / sigma) ** 2)

    # 預測擾動樣本
    y_perturb = np.asarray(model.predict(X_perturb), dtype=float).ravel()

    # 加權線性迴歸
    clf = Ridge(alpha=1.0, fit_intercept=True)
    clf.fit(X_perturb, y_perturb, sample_weight=weights)
    coef = clf.coef_.tolist()
    intercept = float(clf.intercept_)

    total = sum(abs(c) for c in coef) or 1.0
    coef_norm = [c / total for c in coef]

    return {
        "feature_names": fn,
        "coefficients": coef,
        "coefficients_normalized": coef_norm,
        "intercept": intercept,
        "n_perturb": n_perturb,
        "kernel_width": kernel_width,
        "method": "local_linear_lime",
    }


# ---------------------------------------------------------------------------
# 6. 統一 API
# ---------------------------------------------------------------------------

def explain(
    predictor_or_model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    method: str = "auto",
    feature_names: Optional[List[str]] = None,
    X_calib: Optional[np.ndarray] = None,
    n_repeats: int = 10,
    max_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    統一可解釋性入口。

    Parameters
    ----------
    predictor_or_model : UnifiedPredictor 或直接的 model 物件
    X                  : 特徵矩陣（用於計算重要性）
    y                  : 目標值（permutation 需要）
    method             : "auto" | "gradient" | "permutation" | "tree" | "shap"
    feature_names      : 可選特徵名稱

    Returns
    -------
    dict with:
        "method": str
        "feature_names": [...]
        "importances": {...}（各方法回傳不同）
    """
    # 嘗試從 predictor 取出底層 model
    model = predictor_or_model
    fn = feature_names
    if hasattr(predictor_or_model, "model"):
        model = predictor_or_model.model
        if fn is None:
            fn = getattr(predictor_or_model, "_feature_names", None)

    model_cls = type(model).__name__
    m = method.lower()

    # Auto 選擇
    if m == "auto":
        if model_cls == "TorchRegressorWrapper":
            m = "gradient"
        elif model_cls in ("XGBRegressor", "LGBMRegressor"):
            m = "tree"
        elif y is not None:
            m = "permutation"
        else:
            m = "shap"

    results: Dict[str, Any] = {"method": m, "feature_names": list(fn or [])}

    if m == "gradient":
        imp = torch_gradient_feature_vector(model, X, max_samples=max_samples)
        results["importances"] = imp.tolist() if imp is not None else []
        results["importances_normalized"] = imp.tolist() if imp is not None else []

    elif m == "permutation":
        if y is None:
            raise ValueError("permutation 方法需要 y（目標值）。")
        # 包裝 predict：統一成 ndarray
        class _Wrap:
            def __init__(self, m: Any) -> None:
                self._m = m
            def predict(self, X: np.ndarray) -> np.ndarray:
                out = self._m.predict(X)
                return np.asarray(out, dtype=float).ravel()

        out = permutation_importance(_Wrap(model), X, y, n_repeats=n_repeats, seed=seed, feature_names=fn)
        results.update(out)

    elif m == "tree":
        out = tree_feature_importance(model, fn)
        if out:
            results.update(out)
        else:
            logger.warning("tree_feature_importance 無法處理 %s，改用 permutation", model_cls)
            if y is not None:
                out2 = permutation_importance(model, X, y, n_repeats=n_repeats, seed=seed, feature_names=fn)
                results.update(out2)

    elif m == "shap":
        out = shap_explanation(model, X, max_samples=max_samples, feature_names=fn)
        if out:
            results.update(out)
        else:
            results["error"] = "SHAP 計算失敗，請安裝：pip install shap"

    else:
        raise ValueError(f"未知 method={method!r}，請選：auto, gradient, permutation, tree, shap")

    return results
