"""
Walk-forward 交叉驗證：每折僅用過去訓練、未來測試，適合時間序列。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from .time_series_split import walk_forward_splits

_DEEP_MODELS: Set[str] = {"mlp_torch", "lstm", "transformer"}


def walk_forward_model_eval(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "linear",
    n_splits: int = 3,
    test_size: Optional[int] = None,
    min_train_size: int = 15,
    purge_gap: int = 0,
    normalize: bool = True,
    auto_onnx: bool = False,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    wf_allow_deep: bool = False,
    wf_torch_epoch_cap: int = 48,
) -> Dict[str, Any]:
    """
    使用 UnifiedPredictor 做 walk-forward：每折重新 fit，在測試窗上 evaluate。

    Returns:
        folds: 每折 {rmse, mae, r2, n_train, n_test}
        mean_rmse, mean_mae, mean_r2
        n_folds
    """
    from unified_predict import UnifiedPredictor

    fk_in = dict(fit_kwargs or {})
    automl_frozen = fk_in.pop("automl_frozen", None)
    mlow = str(model).lower()
    if mlow == "automl" and automl_frozen:
        # 凍結最佳參數：每折只重訓固定 model，不重跑 Optuna
        model = str(automl_frozen.get("model", "linear")).lower()
        inner = dict(automl_frozen.get("fit_kw") or {})
        fk_in.update(inner)
        fit_kwargs = fk_in
        mlow = str(model).lower()
    elif mlow == "automl":
        return {
            "folds": [],
            "mean_rmse": None,
            "mean_mae": None,
            "mean_r2": None,
            "n_folds": 0,
            "error": "walk-forward 不支援 model=automl（每折跑完整 AutoML 過慢）；請先離線搜參後以 fit_kwargs['automl_frozen'] 傳入最佳組合，或關閉 --walk-forward。",
        }
    if mlow in _DEEP_MODELS and not wf_allow_deep:
        return {
            "folds": [],
            "mean_rmse": None,
            "mean_mae": None,
            "mean_r2": None,
            "n_folds": 0,
            "error": (
                f"walk-forward 預設跳過深度模型 model={model}（每折重訓 PyTorch 極慢）。"
                "若仍要執行請加上：--wf-allow-deep"
            ),
        }

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = min(X.shape[0], len(y))
    X, y = X[:n], y[:n]

    fk = dict(fit_kwargs or {})
    if mlow in _DEEP_MODELS and wf_allow_deep:
        te = int(fk.get("torch_epochs", 120))
        cap = max(5, int(wf_torch_epoch_cap))
        fk["torch_epochs"] = min(te, cap)
        fk.setdefault("random_state", 42)
        fit_kwargs = fk

    folds_out: List[Dict[str, Any]] = []
    for train_idx, test_idx in walk_forward_splits(
        n,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_gap=purge_gap,
    ):
        if len(test_idx) == 0:
            continue
        p = UnifiedPredictor(
            auto_onnx=auto_onnx,
            normalize=normalize,
            rate_limit_enabled=False,
        )
        p.fit(X[train_idx], y[train_idx], model=model, **(fit_kwargs or {}))
        ev = p.evaluate(X[test_idx], y[test_idx])
        folds_out.append(
            {
                "rmse": ev["rmse"],
                "mae": ev["mae"],
                "r2": ev["r2"],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    if not folds_out:
        return {
            "folds": [],
            "mean_rmse": None,
            "mean_mae": None,
            "mean_r2": None,
            "n_folds": 0,
            "error": "樣本過少，無法形成 walk-forward 折數",
        }

    mean_rmse = float(np.mean([f["rmse"] for f in folds_out]))
    mean_mae = float(np.mean([f["mae"] for f in folds_out]))
    mean_r2 = float(np.mean([f["r2"] for f in folds_out]))
    return {
        "folds": folds_out,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
        "n_folds": len(folds_out),
    }
