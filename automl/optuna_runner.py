#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 自動搜尋迴歸模型與超參數；驗證集為時間序後段切分。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def run_optuna_automl(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    val_ratio: float = 0.18,
    include_deep: bool = False,
    strong: bool = False,
    random_state: int = 42,
    normalize: bool = True,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    搜尋最佳 model 名稱與超參，最後在**全量** (X,y) 上重訓最佳組合。

    Returns:
        model: 可 assign 給 UnifiedPredictor.model
        model_name: 例如 xgboost, lstm
        meta: study 摘要、best_params
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("AutoML 需要 optuna：pip install optuna") from e

    from unified_predict import UnifiedPredictor, _time_series_train_val_split

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = min(X.shape[0], len(y))
    X, y = X[:n], y[:n]

    X_tr, y_tr, X_va, y_va = _time_series_train_val_split(
        X, y, val_ratio=val_ratio, min_val_samples=3, min_train_samples=10
    )
    if X_va is None or len(X_va) < 2:
        split = max(10, int(n * 0.75))
        X_tr, y_tr = X[:split], y[:split]
        X_va, y_va = X[split:], y[split:]

    base_models: List[str] = ["linear", "xgboost", "lightgbm"]
    if include_deep:
        base_models.extend(["mlp_torch", "lstm", "transformer"])

    def objective(trial: Any) -> float:
        name = trial.suggest_categorical("model", base_models)
        fit_kw: Dict[str, Any] = {"early_stopping": True, "val_ratio": val_ratio}
        if strong:
            fit_kw["strong"] = True

        if name == "linear":
            pass
        elif name == "xgboost":
            fit_kw["n_estimators"] = trial.suggest_int("xgb_n_est", 100, 600)
            fit_kw["max_depth"] = trial.suggest_int("xgb_depth", 3, 10)
            fit_kw["learning_rate"] = trial.suggest_float("xgb_lr", 0.02, 0.2, log=True)
        elif name == "lightgbm":
            fit_kw["n_estimators"] = trial.suggest_int("lgb_n_est", 200, 800)
            fit_kw["num_leaves"] = trial.suggest_int("lgb_leaves", 31, 127)
            fit_kw["learning_rate"] = trial.suggest_float("lgb_lr", 0.02, 0.12, log=True)
        elif name == "mlp_torch":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["torch_hidden"] = trial.suggest_int("mlp_h", 32, 128)
            fit_kw["torch_epochs"] = trial.suggest_int("mlp_ep", 40, 200)
            fit_kw["torch_lr"] = trial.suggest_float("mlp_lr", 1e-4, 3e-3, log=True)
        elif name == "lstm":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["seq_len"] = trial.suggest_int("seq_len", 5, 21)
            fit_kw["torch_hidden"] = trial.suggest_int("lstm_h", 32, 96)
            fit_kw["torch_epochs"] = trial.suggest_int("lstm_ep", 40, 180)
            fit_kw["torch_lr"] = trial.suggest_float("lstm_lr", 1e-4, 3e-3, log=True)
        elif name == "transformer":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["seq_len"] = trial.suggest_int("tr_seq_len", 6, 18)
            fit_kw["torch_hidden"] = trial.suggest_int("tr_dim_ff", 64, 128)
            fit_kw["torch_epochs"] = trial.suggest_int("tr_ep", 40, 160)
            fit_kw["torch_lr"] = trial.suggest_float("tr_lr", 1e-4, 3e-3, log=True)
            fit_kw["nhead"] = int(trial.suggest_categorical("tr_nhead", [2, 4]))
            fit_kw["transformer_layers"] = trial.suggest_int("tr_nlayers", 1, 3)
            fit_kw["d_model"] = trial.suggest_int("tr_d_model", 32, 96)

        p = UnifiedPredictor(auto_onnx=False, normalize=normalize, rate_limit_enabled=False)
        try:
            p.fit(X_tr, y_tr, model=name, **fit_kw)
            ev = p.evaluate(X_va, y_va)
            return float(ev["rmse"])
        except Exception as ex:
            logger.debug("trial failed %s %s", name, ex)
            return 1e9

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    bp = dict(study.best_params)
    best_name = str(bp.pop("model"))
    refit_kw: Dict[str, Any] = {"early_stopping": False, "strong": strong}
    if best_name == "xgboost":
        refit_kw.update(
            {
                "n_estimators": int(bp.get("xgb_n_est", 200)),
                "max_depth": int(bp.get("xgb_depth", 6)),
                "learning_rate": float(bp.get("xgb_lr", 0.1)),
            }
        )
    elif best_name == "lightgbm":
        refit_kw.update(
            {
                "n_estimators": int(bp.get("lgb_n_est", 400)),
                "num_leaves": int(bp.get("lgb_leaves", 63)),
                "learning_rate": float(bp.get("lgb_lr", 0.05)),
            }
        )
    elif best_name == "mlp_torch":
        refit_kw["torch_hidden"] = int(bp.get("mlp_h", 64))
        refit_kw["torch_epochs"] = int(bp.get("mlp_ep", 100))
        refit_kw["torch_lr"] = float(bp.get("mlp_lr", 1e-3))
    elif best_name == "lstm":
        refit_kw["seq_len"] = int(bp.get("seq_len", 10))
        refit_kw["torch_hidden"] = int(bp.get("lstm_h", 64))
        refit_kw["torch_epochs"] = int(bp.get("lstm_ep", 100))
        refit_kw["torch_lr"] = float(bp.get("lstm_lr", 1e-3))
    elif best_name == "transformer":
        refit_kw["seq_len"] = int(bp.get("tr_seq_len", 12))
        refit_kw["torch_hidden"] = int(bp.get("tr_dim_ff", 96))
        refit_kw["torch_epochs"] = int(bp.get("tr_ep", 100))
        refit_kw["torch_lr"] = float(bp.get("tr_lr", 1e-3))
        refit_kw["nhead"] = int(bp.get("tr_nhead", 4))
        refit_kw["transformer_layers"] = int(bp.get("tr_nlayers", 2))
        refit_kw["d_model"] = int(bp.get("tr_d_model", 64))

    final = UnifiedPredictor(auto_onnx=False, normalize=normalize, rate_limit_enabled=False)
    final.fit(X, y, model=best_name, **refit_kw)
    meta = {
        "best_rmse_val_approx": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    return final.model, best_name, meta
