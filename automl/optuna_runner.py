#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 自動搜尋迴歸模型與超參數；驗證集為時間序後段切分。

新功能（v0.9.5+）：
- MedianPruner：早期剔除劣質 trial，加快搜尋
- n_jobs 平行 trial（需 optuna-distributed 或多程序）
- 更大搜尋空間：regularization、subsample、dropout、weight_decay
- 支援 cross_val 模式（時序 K-fold）
- best_model_path：搜尋完成後自動儲存最佳模型 checkpoint
- 回傳 study 物件，供外部繼續分析
"""
from __future__ import annotations

import logging
import os
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
    n_jobs: int = 1,
    use_pruner: bool = True,
    cross_val_folds: int = 0,
    best_model_path: Optional[str] = None,
    study_name: Optional[str] = None,
    extra_models: Optional[List[str]] = None,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    搜尋最佳 model 名稱與超參，最後在**全量** (X,y) 上重訓最佳組合。

    Parameters
    ----------
    n_trials        : Optuna 試驗次數
    timeout         : 秒數上限（None = 無限）
    val_ratio       : 驗證集比例（cross_val_folds=0 時用）
    include_deep    : 是否加入 mlp_torch / lstm / transformer / tcn
    strong          : 使用更大模型預設值
    random_state    : 隨機種子
    n_jobs          : 平行 trial 數（1 = 串行）
    use_pruner      : 啟用 MedianPruner（加速搜尋）
    cross_val_folds : >0 時用時序 walk-forward K-fold 驗證
    best_model_path : 若指定，搜尋完成後儲存最佳模型（.pkl）
    extra_models    : 額外加入的模型名稱清單

    Returns
    -------
    (model, model_name, meta)
    meta 包含 best_rmse_val_approx、best_params、n_trials、study（optuna Study 物件）
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

    # 驗證集切分
    X_tr, y_tr, X_va, y_va = _time_series_train_val_split(
        X, y, val_ratio=val_ratio, min_val_samples=3, min_train_samples=10
    )
    if X_va is None or len(X_va) < 2:
        split = max(10, int(n * 0.75))
        X_tr, y_tr = X[:split], y[:split]
        X_va, y_va = X[split:], y[split:]

    # 可選模型集合
    base_models: List[str] = ["linear", "xgboost", "lightgbm"]
    if include_deep:
        base_models.extend(["mlp_torch", "lstm", "transformer", "tcn"])
    if extra_models:
        for m in extra_models:
            if m not in base_models:
                base_models.append(m)

    # Walk-forward splits（cross_val_folds > 0 時使用）
    wf_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if cross_val_folds > 0:
        try:
            from validation.time_series_split import walk_forward_splits
            splits_idx = walk_forward_splits(n, n_splits=cross_val_folds, min_train_size=max(10, n // 4))
            for tr_idx, va_idx in splits_idx:
                wf_splits.append((X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]))
        except Exception as ex:
            logger.warning("walk_forward_splits 失敗，退回單一驗證集：%s", ex)

    def _eval_predictor(p: UnifiedPredictor, Xv: np.ndarray, yv: np.ndarray) -> float:
        try:
            ev = p.evaluate(Xv, yv)
            return float(ev["rmse"])
        except Exception:
            return 1e9

    def objective(trial: Any) -> float:
        name = trial.suggest_categorical("model", base_models)
        fit_kw: Dict[str, Any] = {
            "early_stopping": True,
            "val_ratio": val_ratio,
        }
        if strong:
            fit_kw["strong"] = True

        if name == "linear":
            pass  # 無可調超參

        elif name == "xgboost":
            fit_kw["n_estimators"] = trial.suggest_int("xgb_n_est", 100, 800)
            fit_kw["max_depth"] = trial.suggest_int("xgb_depth", 3, 12)
            fit_kw["learning_rate"] = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
            fit_kw["subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            fit_kw["colsample_bytree"] = trial.suggest_float("xgb_col", 0.6, 1.0)
            fit_kw["min_child_weight"] = trial.suggest_int("xgb_mcw", 1, 20)
            fit_kw["reg_alpha"] = trial.suggest_float("xgb_alpha", 0.0, 5.0)
            fit_kw["reg_lambda"] = trial.suggest_float("xgb_lambda", 0.5, 5.0)

        elif name == "lightgbm":
            fit_kw["n_estimators"] = trial.suggest_int("lgb_n_est", 200, 1000)
            fit_kw["num_leaves"] = trial.suggest_int("lgb_leaves", 20, 255)
            fit_kw["learning_rate"] = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
            fit_kw["subsample"] = trial.suggest_float("lgb_subsample", 0.6, 1.0)
            fit_kw["colsample_bytree"] = trial.suggest_float("lgb_col", 0.6, 1.0)
            fit_kw["min_child_samples"] = trial.suggest_int("lgb_mcs", 5, 50)
            fit_kw["reg_alpha"] = trial.suggest_float("lgb_alpha", 0.0, 5.0)
            fit_kw["reg_lambda"] = trial.suggest_float("lgb_lambda", 0.0, 5.0)

        elif name == "mlp_torch":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["torch_hidden"] = trial.suggest_int("mlp_h", 32, 256)
            fit_kw["torch_epochs"] = trial.suggest_int("mlp_ep", 40, 200)
            fit_kw["torch_lr"] = trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)
            fit_kw["dropout"] = trial.suggest_float("mlp_dropout", 0.0, 0.4)
            fit_kw["scheduler_type"] = trial.suggest_categorical("mlp_sched", ["cosine", "plateau", "none"])

        elif name == "lstm":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["seq_len"] = trial.suggest_int("seq_len", 5, 30)
            fit_kw["torch_hidden"] = trial.suggest_int("lstm_h", 32, 128)
            fit_kw["torch_epochs"] = trial.suggest_int("lstm_ep", 40, 200)
            fit_kw["torch_lr"] = trial.suggest_float("lstm_lr", 1e-4, 5e-3, log=True)
            fit_kw["num_layers"] = trial.suggest_int("lstm_layers", 1, 3)
            fit_kw["dropout"] = trial.suggest_float("lstm_dropout", 0.0, 0.4)

        elif name == "transformer":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["seq_len"] = trial.suggest_int("tr_seq_len", 6, 24)
            fit_kw["torch_hidden"] = trial.suggest_int("tr_dim_ff", 64, 256)
            fit_kw["torch_epochs"] = trial.suggest_int("tr_ep", 40, 180)
            fit_kw["torch_lr"] = trial.suggest_float("tr_lr", 1e-4, 5e-3, log=True)
            fit_kw["nhead"] = int(trial.suggest_categorical("tr_nhead", [2, 4, 8]))
            fit_kw["transformer_layers"] = trial.suggest_int("tr_nlayers", 1, 4)
            fit_kw["d_model"] = trial.suggest_int("tr_d_model", 32, 128)
            fit_kw["dropout"] = trial.suggest_float("tr_dropout", 0.0, 0.3)

        elif name == "tcn":
            try:
                import torch  # noqa: F401
            except ImportError:
                return 1e9
            fit_kw["seq_len"] = trial.suggest_int("tcn_seq_len", 8, 32)
            fit_kw["torch_hidden"] = trial.suggest_int("tcn_h", 32, 128)
            fit_kw["torch_epochs"] = trial.suggest_int("tcn_ep", 40, 150)
            fit_kw["torch_lr"] = trial.suggest_float("tcn_lr", 1e-4, 5e-3, log=True)
            fit_kw["tcn_layers"] = trial.suggest_int("tcn_layers", 2, 6)
            fit_kw["tcn_kernel_size"] = trial.suggest_int("tcn_ks", 3, 7)
            fit_kw["dropout"] = trial.suggest_float("tcn_dropout", 0.0, 0.4)

        # Walk-forward cross-validation
        if wf_splits:
            rmse_list: List[float] = []
            for fold_i, (Xtr_f, ytr_f, Xva_f, yva_f) in enumerate(wf_splits):
                p = UnifiedPredictor(auto_onnx=False, normalize=normalize, rate_limit_enabled=False)
                try:
                    p.fit(Xtr_f, ytr_f, model=name, **fit_kw)
                    rmse_list.append(_eval_predictor(p, Xva_f, yva_f))
                except Exception as ex:
                    logger.debug("fold %d trial failed: %s", fold_i, ex)
                    rmse_list.append(1e9)
                # Pruning 支援
                if use_pruner:
                    trial.report(float(np.mean(rmse_list)), step=fold_i)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            return float(np.mean(rmse_list))

        # 單一驗證集
        p = UnifiedPredictor(auto_onnx=False, normalize=normalize, rate_limit_enabled=False)
        try:
            p.fit(X_tr, y_tr, model=name, **fit_kw)
            return _eval_predictor(p, X_va, y_va)
        except Exception as ex:
            logger.debug("trial failed %s %s", name, ex)
            return 1e9

    # 建立 Study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2) if use_pruner else None
    study = optuna.create_study(
        study_name=study_name or "predict_ai_automl",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=False,
        catch=(Exception,),
    )

    # 建構最佳超參
    bp = dict(study.best_params)
    best_name = str(bp.pop("model"))
    refit_kw = _build_refit_kwargs(best_name, bp, strong=strong)

    # 全量重訓
    final = UnifiedPredictor(auto_onnx=False, normalize=normalize, rate_limit_enabled=False)
    final.fit(X, y, model=best_name, **refit_kw)

    # 選配儲存
    if best_model_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(best_model_path)) or ".", exist_ok=True)
            final.save_model(best_model_path)
            logger.info("AutoML 最佳模型已儲存至 %s", best_model_path)
        except Exception as ex:
            logger.warning("AutoML 模型儲存失敗：%s", ex)

    meta = {
        "best_rmse_val_approx": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "best_model": best_name,
        "study": study,
    }
    return final.model, best_name, meta


def _build_refit_kwargs(best_name: str, bp: Dict[str, Any], *, strong: bool) -> Dict[str, Any]:
    """將 Optuna 最佳參數映射為 UnifiedPredictor.fit() 的 kwargs。"""
    kw: Dict[str, Any] = {"early_stopping": False, "strong": strong}
    if best_name == "xgboost":
        kw.update({
            "n_estimators": int(bp.get("xgb_n_est", 300)),
            "max_depth": int(bp.get("xgb_depth", 6)),
            "learning_rate": float(bp.get("xgb_lr", 0.1)),
            "subsample": float(bp.get("xgb_subsample", 0.9)),
            "colsample_bytree": float(bp.get("xgb_col", 0.9)),
            "min_child_weight": int(bp.get("xgb_mcw", 1)),
            "reg_alpha": float(bp.get("xgb_alpha", 0.0)),
            "reg_lambda": float(bp.get("xgb_lambda", 1.0)),
        })
    elif best_name == "lightgbm":
        kw.update({
            "n_estimators": int(bp.get("lgb_n_est", 400)),
            "num_leaves": int(bp.get("lgb_leaves", 63)),
            "learning_rate": float(bp.get("lgb_lr", 0.05)),
            "subsample": float(bp.get("lgb_subsample", 0.9)),
            "colsample_bytree": float(bp.get("lgb_col", 0.9)),
            "min_child_samples": int(bp.get("lgb_mcs", 20)),
            "reg_alpha": float(bp.get("lgb_alpha", 0.0)),
            "reg_lambda": float(bp.get("lgb_lambda", 0.0)),
        })
    elif best_name == "mlp_torch":
        kw.update({
            "torch_hidden": int(bp.get("mlp_h", 64)),
            "torch_epochs": int(bp.get("mlp_ep", 100)),
            "torch_lr": float(bp.get("mlp_lr", 1e-3)),
            "dropout": float(bp.get("mlp_dropout", 0.1)),
            "scheduler_type": str(bp.get("mlp_sched", "cosine")),
        })
    elif best_name == "lstm":
        kw.update({
            "seq_len": int(bp.get("seq_len", 10)),
            "torch_hidden": int(bp.get("lstm_h", 64)),
            "torch_epochs": int(bp.get("lstm_ep", 100)),
            "torch_lr": float(bp.get("lstm_lr", 1e-3)),
            "num_layers": int(bp.get("lstm_layers", 2)),
            "dropout": float(bp.get("lstm_dropout", 0.1)),
        })
    elif best_name == "transformer":
        kw.update({
            "seq_len": int(bp.get("tr_seq_len", 12)),
            "torch_hidden": int(bp.get("tr_dim_ff", 96)),
            "torch_epochs": int(bp.get("tr_ep", 100)),
            "torch_lr": float(bp.get("tr_lr", 1e-3)),
            "nhead": int(bp.get("tr_nhead", 4)),
            "transformer_layers": int(bp.get("tr_nlayers", 2)),
            "d_model": int(bp.get("tr_d_model", 64)),
            "dropout": float(bp.get("tr_dropout", 0.1)),
        })
    elif best_name == "tcn":
        kw.update({
            "seq_len": int(bp.get("tcn_seq_len", 16)),
            "torch_hidden": int(bp.get("tcn_h", 64)),
            "torch_epochs": int(bp.get("tcn_ep", 100)),
            "torch_lr": float(bp.get("tcn_lr", 1e-3)),
            "tcn_layers": int(bp.get("tcn_layers", 4)),
            "tcn_kernel_size": int(bp.get("tcn_ks", 3)),
            "dropout": float(bp.get("tcn_dropout", 0.1)),
        })
    return kw
