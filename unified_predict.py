from __future__ import annotations

import json
import os
import pickle
import time
import threading
import hashlib
import numbers
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from prediction_contracts import ModelConfig, build_fit_result, build_predict_result

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except Exception:  # pragma: no cover
    LinearRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    mean_squared_error = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    r2_score = None  # type: ignore

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore

try:
    from skl2onnx import convert_sklearn  # type: ignore
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore
except Exception:
    convert_sklearn = None  # type: ignore
    FloatTensorType = None  # type: ignore

logger = logging.getLogger(__name__)


def _sklearn_compat_predict(model: Any, X_arr: np.ndarray) -> np.ndarray:
    """當 estimator 帶有 ``feature_names_in_`` 時，以同名欄位 DataFrame 預測，避免 sklearn 警告。"""
    fn_in = getattr(model, "feature_names_in_", None)
    if pd is None or fn_in is None:
        return np.asarray(model.predict(X_arr), dtype=float)
    try:
        cols = [str(x) for x in np.asarray(fn_in).ravel()]
    except Exception:
        return np.asarray(model.predict(X_arr), dtype=float)
    if len(cols) != X_arr.shape[1]:
        return np.asarray(model.predict(X_arr), dtype=float)
    X_df = pd.DataFrame(X_arr, columns=cols)
    return np.asarray(model.predict(X_df), dtype=float)


def _time_series_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    min_val_samples: int = 4,
    min_train_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """依時間順序：前段訓練、後段驗證（不切亂序）。"""
    n = X.shape[0]
    if n < min_train_samples + min_val_samples:
        return X, y, None, None
    n_val = max(min_val_samples, int(round(n * val_ratio)))
    n_val = min(n_val, max(min_val_samples, n // 3))
    n_train = n - n_val
    if n_train < min_train_samples:
        return X, y, None, None
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class _TokenBucket:
    """Token Bucket 限流（供測試用，UnifiedPredictor 使用內部 _acquire_token）。"""
    def __init__(self, capacity: int = 50, refill_per_sec: float = 25.0) -> None:
        self.capacity = capacity
        self._tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False


class _LRUCache:
    def __init__(self, max_items: int = 64) -> None:
        self.max_items = max_items
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_items:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


class ONNXRunner:
    def __init__(self, session: "ort.InferenceSession") -> None:  # type: ignore[name-defined]
        self.session = session
        outs = [o.name for o in session.get_outputs()]
        self.output_name = outs[0] if outs else None
        ins = [i.name for i in session.get_inputs()]
        self.input_name = ins[0] if ins else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        feed = {self.input_name: X.astype(np.float32)}
        out = self.session.run([self.output_name], feed)[0]
        return np.asarray(out)


class UnifiedPredictor:
    """CPU 友善的統一預測介面，支援多種迴歸模型、ONNX、批次推論、多地平線輸出。

    Features:
        - 模型：linear, xgboost, lightgbm, ensemble, **automl**（Optuna）,
          **mlp_torch / lstm / transformer**（PyTorch，見 ``nn_models.torch_regressors``）
        - ONNX：線性／樹模型可轉；Torch 預設不轉，可用 ``export_torch_model_onnx()``
        - 自動資料正規化（可選）、模型持久化、訓練／驗證指標（RMSE, MAE, R²）
        - Torch 訓練後可儲存梯度近似特徵重要性向量 ``_torch_feature_attr``
        - LRU 快取與 Token Bucket 限流；支援 numpy / list / dict / DataFrame 輸入
    """

    def __init__(
        self,
        tasks_path: str = "config/tasks.yaml",
        auto_onnx: bool = True,
        normalize: bool = False,
        rate_limit_enabled: bool = True,
        rate_limit_capacity: int = 60,
        rate_limit_refill_per_sec: float = 1.0,
        rate_limit_mode: Literal["raise", "sleep"] = "raise",
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        cfg = model_config or ModelConfig(
            normalize=bool(normalize),
            auto_onnx=bool(auto_onnx),
            rate_limit_enabled=bool(rate_limit_enabled),
            rate_limit_capacity=max(1, int(rate_limit_capacity)),
            rate_limit_refill_per_sec=max(1e-9, float(rate_limit_refill_per_sec)),
            rate_limit_mode=str(rate_limit_mode),
            tasks_path=tasks_path,
        )
        self._model_config: ModelConfig = cfg
        self.tasks: Dict[str, Any] = {}
        self.model_name: str = "linear"
        self.model: Any = None
        self.onnx_runner: Optional[ONNXRunner] = None
        self._cache = _LRUCache(max_items=64)
        self.rate_limit_enabled: bool = bool(cfg.rate_limit_enabled)
        self.rate_limit_capacity: int = max(1, int(cfg.rate_limit_capacity))
        self.rate_limit_refill_per_sec: float = max(1e-9, float(cfg.rate_limit_refill_per_sec))
        self.rate_limit_mode: Literal["raise", "sleep"] = (
            cfg.rate_limit_mode if cfg.rate_limit_mode in ("raise", "sleep") else "raise"
        )
        self._bucket_tokens: float = float(self.rate_limit_capacity)
        self._bucket_last_ts: float = time.monotonic()
        self._bucket_lock: threading.Lock = threading.Lock()
        self.auto_onnx: bool = bool(cfg.auto_onnx)
        self.normalize: bool = bool(cfg.normalize)
        self._scaler: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None
        self._input_dim: Optional[int] = None
        self._fit_metrics: Optional[Dict[str, float]] = None
        self._automl_meta: Optional[Dict[str, Any]] = None
        self._torch_feature_attr: Optional[np.ndarray] = None
        self._load_tasks(cfg.tasks_path)

    def _acquire_token(self, cost: float = 1.0) -> None:
        """取得限流 token，cost 不足時依 mode 決定 raise 或 sleep 等待。"""
        if not self.rate_limit_enabled:
            return
        cost = max(0.001, float(cost))
        with self._bucket_lock:
            now = time.monotonic()
            elapsed = now - self._bucket_last_ts
            self._bucket_tokens = min(
                float(self.rate_limit_capacity),
                self._bucket_tokens + elapsed * self.rate_limit_refill_per_sec,
            )
            self._bucket_last_ts = now

            if self._bucket_tokens >= cost:
                self._bucket_tokens -= cost
                return

            if self.rate_limit_mode == "sleep":
                need = cost - self._bucket_tokens
                wait_s = need / max(self.rate_limit_refill_per_sec, 1e-9)
                wait_s = min(wait_s, 60.0)
        if self.rate_limit_mode == "sleep":
            time.sleep(wait_s)
            self._acquire_token(cost)
            return
        raise RuntimeError("Rate limited. 請稍後再試或調整限流參數。")

    def _load_tasks(self, tasks_path: str) -> None:
        if yaml is None or not os.path.exists(tasks_path):
            self.tasks = {
                "version": 1,
                "domains": [
                    {"id": "financial", "horizons": [1, 5, 10, 20]},
                    {"id": "weather", "horizons": [1, 5, 10]},
                    {"id": "medical", "horizons": [1, 5, 10]},
                    {"id": "energy", "horizons": [1, 5, 10]},
                ],
                "routing": {"default_model": "linear"},
            }
            return
        with open(tasks_path, "r", encoding="utf-8") as f:
            self.tasks = yaml.safe_load(f) or {}

    def _to_array(self, X: Union[np.ndarray, "pd.DataFrame", List[List[float]], List[Dict[str, Any]]]) -> np.ndarray:
        if pd is not None and isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_arr = X[numeric_cols].values.astype(float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            return X_arr
        if isinstance(X, np.ndarray):
            X_arr = X.astype(float, copy=False)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            return X_arr
        if isinstance(X, list) and X and isinstance(X[0], dict):
            return self._adapt_features_from_dicts(X)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr

    def _validate_input(self, X_arr: np.ndarray, context: str = "input") -> np.ndarray:
        """檢查並清理輸入資料中的 NaN/Inf 值"""
        if X_arr.size == 0:
            raise ValueError(f"{context} 不可為空陣列")
        nan_count = int(np.isnan(X_arr).sum())
        inf_count = int(np.isinf(X_arr).sum())
        if nan_count > 0 or inf_count > 0:
            logger.warning(
                "%s 包含 %d 個 NaN 和 %d 個 Inf 值，已替換為 0",
                context, nan_count, inf_count,
            )
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr

    def _adapt_features_from_dicts(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        keys: List[str] = sorted({k for r in rows for k in r.keys()})

        def encode_value(v: Any) -> float:
            if isinstance(v, numbers.Number):
                try:
                    return float(v)
                except Exception:
                    return 0.0
            if v is None:
                return 0.0
            s = str(v)
            h = int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)
            return (h % 1000) / 1000.0

        data: List[List[float]] = []
        for r in rows:
            data.append([encode_value(r.get(k, 0.0)) for k in keys])
        return np.asarray(data, dtype=float)

    def _hash_batch(self, X: np.ndarray, max_bytes: int = 1_000_000) -> str:
        X_c = np.ascontiguousarray(X)
        view = X_c.view(np.uint8)
        if view.nbytes > max_bytes:
            step = max(1, int(np.ceil(view.size / max_bytes)))
            view = view[::step]
        return hashlib.sha1(view.tobytes()).hexdigest()

    def export_onnx(self, path: str, input_dim: int) -> str:
        if convert_sklearn is None or FloatTensorType is None:
            raise RuntimeError("缺少 skl2onnx，請先安裝 pip install skl2onnx")
        if not hasattr(self.model, "predict"):
            raise RuntimeError("目前僅支援可由 skl2onnx 轉換的 Sklearn 類模型")
        initial_type = [("input", FloatTensorType([None, int(input_dim)]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return path

    def load_onnx(self, path: str, intra_threads: int = 0, inter_threads: int = 0) -> None:
        if ort is None:
            raise RuntimeError("缺少 onnxruntime，請先安裝 pip install onnxruntime")
        so = ort.SessionOptions()
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            so.inter_op_num_threads = inter_threads
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.onnx_runner = ONNXRunner(sess)
        self.model_name = "onnx"

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame", List[List[float]]],
        y: Union[np.ndarray, List[float], List[List[float]]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """訓練模型。

        Returns:
            包含訓練指標的字典 (train_rmse, train_mae, train_r2, samples, features)
        """
        X_arr = self._to_array(X)
        X_arr = self._validate_input(X_arr, context="訓練特徵 X")
        y_arr = np.asarray(y, dtype=float)
        y_arr = self._validate_input(y_arr, context="目標值 y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X 與 y 樣本數不一致：X={X_arr.shape[0]}, y={y_arr.shape[0]}"
            )

        self._input_dim = X_arr.shape[1]
        self._automl_meta = None
        self._torch_feature_attr = None

        if self.normalize and StandardScaler is not None:
            self._scaler = StandardScaler()
            X_arr = self._scaler.fit_transform(X_arr)

        y_arr_fit = y_arr

        self.model_name = (
            model or self.tasks.get("routing", {}).get("default_model", "linear")
        ).lower()

        strong = bool(kwargs.get("strong", False))
        use_es = bool(kwargs.get("early_stopping", True))
        es_rounds = int(kwargs.get("early_stopping_rounds", 60))
        val_ratio = float(kwargs.get("val_ratio", 0.15))

        if self.model_name == "automl":
            from automl.optuna_runner import run_optuna_automl

            self.model, self.model_name, meta = run_optuna_automl(
                X_arr,
                y_arr_fit,
                n_trials=int(kwargs.get("automl_trials", 30)),
                timeout=kwargs.get("automl_timeout"),
                include_deep=bool(kwargs.get("automl_include_deep", False)),
                strong=strong,
                normalize=False,
            )
            self._automl_meta = meta

        elif self.model_name == "xgboost" and xgb is not None:
            params = {
                "n_estimators": kwargs.get("n_estimators", 200),
                "max_depth": kwargs.get("max_depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.1),
                "subsample": kwargs.get("subsample", 0.9),
                "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
                "tree_method": kwargs.get("tree_method", "hist"),
                "n_jobs": kwargs.get("n_jobs", os.cpu_count() or 1),
                "random_state": kwargs.get("random_state", 42),
            }
            if strong:
                params["n_estimators"] = max(int(params["n_estimators"]), 800)
                params["max_depth"] = max(int(params["max_depth"]), 8)
                params["learning_rate"] = min(float(params["learning_rate"]), 0.05)
            model_obj = xgb.XGBRegressor(**params)
            X_tr, y_tr, X_va, y_va = _time_series_train_val_split(
                X_arr, y_arr_fit, val_ratio=val_ratio
            )
            if (
                use_es
                and X_va is not None
                and X_tr.shape[0] >= 12
                and X_va.shape[0] >= 3
            ):
                try:
                    model_obj.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        early_stopping_rounds=es_rounds,
                        verbose=False,
                    )
                except TypeError:
                    model_obj.fit(X_arr, y_arr_fit)
            else:
                model_obj.fit(X_arr, y_arr_fit)
            self.model = model_obj

        elif self.model_name == "lightgbm" and lgb is not None:
            params = {
                "n_estimators": kwargs.get("n_estimators", 400),
                "num_leaves": kwargs.get("num_leaves", 63),
                "learning_rate": kwargs.get("learning_rate", 0.05),
                "subsample": kwargs.get("subsample", 0.9),
                "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
                "n_jobs": kwargs.get("n_jobs", os.cpu_count() or 1),
                "random_state": kwargs.get("random_state", 42),
                "verbose": kwargs.get("verbose", -1),
            }
            if strong:
                params["n_estimators"] = max(int(params["n_estimators"]), 900)
                params["num_leaves"] = max(int(params["num_leaves"]), 64)
                params["learning_rate"] = min(float(params["learning_rate"]), 0.04)
            model_obj = lgb.LGBMRegressor(**params)
            X_tr, y_tr, X_va, y_va = _time_series_train_val_split(
                X_arr, y_arr_fit, val_ratio=val_ratio
            )
            if (
                use_es
                and X_va is not None
                and X_tr.shape[0] >= 12
                and X_va.shape[0] >= 3
            ):
                try:
                    callbacks = [
                        lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]
                    model_obj.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=callbacks,
                    )
                except Exception:
                    model_obj.fit(X_arr, y_arr_fit)
            else:
                model_obj.fit(X_arr, y_arr_fit)
            self.model = model_obj

        elif (
            self.model_name == "ensemble"
            and xgb is not None
            and LinearRegression is not None
        ):
            lr = LinearRegression(n_jobs=None)
            lr.fit(X_arr, y_arr_fit)
            params = {
                "n_estimators": kwargs.get("n_estimators", 500 if strong else 300),
                "max_depth": kwargs.get("max_depth", 8 if strong else 6),
                "learning_rate": kwargs.get(
                    "learning_rate", 0.05 if strong else 0.08
                ),
                "subsample": kwargs.get("subsample", 0.9),
                "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
                "tree_method": kwargs.get("tree_method", "hist"),
                "n_jobs": kwargs.get("n_jobs", os.cpu_count() or 1),
                "random_state": kwargs.get("random_state", 42),
            }
            xgb_m = xgb.XGBRegressor(**params)
            X_tr, y_tr, X_va, y_va = _time_series_train_val_split(
                X_arr, y_arr_fit, val_ratio=val_ratio
            )
            if (
                use_es
                and X_va is not None
                and X_tr.shape[0] >= 12
                and X_va.shape[0] >= 3
            ):
                try:
                    xgb_m.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        early_stopping_rounds=es_rounds,
                        verbose=False,
                    )
                except TypeError:
                    xgb_m.fit(X_arr, y_arr_fit)
            else:
                xgb_m.fit(X_arr, y_arr_fit)
            w_lin = float(kwargs.get("ensemble_linear_weight", 0.25))
            w_xgb = float(kwargs.get("ensemble_xgb_weight", 0.75))
            s = w_lin + w_xgb
            if s > 0:
                w_lin, w_xgb = w_lin / s, w_xgb / s
            self.model = {
                "type": "ensemble",
                "members": [("linear", lr), ("xgboost", xgb_m)],
                "weights": [w_lin, w_xgb],
            }

        elif self.model_name == "ensemble":
            logger.warning("ensemble 需要 xgboost 與 sklearn；改用 linear")
            self.model_name = "linear"
            if LinearRegression is not None:
                lr = LinearRegression(n_jobs=None)
                lr.fit(X_arr, y_arr_fit)
                self.model = lr
            else:
                self.model = {"mean_y": float(y_arr_fit.mean()), "coef": []}

        elif self.model_name in ("mlp_torch", "lstm", "transformer"):
            try:
                from nn_models.torch_regressors import TorchRegressorWrapper
            except ImportError as e:
                raise ImportError(
                    "深度模型需要 PyTorch：pip install torch"
                ) from e
            kind = self.model_name
            seq_len = int(kwargs.get("seq_len", 14))
            if kind == "mlp_torch":
                seq_len = 1
            hidden = int(kwargs.get("torch_hidden", 96 if strong else 64))
            tw = TorchRegressorWrapper(
                kind,
                X_arr.shape[1],
                seq_len=max(2, seq_len),
                hidden=hidden,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "num_layers",
                        "nhead",
                        "transformer_layers",
                        "d_model",
                    )
                },
            )
            tw.fit(
                X_arr,
                y_arr_fit,
                epochs=int(kwargs.get("torch_epochs", 150)),
                lr=float(kwargs.get("torch_lr", 1e-3)),
                batch_size=int(kwargs.get("torch_batch_size", 32)),
            )
            self.model = tw

        elif self.model_name == "linear" and LinearRegression is not None:
            lr = LinearRegression(n_jobs=None)
            lr.fit(X_arr, y_arr_fit)
            self.model = lr

        else:
            coef = []
            try:
                coef = np.linalg.pinv(X_arr).dot(y_arr_fit).tolist()
            except Exception:
                coef = []
            mean_y: Union[float, List[float]]
            if y_arr_fit.ndim == 1:
                mean_y = float(y_arr_fit.mean()) if y_arr_fit.size > 0 else 0.0
            else:
                mean_y = np.mean(y_arr_fit, axis=0).tolist() if y_arr_fit.size > 0 else [0.0]
            self.model = {"mean_y": mean_y, "coef": coef}

        self._cache.clear()
        self.onnx_runner = None
        self._maybe_compute_torch_feature_attr(X_arr)

        metrics = self._compute_metrics(X_arr, y_arr)
        metrics["samples"] = X_arr.shape[0]
        metrics["features"] = X_arr.shape[1]
        metrics["model"] = self.model_name
        self._fit_metrics = metrics

        if self.auto_onnx:
            try:
                _tn = type(self.model).__name__
                if _tn != "TorchRegressorWrapper" and not (
                    isinstance(self.model, dict)
                    and self.model.get("type") == "ensemble"
                ):
                    self._manual_onnx_convert(X_arr.shape[1])
            except Exception:
                pass

        train_preds = self._predict_array(X_arr).tolist()
        art: Dict[str, Any] = {
            "automl_meta": self._automl_meta,
            "torch_feature_attr": self._torch_feature_attr.tolist()
            if self._torch_feature_attr is not None
            else None,
        }
        return build_fit_result(
            predictions=train_preds,
            model_type=self.model_name,
            feature_names=self._feature_names,
            metrics=metrics,
            artifacts=art,
            legacy_flat_metrics=True,
        )

    def _compute_metrics(self, X_arr: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """計算訓練集上的評估指標"""
        try:
            preds = self._predict_array(X_arr)
            y_flat = y_true.ravel() if y_true.ndim > 1 else y_true
            p_flat = preds.ravel() if preds.ndim > 1 else preds

            min_len = min(len(y_flat), len(p_flat))
            y_flat = y_flat[:min_len]
            p_flat = p_flat[:min_len]

            rmse = float(np.sqrt(np.mean((y_flat - p_flat) ** 2)))
            mae = float(np.mean(np.abs(y_flat - p_flat)))
            ss_res = np.sum((y_flat - p_flat) ** 2)
            ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            return {"train_rmse": rmse, "train_mae": mae, "train_r2": r2}
        except Exception as e:
            logger.warning("計算訓練指標失敗: %s", e)
            return {"train_rmse": -1.0, "train_mae": -1.0, "train_r2": -1.0}

    def evaluate(
        self,
        X: Union[np.ndarray, "pd.DataFrame", List[List[float]]],
        y: Union[np.ndarray, List[float]],
    ) -> Dict[str, float]:
        """在給定資料集上計算 RMSE、MAE、R²。"""
        if self.model is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
        X_arr = self._to_array(X)
        X_arr = self._validate_input(X_arr, context="評估特徵 X")
        if self.normalize and self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)

        y_true = np.asarray(y, dtype=float)
        preds = self._predict_array(X_arr)
        y_flat = y_true.ravel() if y_true.ndim > 1 else y_true
        p_flat = preds.ravel() if preds.ndim > 1 else preds
        min_len = min(len(y_flat), len(p_flat))
        y_flat, p_flat = y_flat[:min_len], p_flat[:min_len]

        rmse = float(np.sqrt(np.mean((y_flat - p_flat) ** 2)))
        mae = float(np.mean(np.abs(y_flat - p_flat)))
        ss_res = np.sum((y_flat - p_flat) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"rmse": rmse, "mae": mae, "r2": r2}

    def _maybe_compute_torch_feature_attr(self, X_arr: np.ndarray) -> None:
        """Torch 模型可解釋性（委派 ``model_explainability``）。"""
        from model_explainability import torch_gradient_feature_vector

        self._torch_feature_attr = torch_gradient_feature_vector(
            self.model, X_arr, max_samples=64
        )

    def export_torch_model_onnx(self, path: str, *, opset_version: int = 17) -> bool:
        """若目前為 ``TorchRegressorWrapper``，嘗試匯出 ONNX（失敗回傳 False）。"""
        if type(self.model).__name__ != "TorchRegressorWrapper":
            logger.warning("export_torch_model_onnx 僅適用於 PyTorch 包裝模型")
            return False
        try:
            from nn_models.torch_regressors import try_export_torch_onnx

            return bool(try_export_torch_onnx(self.model, path, opset_version=opset_version))
        except Exception as ex:
            logger.info("Torch ONNX 匯出略過: %s", ex)
            return False

    def predict_interval_naive(
        self,
        X: Union[np.ndarray, "pd.DataFrame", List[List[float]]],
        *,
        z_score: float = 1.96,
        domain: str = "financial",
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """委派 ``prediction_intervals``（與核心推論分離）。"""
        from prediction_intervals import naive_interval_from_predictor

        return naive_interval_from_predictor(
            self, X, z_score=z_score, domain=domain, horizons=horizons
        )

    def save_model(self, path: str) -> str:
        """將模型、scaler、設定序列化儲存到檔案。"""
        if self.model is None:
            raise RuntimeError("模型尚未訓練，無法儲存。")
        state = {
            "model": self.model,
            "model_name": self.model_name,
            "scaler": self._scaler,
            "normalize": self.normalize,
            "input_dim": self._input_dim,
            "feature_names": self._feature_names,
            "fit_metrics": self._fit_metrics,
            "auto_onnx": self.auto_onnx,
            "automl_meta": getattr(self, "_automl_meta", None),
            "torch_feature_attr": getattr(self, "_torch_feature_attr", None),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("模型已儲存至 %s", path)
        return path

    def load_model(self, path: str) -> None:
        """從檔案載入模型。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型檔案不存在: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        self.model = state["model"]
        self.model_name = state.get("model_name", "linear")
        self._scaler = state.get("scaler")
        self.normalize = state.get("normalize", False)
        self._input_dim = state.get("input_dim")
        self._feature_names = state.get("feature_names")
        self._fit_metrics = state.get("fit_metrics")
        self._automl_meta = state.get("automl_meta")
        self._torch_feature_attr = state.get("torch_feature_attr")
        self._cache.clear()
        self.onnx_runner = None

        if self.auto_onnx and self._input_dim is not None:
            try:
                if type(self.model).__name__ != "TorchRegressorWrapper":
                    self._manual_onnx_convert(self._input_dim)
            except Exception:
                pass
        logger.info("模型已從 %s 載入", path)

    def _manual_onnx_convert(self, input_dim: int) -> None:
        """手動 ONNX 風格快速推論包裝（不依賴 skl2onnx）"""
        try:
            if self.model_name == "linear" and hasattr(self.model, "coef_"):

                class LinearONNXRunner:
                    def __init__(self, model: Any) -> None:
                        self.coef_ = np.array(model.coef_, dtype=np.float32)
                        self.intercept_ = np.array(model.intercept_, dtype=np.float32)

                    def predict(self, X: np.ndarray) -> np.ndarray:
                        if X.ndim == 1:
                            X = X.reshape(1, -1)
                        return X @ self.coef_.T + self.intercept_

                self.onnx_runner = LinearONNXRunner(self.model)
                self.model_name = f"onnx_{self.model_name}"

            elif self.model_name == "xgboost" and hasattr(self.model, "predict"):

                class XGBoostONNXRunner:
                    def __init__(self, model: Any) -> None:
                        self.model = model

                    def predict(self, X: np.ndarray) -> np.ndarray:
                        return _sklearn_compat_predict(self.model, X)

                self.onnx_runner = XGBoostONNXRunner(self.model)
                self.model_name = f"onnx_{self.model_name}"

            elif self.model_name == "lightgbm" and hasattr(self.model, "predict"):

                class LightGBMONNXRunner:
                    def __init__(self, model: Any) -> None:
                        self.model = model

                    def predict(self, X: np.ndarray) -> np.ndarray:
                        return _sklearn_compat_predict(self.model, X)

                self.onnx_runner = LightGBMONNXRunner(self.model)
                self.model_name = f"onnx_{self.model_name}"

        except Exception:
            pass

    def _predict_array(self, X_arr: np.ndarray) -> np.ndarray:
        if self.onnx_runner is not None:
            out = self.onnx_runner.predict(X_arr)
            out = np.asarray(out, dtype=float)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            return out

        if isinstance(self.model, dict) and self.model.get("type") == "ensemble":
            members = self.model.get("members") or []
            weights = self.model.get("weights") or []
            if not members:
                return np.zeros((X_arr.shape[0], 1), dtype=float)
            if len(weights) != len(members):
                weights = [1.0 / len(members)] * len(members)
            acc: Optional[np.ndarray] = None
            for (_name, m), w in zip(members, weights):
                if not hasattr(m, "predict"):
                    continue
                p = _sklearn_compat_predict(m, X_arr).reshape(-1, 1)
                acc = p * w if acc is None else acc + p * w
            if acc is None:
                return np.zeros((X_arr.shape[0], 1), dtype=float)
            return acc

        if hasattr(self.model, "predict"):
            preds = _sklearn_compat_predict(self.model, X_arr)
            preds = np.asarray(preds, dtype=float)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            return preds

        coef = np.asarray(self.model.get("coef", []), dtype=float)
        mean_y = self.model.get("mean_y", 0.0)
        if coef.size == X_arr.shape[1] or (coef.ndim == 2 and coef.shape[0] == X_arr.shape[1]):
            preds = X_arr.dot(coef)
        else:
            if isinstance(mean_y, list):
                preds = np.tile(np.asarray(mean_y, dtype=float), (X_arr.shape[0], 1))
            else:
                preds = np.full((X_arr.shape[0], 1), float(mean_y), dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def predict(
        self,
        X: Union[np.ndarray, "pd.DataFrame", List[List[float]]],
        domain: str = "financial",
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        self._acquire_token(1.0)
        X_arr = self._to_array(X)
        X_arr = self._validate_input(X_arr, context="預測特徵 X")
        if self.model is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
        if self.normalize and self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)

        key = (self.model_name, domain, tuple(horizons or self._default_horizons(domain)), self._hash_batch(X_arr))
        cache_key = json.dumps(key)
        if self._model_config.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        preds = self._predict_array(X_arr)
        conf = self._compute_confidence(preds)
        hz = horizons or self._default_horizons(domain)
        result = build_predict_result(
            predictions=preds.tolist(),
            model_type=self.model_name,
            feature_names=self._feature_names,
            metrics=None,
            artifacts={
                "confidence": conf,
                "horizons": hz,
                "n_samples": int(X_arr.shape[0]),
                "domain": domain,
            },
            legacy=True,
        )
        if self._model_config.cache_enabled:
            self._cache.set(cache_key, result)
        return result

    def predict_many(
        self,
        X: Union[np.ndarray, "pd.DataFrame", List[List[float]]],
        domain: str = "financial",
        horizons: Optional[List[int]] = None,
        batch_size: int = 1024,
    ) -> Dict[str, Any]:
        X_arr = self._to_array(X)
        X_arr = self._validate_input(X_arr, context="批次預測特徵 X")
        n_samples = X_arr.shape[0]

        cost = max(1.0, float(n_samples) / batch_size)
        self._acquire_token(cost)
        if self.model is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
        if self.normalize and self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)

        outputs: List[np.ndarray] = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X_arr[start:end]
            pred = self._predict_array(batch)
            outputs.append(pred)

        preds = np.vstack(outputs) if outputs else np.zeros((0, 1), dtype=float)
        conf = self._compute_confidence(preds)
        hz = horizons or self._default_horizons(domain)
        return build_predict_result(
            predictions=preds.tolist(),
            model_type=self.model_name,
            feature_names=self._feature_names,
            metrics=None,
            artifacts={
                "confidence": conf,
                "horizons": hz,
                "n_samples": n_samples,
                "domain": domain,
            },
            legacy=True,
        )

    def _compute_confidence(self, preds: np.ndarray) -> float:
        """基於預測分佈計算置信度 [0, 1]。

        使用 1/(1+std) 轉換使得低變異度映射到高置信度。
        """
        if preds.size == 0:
            return 0.0
        std = float(np.mean(np.std(preds, axis=0)))
        return float(np.clip(1.0 / (1.0 + std), 0.0, 1.0))

    def _default_horizons(self, domain: str) -> List[int]:
        domains = self.tasks.get("domains", [])
        for d in domains:
            if d.get("id") == domain:
                return list(d.get("horizons", []))
        for d in domains:
            if d.get("id") == "custom":
                return list(d.get("horizons", []))
        return [1]

    def get_model_info(self) -> Dict[str, Any]:
        """回傳模型目前狀態摘要。"""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "is_fitted": self.model is not None,
            "normalize": self.normalize,
            "auto_onnx": self.auto_onnx,
            "input_dim": self._input_dim,
            "has_onnx_runner": self.onnx_runner is not None,
            "cache_size": self._cache.size,
            "cache_enabled": bool(self._model_config.cache_enabled),
            "rate_limit_enabled": bool(self._model_config.rate_limit_enabled),
        }
        if self._fit_metrics:
            info["fit_metrics"] = self._fit_metrics
        return info

    def predict_sync(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.predict(*args, **kwargs)

    def predict_many_sync(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.predict_many(*args, **kwargs)


def quick_demo() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2048, 16))
    w = rng.normal(size=(16,))
    y1 = X @ w + rng.normal(scale=0.1, size=(2048,))
    y2 = X @ (w * 0.5) + rng.normal(scale=0.15, size=(2048,))
    y = np.stack([y1, y2], axis=1)

    predictor = UnifiedPredictor(normalize=True)
    metrics = predictor.fit(X[:512], y[:512], model="linear")
    print("=== 訓練指標 ===")
    print(json.dumps(metrics.get("metrics", metrics), ensure_ascii=False, indent=2))

    eval_metrics = predictor.evaluate(X[512:640], y[512:640])
    print("\n=== 驗證指標 ===")
    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    result_small = predictor.predict(X[:2], domain="financial")
    result_batch = predictor.predict_many(X[:128], domain="financial", batch_size=64)
    print("\n=== 預測結果 ===")
    print(json.dumps(
        {
            "small_prediction_shape": f"{len(result_small['prediction'])}x{len(result_small['prediction'][0])}",
            "batch_prediction_shape": f"{len(result_batch['prediction'])}x{len(result_batch['prediction'][0])}",
            "small_confidence": result_small["confidence"],
            "batch_confidence": result_batch["confidence"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    print("\n=== 模型資訊 ===")
    print(json.dumps(predictor.get_model_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    quick_demo()
