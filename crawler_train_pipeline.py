#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬蟲訓練管線：爬取資料 → 轉換為預測特徵 → 訓練 AI → 學習預測所需資料

功能：
1. 根據預測任務選擇要爬取的資料來源
2. 自動爬取並轉換為 X/y 訓練格式
3. 訓練 UnifiedPredictor
4. 分析特徵重要性，告訴 AI 預測需要什麼資料

時間序列驗證切分請用 validation.time_series_split（purge gap、walk-forward），
勿對時間序列做隨機 train_test_split。
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from runtime_paths import resolve_data_dir, resolve_models_dir
from schema_infer import align_sources
from feature_expansion import expand_timeseries_features, merge_expansion_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 專案根目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _json_sanitize(o: Any) -> Any:
    """實驗紀錄 JSON：numpy 標量／陣列轉原生型別。"""
    if isinstance(o, dict):
        return {str(k): _json_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_sanitize(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def check_model_dependencies(model: str, *, automl_include_deep: bool = False) -> None:
    """在訓練前檢查選用模型所需套件；缺少時印出提示並以代碼 2 結束。"""
    m = (model or "").lower()
    if m == "automl":
        if importlib.util.find_spec("optuna") is None:
            print(
                "錯誤：model=automl 需要 Optuna。\n"
                "請執行: pip install -r requirements-automl.txt\n"
                "或: pip install optuna",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if automl_include_deep and importlib.util.find_spec("torch") is None:
            print(
                "錯誤：--automl-include-deep 需要 PyTorch。\n"
                "請執行: pip install -r requirements-automl.txt\n"
                "或: pip install torch",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if m in ("mlp_torch", "lstm", "transformer") and importlib.util.find_spec("torch") is None:
        print(
            f"錯誤：model={m} 需要 PyTorch。\n"
            "請執行: pip install -r requirements-automl.txt\n"
            "或: pip install torch",
            file=sys.stderr,
        )
        raise SystemExit(2)


def build_fit_kwargs_from_args(args: Any) -> Dict[str, Any]:
    """從 argparse 組出傳給 UnifiedPredictor.fit 的 kwargs。"""
    kw: Dict[str, Any] = {}
    if getattr(args, "preset", None) == "strong":
        kw["strong"] = True
    if getattr(args, "model", "") == "automl":
        kw["automl_trials"] = int(getattr(args, "automl_trials", 30))
        if getattr(args, "automl_timeout", None) is not None:
            kw["automl_timeout"] = float(args.automl_timeout)
        kw["automl_include_deep"] = bool(getattr(args, "automl_include_deep", False))
    m = getattr(args, "model", "")
    if m in ("mlp_torch", "lstm", "transformer"):
        if getattr(args, "seq_len", None) is not None:
            kw["seq_len"] = int(args.seq_len)
        if getattr(args, "torch_hidden", None) is not None:
            kw["torch_hidden"] = int(args.torch_hidden)
        if getattr(args, "torch_epochs", None) is not None:
            kw["torch_epochs"] = int(args.torch_epochs)
        if getattr(args, "torch_lr", None) is not None:
            kw["torch_lr"] = float(args.torch_lr)
        if getattr(args, "torch_batch_size", None) is not None:
            kw["torch_batch_size"] = int(args.torch_batch_size)
        if getattr(args, "lstm_num_layers", None) is not None:
            kw["num_layers"] = int(args.lstm_num_layers)
        if getattr(args, "transformer_nhead", None) is not None:
            kw["nhead"] = int(args.transformer_nhead)
        if getattr(args, "transformer_layers", None) is not None:
            kw["transformer_layers"] = int(args.transformer_layers)
        if getattr(args, "transformer_d_model", None) is not None:
            kw["d_model"] = int(args.transformer_d_model)
    return kw


def build_train_log_payload(
    task_id: str,
    args: Any,
    predictor: Any,
    metrics: Dict[str, Any],
    n_samples: int,
    n_features: int,
    normalize: bool,
    *,
    run_id: Optional[str] = None,
    train_time_sec: Optional[float] = None,
    dependency_mode: Optional[str] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
    data_sources: Optional[List[str]] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """實驗紀錄用：含請求模型名與實際訓練後模型名、AutoML 摘要。"""
    core_metrics = metrics.get("metrics", metrics) if isinstance(metrics, dict) else metrics
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "model_requested": getattr(args, "model", None),
        "model_trained": getattr(predictor, "model_name", None)
        or getattr(args, "model", None),
        "metrics": _json_sanitize(core_metrics),
        "samples": n_samples,
        "features": n_features,
        "normalize": normalize,
        "run_id": run_id,
        "train_time_sec": train_time_sec,
        "dependency_mode": dependency_mode,
        "artifact_paths": artifact_paths or {},
        "data_sources": data_sources or [],
        "git_commit": git_commit,
        "config_hash": config_hash,
    }
    meta = getattr(predictor, "_automl_meta", None)
    if meta:
        payload["automl"] = _json_sanitize(
            {
                "best_rmse_val_approx": meta.get("best_rmse_val_approx"),
                "best_params": meta.get("best_params"),
                "n_trials": meta.get("n_trials"),
            }
        )
    return payload


def load_config(path: str = "config/prediction_schema.yaml") -> Dict[str, Any]:
    """載入預測任務 schema"""
    if yaml is None:
        raise ImportError("請安裝 pyyaml: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_connector(name: str):
    """根據 connector 名稱取得實例"""
    registry = {
        "yahoo": ("data_connectors.yahoo", "YahooFinanceConnector"),
        "eia": ("data_connectors.eia", "EIAConnector"),
        "newsapi": ("data_connectors.newsapi", "NewsAPIConnector"),
        "open_meteo": ("data_connectors.open_meteo", "OpenMeteoConnector"),
        "owid": ("data_connectors.owid", "OWIDConnector"),
    }
    if name not in registry:
        raise ValueError(f"未知 connector: {name}，支援: {list(registry.keys())}")
    mod_name, cls_name = registry[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def rows_to_matrix(
    rows: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """將 list[dict] 轉為數值矩陣 X 與欄位列表"""
    if not rows:
        return np.zeros((0, 0)), []

    all_keys = sorted({k for r in rows for k in r.keys()})
    exclude_set = set(exclude or []) | {"timestamp"}
    keys = [k for k in all_keys if k not in exclude_set]
    if feature_keys:
        keys = [k for k in feature_keys if k in keys]

    def enc(v: Any) -> float:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                return float(v)
            except Exception:
                return 0.0
        if v is None:
            return 0.0
        s = str(v)
        h = int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)
        return (h % 1000) / 1000.0

    X = np.asarray([[enc(r.get(k)) for k in keys] for r in rows], dtype=float)
    return X, keys


def build_target(
    rows: List[Dict[str, Any]],
    target_source: str,
    horizon: int = 1,
) -> np.ndarray:
    """從 rows 建立目標 y（延遲 target_source 欄位）
    y[i] = target_source 在 i+horizon 時刻的值（預測 horizon 步後）
    """
    vals = []
    for r in rows:
        v = r.get(target_source)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
        else:
            vals.append(0.0)
    vals = np.array(vals)

    if horizon <= 0:
        return vals

    # y[i] = vals[i+horizon]，有效長度為 len(vals)-horizon
    valid_len = len(vals) - horizon
    if valid_len <= 0:
        return vals  #  fallback
    y = vals[horizon : horizon + valid_len].astype(float)
    return y


def crawl_and_build_xy(
    task_id: str,
    config: Dict[str, Any],
    *,
    enable_feature_expansion: bool = False,
    **override_params: Any,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    根據任務爬取資料並建立 X, y。

    Returns:
        X, y, feature_names, task_info
    """
    tasks = config.get("prediction_tasks", {})
    if task_id not in tasks:
        raise ValueError(f"未知任務: {task_id}，可用: {list(tasks.keys())}")

    task = tasks[task_id]
    target_source = task.get("target_source", "close")
    target_horizon = task.get("target_horizon", 1)
    data_sources = task.get("data_sources", [])

    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    all_feature_keys: List[str] = []

    for ds in data_sources:
        conn_name = ds.get("connector", "")
        params = dict(ds.get("params", {}))
        features = ds.get("features")
        exclude = ds.get("exclude", [])

        # 允許覆寫參數（如 symbol, period）
        for k, v in override_params.items():
            if k in params or k in ["symbol", "period", "country_code", "latitude", "longitude"]:
                params[k] = v

        try:
            connector = get_connector(conn_name)
            rows = list(connector.fetch(**params))
            if rows:
                rows_by_source[conn_name] = rows
                if features:
                    all_feature_keys.extend(f for f in features if f not in all_feature_keys)
        except Exception as e:
            logger.warning("爬取 %s 失敗: %s，跳過", conn_name, e)

    if not rows_by_source:
        raise RuntimeError("無資料可訓練，請檢查網路或 API 設定")

    # 多資料源且任務有時間對齊設定：做時間窗對齊
    time_granularity = task.get("time_granularity")
    target_source_conn = task.get("target_source_connector") or (
        list(rows_by_source.keys())[0] if rows_by_source else None
    )
    join_strategy = task.get("join_strategy", "left")
    join_tolerance = task.get("join_tolerance", "36h")
    missing_policy = task.get("missing_policy", "ffill")

    ts_key = "timestamp"
    if time_granularity and target_source_conn and target_source_conn in rows_by_source and len(rows_by_source) > 1:
        prevent_leakage = bool(task.get("prevent_leakage", True))
        all_rows = align_sources(
            rows_by_source,
            target_source=target_source_conn,
            freq=time_granularity,
            join_strategy=join_strategy,
            join_tolerance=join_tolerance,
            missing_policy=missing_policy,
            timestamp_key=ts_key,
            prevent_leakage=prevent_leakage,
            log_summary=True,
        )
    else:
        all_rows = []
        for rows in rows_by_source.values():
            all_rows.extend(rows)
        if all_rows and ts_key in all_rows[0]:
            all_rows.sort(key=lambda r: str(r.get(ts_key, "")))

    X, keys = rows_to_matrix(
        all_rows,
        feature_keys=all_feature_keys or None,
        exclude=["timestamp"],
    )
    y = build_target(all_rows, target_source, target_horizon)

    # 確保 X, y 長度一致（y 因 horizon 已較短）
    n = min(X.shape[0], len(y))
    X = X[:n]
    y = np.asarray(y[:n], dtype=float)

    exp_cfg = merge_expansion_config(
        task.get("feature_expansion"),
        enable_feature_expansion,
    )
    if exp_cfg:
        X, keys = expand_timeseries_features(
            X,
            list(keys),
            lag_steps=exp_cfg["lag_steps"],
            rolling_windows=exp_cfg["rolling_windows"],
            expand_columns=exp_cfg["expand_columns"],
        )
        n = min(X.shape[0], len(y))
        X, y = X[:n], np.asarray(y[:n], dtype=float)

    task_info = {
        "task_id": task_id,
        "display_name": task.get("display_name", task_id),
        "domain": task.get("domain", "custom"),
        "samples": n,
        "features": len(keys),
    }
    return X, y, keys, task_info


def get_feature_importance(
    predictor: Any,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    根據訓練後的模型取得特徵重要性，告訴 AI 預測需要什麼資料。
    支援：LinearRegression（係數絕對值）、XGBoost、LightGBM、回退模型。
    PyTorch 包裝模型：優先使用 ``UnifiedPredictor._torch_feature_attr``（訓練後梯度近似），
    否則均分佔位。
    """
    importance = []

    if hasattr(predictor, "model") and predictor.model is not None:
        model = predictor.model

        if type(model).__name__ == "TorchRegressorWrapper":
            attr = getattr(predictor, "_torch_feature_attr", None)
            if attr is not None:
                imp_arr = np.asarray(attr, dtype=float).ravel()
                n = len(feature_names)
                for i, name in enumerate(feature_names):
                    v = float(imp_arr[i]) if i < len(imp_arr) else 0.0
                    importance.append({"feature": name, "importance": v})
                importance.sort(key=lambda x: x["importance"], reverse=True)
                return importance
            n = max(1, len(feature_names))
            eq = 1.0 / n
            for name in feature_names:
                importance.append({"feature": name, "importance": float(eq)})
            importance.sort(key=lambda x: x["importance"], reverse=True)
            return importance

        if isinstance(model, dict) and model.get("type") == "ensemble":
            for _nm, sub in model.get("members", []):
                if hasattr(sub, "feature_importances_"):
                    imp = sub.feature_importances_
                    for i, name in enumerate(feature_names):
                        if i < len(imp):
                            importance.append(
                                {"feature": name, "importance": float(imp[i])}
                            )
                    break
            if importance:
                importance.sort(key=lambda x: x["importance"], reverse=True)
                return importance

        # XGBoost
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(imp):
                    importance.append({"feature": name, "importance": float(imp[i])})

        # LightGBM
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(imp):
                    importance.append({"feature": name, "importance": float(imp[i])})

        # LinearRegression
        elif hasattr(model, "coef_"):
            coef = np.abs(np.asarray(model.coef_).ravel())
            for i, name in enumerate(feature_names):
                if i < len(coef):
                    importance.append({"feature": name, "importance": float(coef[i])})

        # 回退 dict 模型
        elif isinstance(model, dict) and "coef" in model:
            coef = np.abs(np.asarray(model["coef"]).ravel())
            for i, name in enumerate(feature_names):
                if i < len(coef):
                    importance.append({"feature": name, "importance": float(coef[i])})

    if not importance:
        for name in feature_names:
            importance.append({"feature": name, "importance": 0.0})

    # 依重要性排序
    importance.sort(key=lambda x: x["importance"], reverse=True)
    return importance


def run_train_all(config: Dict[str, Any], args: Any) -> None:
    """訓練所有任務並建立通用模型"""
    from unified_predict import UnifiedPredictor
    from training_data_store import TrainingDataStore
    from universal_predictor import map_to_canonical, CANONICAL_FEATURES

    tasks = config.get("prediction_tasks", {})
    canonical = config.get("canonical_features", CANONICAL_FEATURES)
    store = TrainingDataStore(path=args.memory_path)

    all_X, all_y = [], []
    success_tasks = []

    print("=== 全任務訓練 ===\n")
    for task_id in tasks:
        try:
            X, y, feature_names, task_info = crawl_and_build_xy(task_id, config)
            X_can = map_to_canonical(X, feature_names, canonical)
            all_X.append(X_can)
            all_y.append(y)
            store.add(X, y, task_id, task_info.get("domain", "custom"), feature_names)
            success_tasks.append((task_id, task_info["display_name"], X.shape[0]))
            print(f"  [OK] {task_id}: {X.shape[0]} 樣本")
        except Exception as e:
            print(f"  [SKIP] {task_id}: {e}")

    if not all_X:
        print("\n無任何任務成功，無法建立通用模型")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\n合併後: {X_all.shape[0]} 樣本, {X_all.shape[1]} 特徵 (canonical)")

    check_model_dependencies(
        str(getattr(args, "model", "linear")),
        automl_include_deep=bool(getattr(args, "automl_include_deep", False)),
    )
    predictor = UnifiedPredictor(auto_onnx=False, normalize=True)
    ta_kw = build_fit_kwargs_from_args(args)
    metrics = predictor.fit(X_all, y_all, model=args.model, **ta_kw)
    print(f"通用模型訓練 R2: {metrics.get('train_r2', 0):.4f}")

    models_root = resolve_models_dir(getattr(args, "models_dir", None))
    models_root.mkdir(parents=True, exist_ok=True)
    predictor._feature_names = list(canonical)
    univ_path = models_root / "universal_model.pkl"
    predictor.save_model(str(univ_path))
    print(f"\n通用模型已儲存至 {univ_path}")
    print("此模型可預測未訓練過的目標（透過 canonical 特徵對齊）")


def get_suggested_data_for_prediction(
    importance: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[str]:
    """根據特徵重要性，建議預測時需要哪些資料（最重要）"""
    total = sum(imp["importance"] for imp in importance)
    if total <= 0:
        return [imp["feature"] for imp in importance[:top_k]]

    threshold = 0.0
    suggested = []
    for imp in importance:
        if imp["importance"] > 0:
            suggested.append(imp["feature"])
            if len(suggested) >= top_k:
                break
    return suggested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="爬蟲訓練管線：爬取資料 → 訓練預測 AI → 學習預測所需資料",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="stock_price_next",
        help="預測任務 ID 或自然語言描述（如：我想預測股價、台北氣溫）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=[
            "linear",
            "xgboost",
            "lightgbm",
            "ensemble",
            "automl",
            "mlp_torch",
            "lstm",
            "transformer",
        ],
        help="訓練模型；automl=Optuna 搜模組與超參（需 optuna）；mlp_torch/lstm/transformer 需 torch",
    )
    parser.add_argument(
        "--automl-trials",
        type=int,
        default=30,
        help="model=automl 時 Optuna trial 數",
    )
    parser.add_argument(
        "--automl-timeout",
        type=float,
        default=None,
        help="model=automl 時總時限（秒），可與 --automl-trials 並用",
    )
    parser.add_argument(
        "--automl-include-deep",
        action="store_true",
        help="AutoML 搜尋納入 mlp_torch / lstm / transformer（需 PyTorch）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="lstm/transformer 滑動視窗長度（mlp_torch 忽略）",
    )
    parser.add_argument(
        "--torch-hidden",
        type=int,
        default=None,
        help="Torch 隱層寬度／Transformer dim_feedforward 相關寬度",
    )
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=None,
        help="Torch 訓練輪數",
    )
    parser.add_argument(
        "--torch-lr",
        type=float,
        default=None,
        help="Torch AdamW 學習率",
    )
    parser.add_argument(
        "--torch-batch-size",
        type=int,
        default=None,
        help="Torch 批次大小",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=None,
        help="LSTM 堆疊層數",
    )
    parser.add_argument(
        "--transformer-nhead",
        type=int,
        default=None,
        help="Transformer 注意力頭數",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=None,
        help="TransformerEncoder 層數",
    )
    parser.add_argument(
        "--transformer-d-model",
        type=int,
        default=None,
        help="Transformer d_model（會自動調整為 nhead 倍數）",
    )
    parser.add_argument(
        "--wf-allow-deep",
        action="store_true",
        help="walk-forward 時允許每折重訓 mlp_torch/lstm/transformer（慢，預設跳過）",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="config/prediction_schema.yaml",
        help="預測 schema 路徑",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="覆寫股票代碼（如 AAPL, ^GSPC）",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="覆寫爬取期間（如 3mo, 6mo）",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="儲存訓練好的模型路徑（預設為 --models-dir 下 task_<task_id>.pkl）",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="模型輸出目錄（預設：環境變數 PREDICT_AI_MODELS_DIR 或 ./models）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="資料根目錄；若未指定 --memory-path 且仍為預設 data/training_memory.jsonl，則改為 <data-dir>/training_memory.jsonl",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="列出所有預測任務與所需資料",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="關閉資料正規化",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="啟用經驗回放防遺忘（混合舊任務樣本訓練）",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="訓練所有任務並建立通用模型（可預測未訓練過的目標）",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default="data/training_memory.jsonl",
        help="訓練記憶儲存路徑",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="啟用 walk-forward 時間序列交叉驗證（在爬蟲原始 X,y 上，不含 replay 混合後資料）",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=None,
        help="walk-forward 折數上限（預設見 schema pipeline_defaults.walk_forward）",
    )
    parser.add_argument(
        "--wf-test-size",
        type=int,
        default=None,
        help="每折測試窗樣本數",
    )
    parser.add_argument(
        "--wf-purge",
        type=int,
        default=None,
        help="訓練與測試窗之間 purge 間隔（樣本數）",
    )
    parser.add_argument(
        "--wf-min-train",
        type=int,
        default=None,
        help="每折最少訓練樣本數",
    )
    parser.add_argument(
        "--no-experiment-log",
        action="store_true",
        help="關閉實驗紀錄（預設寫入 data/experiment_runs.jsonl）",
    )
    parser.add_argument(
        "--rich-features",
        action="store_true",
        help="啟用時序衍生特徵（滯後、滾動均值／標準差），需 pandas；顯著提升樹模型上限",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "strong"],
        help="strong：更多樹、更深、較低 learning_rate，並啟用驗證集 early stopping（樹模型／ensemble）",
    )
    args = parser.parse_args()

    if getattr(args, "data_dir", None):
        dr = resolve_data_dir(args.data_dir)
        if args.memory_path == "data/training_memory.jsonl":
            args.memory_path = str(dr / "training_memory.jsonl")

    config = load_config(args.schema)
    tasks = config.get("prediction_tasks", {})

    # 若 task 不是已知任務 ID，嘗試用預測目標顧問解析
    task_id = args.task
    if task_id not in tasks:
        try:
            from prediction_target_advisor import advise_for_prediction_target
            advice = advise_for_prediction_target(task_id)
            task_id = advice["task_id"]
            if advice.get("matched"):
                print(f"根據「{args.task}」匹配到任務: {task_id} ({advice['display_name']})\n")
            else:
                print(f"未匹配到任務，使用預設: {task_id}\n")
        except Exception as e:
            logger.warning("預測目標解析失敗: %s，使用原始輸入", e)

    if args.list_tasks:
        print("=== 可用預測任務與所需資料 ===\n")
        for tid, t in config.get("prediction_tasks", {}).items():
            print(f"[{tid}] {t.get('display_name', tid)}")
            print(f"  領域: {t.get('domain')}")
            print(f"  目標: {t.get('target_column')} (從 {t.get('target_source')} 衍生)")
            print("  資料來源:")
            for ds in t.get("data_sources", []):
                print(f"    - {ds.get('connector')}: {ds.get('features')}")
            print()
        return

    # 全任務訓練
    if args.train_all:
        run_train_all(config, args)
        return

    override = {}
    if args.symbol:
        override["symbol"] = args.symbol
    if args.period:
        override["period"] = args.period

    print(f"=== 爬蟲訓練管線：{task_id} ===\n")

    # 1. 爬取並建立 X, y
    print("1. 爬取資料...")
    X, y, feature_names, task_info = crawl_and_build_xy(
        task_id,
        config,
        enable_feature_expansion=args.rich_features,
        **override,
    )
    print(f"   取得 {task_info['samples']} 筆樣本，{task_info['features']} 個特徵")
    print(f"   特徵: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}\n")

    # walk-forward 僅使用爬蟲原始序列（replay 會破壞時間順序）
    X_wf = np.asarray(X, dtype=float).copy()
    y_wf = np.asarray(y, dtype=float).copy()

    # 2. 訓練（可選經驗回放防遺忘）
    print("2. 訓練預測模型...")
    from unified_predict import UnifiedPredictor
    from training_data_store import TrainingDataStore, merge_with_replay

    store = TrainingDataStore(path=args.memory_path)
    if args.replay:
        X, y = merge_with_replay(X, y, store, task_id)
        print(f"   已混合經驗回放樣本，總樣本數: {X.shape[0]}")

    check_model_dependencies(
        args.model,
        automl_include_deep=bool(args.automl_include_deep),
    )
    predictor = UnifiedPredictor(
        auto_onnx=False,
        normalize=not args.no_normalize,
    )
    fit_kw = build_fit_kwargs_from_args(args)
    from artifact_registry import ensure_run_dir, write_run_bundle
    from pipeline_run_context import hash_config_blob, new_run_id, try_git_commit

    run_id = new_run_id()
    t_train0 = time.perf_counter()
    metrics = predictor.fit(X, y, model=args.model, **fit_kw)
    train_time_sec = time.perf_counter() - t_train0

    # 存入訓練記憶（供後續防遺忘與通用模型）
    store.add(X, y, task_id, task_info.get("domain", "custom"), feature_names)
    print(f"   訓練 R2: {metrics.get('train_r2', 0):.4f}")
    print(f"   訓練 RMSE: {metrics.get('train_rmse', 0):.4f}\n")

    task_cfg = tasks.get(task_id, {}) or {}
    ds_names = [str(ds.get("connector", "")) for ds in task_cfg.get("data_sources", [])]
    dep_mode = "automl" if args.model == "automl" else "core"
    run_dir = ensure_run_dir(task_id, run_id)
    gcom = try_git_commit()
    chash = hash_config_blob({"task": task_id, "model": args.model, "preset": args.preset})
    core_m = metrics.get("metrics", metrics) if isinstance(metrics, dict) else metrics
    bundle_paths = write_run_bundle(
        run_dir,
        config={"task_id": task_id, "model": args.model, "preset": args.preset, "run_id": run_id},
        metrics=dict(core_m) if isinstance(core_m, dict) else {},
        summary={
            "run_id": run_id,
            "train_time_sec": train_time_sec,
            "dependency_mode": dep_mode,
            "git_commit": gcom,
            "config_hash": chash,
        },
        feature_manifest={"features": feature_names, "count": len(feature_names)},
    )

    pd_cfg = config.get("pipeline_defaults", {}) or {}
    exp_cfg = pd_cfg.get("experiment_log", {}) or {}
    log_path = str(exp_cfg.get("path", "data/experiment_runs.jsonl"))
    log_enabled = bool(exp_cfg.get("enabled", True)) and not args.no_experiment_log
    if log_enabled:
        try:
            from experiment_log import log_training_event

            pl = build_train_log_payload(
                task_id,
                args,
                predictor,
                metrics,
                int(X.shape[0]),
                int(X.shape[1]),
                not args.no_normalize,
                run_id=run_id,
                train_time_sec=train_time_sec,
                dependency_mode=dep_mode,
                artifact_paths=dict(bundle_paths),
                data_sources=ds_names,
                git_commit=gcom,
                config_hash=chash,
            )
            log_training_event(
                event_type="model_trained",
                run_id=run_id,
                task_name=task_id,
                model_requested=str(args.model),
                model_trained=str(predictor.model_name),
                row_count=int(X.shape[0]),
                feature_count=int(X.shape[1]),
                metrics=pl.get("metrics", {}),
                artifacts={"paths": pl.get("artifact_paths", {}), "normalize": pl.get("normalize")},
                status="success",
                duration_ms=int(train_time_sec * 1000),
                path=log_path,
                extra={"legacy_train_complete": {k: v for k, v in pl.items()}},
            )
        except Exception as e:
            logger.warning("實驗紀錄寫入失敗: %s", e)

    wf_summary: Optional[Dict[str, Any]] = None
    if args.walk_forward:
        from validation.walk_forward_eval import walk_forward_model_eval

        wf_def = pd_cfg.get("walk_forward", {}) or {}
        n_splits = args.wf_splits if args.wf_splits is not None else int(wf_def.get("n_splits", 3))
        test_size = args.wf_test_size if args.wf_test_size is not None else wf_def.get("test_size")
        test_size = int(test_size) if test_size is not None else None
        purge = args.wf_purge if args.wf_purge is not None else int(wf_def.get("purge_gap", 0))
        min_train = args.wf_min_train if args.wf_min_train is not None else int(wf_def.get("min_train_size", 15))

        print("2b. Walk-forward 交叉驗證（時間序列）...")
        wf_summary = walk_forward_model_eval(
            X_wf,
            y_wf,
            model=args.model,
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train,
            purge_gap=purge,
            normalize=not args.no_normalize,
            auto_onnx=False,
            fit_kwargs=fit_kw,
            wf_allow_deep=bool(args.wf_allow_deep),
        )
        print(f"   折數: {wf_summary.get('n_folds', 0)}")
        if wf_summary.get("mean_rmse") is not None:
            print(
                f"   平均 RMSE: {wf_summary['mean_rmse']:.4f}  MAE: {wf_summary['mean_mae']:.4f}  R2: {wf_summary['mean_r2']:.4f}\n"
            )
        else:
            print(f"   （跳過）{wf_summary.get('error', '樣本不足')}\n")
        if log_enabled and wf_summary:
            try:
                from experiment_log import log_experiment

                log_experiment(
                    "walk_forward",
                    _json_sanitize(
                        {
                            "task_id": task_id,
                            "model_requested": args.model,
                            "wf_allow_deep": bool(args.wf_allow_deep),
                            "summary": wf_summary,
                        }
                    ),
                    path=log_path,
                )
            except Exception as e:
                logger.warning("walk-forward 實驗紀錄失敗: %s", e)

    # 3. 取得特徵重要性（AI 學習預測需要什麼資料）
    print("3. 分析預測所需資料重要性...")
    importance = get_feature_importance(predictor, feature_names)
    suggested = get_suggested_data_for_prediction(importance, top_k=5)

    print("   特徵重要性排序（預測此目標最需要的資料）:")
    for i, imp in enumerate(importance[:10], 1):
        bar = "#" * int(imp["importance"] * 20) + "-" * (20 - int(imp["importance"] * 20))
        print(f"   {i:2}. {imp['feature']:20s} {bar} {imp['importance']:.4f}")
    print(f"\n   -> 預測 [{task_info['display_name']}] 建議優先收集的資料: {suggested}\n")

    # 4. 驗證預測
    print("4. 驗證預測...")
    n = min(5, X.shape[0])
    result = predictor.predict(X[:n], domain=task_info.get("domain", "custom"))
    print(f"   前 {n} 筆預測: {[round(p[0], 4) for p in result['prediction'][:n]]}")
    print(f"   置信度: {result['confidence']:.4f}\n")

    # 5. 儲存（目錄由 --models-dir / PREDICT_AI_MODELS_DIR 決定）
    models_root = resolve_models_dir(getattr(args, "models_dir", None))
    models_root.mkdir(parents=True, exist_ok=True)
    save_path = args.save or str(models_root / f"task_{task_id}.pkl")
    predictor._feature_names = feature_names
    predictor.save_model(save_path)
    print(f"5. 模型已儲存至 {save_path}")
    try:
        from artifact_registry import write_json

        write_json(os.path.join(run_dir, "model_path.txt"), {"path": save_path})
    except Exception:
        pass

    # 輸出摘要 JSON
    summary = {
        "task": task_id,
        "run_id": run_id,
        "task_info": task_info,
        "metrics": metrics,
        "walk_forward": wf_summary,
        "suggested_data": suggested,
        "feature_importance": importance[:10],
    }
    print("\n=== 摘要 JSON ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# --- 管線階段別名（供測試／外部 orchestrator hook） ---
def stage_precheck(args: Any) -> None:
    check_model_dependencies(
        args.model,
        automl_include_deep=bool(getattr(args, "automl_include_deep", False)),
    )


def stage_collect(task_id: str, config: Dict[str, Any], **override: Any):
    return crawl_and_build_xy(task_id, config, **override)


def stage_train(
    predictor: Any,
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    fit_kw: Dict[str, Any],
):
    return predictor.fit(X, y, model=model, **fit_kw)


if __name__ == "__main__":
    main()
