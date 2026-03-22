#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用預測器：支援未訓練過的預測目標

**Feature contract 層（優先使用本檔頂層函式）**
- ``map_to_canonical``：欄位對齊 canonical schema、缺值填 0
- ``CANONICAL_FEATURES``：跨任務欄位順序契約

**本檔 ``UniversalPredictor`` 類別**另含「多模型載入與簡化推論」便利層，
不負責：Optuna、ONNX 匯出、實驗紀錄、HTTP —— 請用 ``UnifiedPredictor``／管線模組。

- 通用模型：在 canonical 特徵空間上訓練，可泛化到新任務
- 最近任務回退：未知目標時，使用最相似已訓練任務的模型
- 零樣本預測：無任何模型時，使用簡單啟發式（均值、線性外推）
"""
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical 特徵順序（跨任務共用）
CANONICAL_FEATURES = [
    "open", "high", "low", "close", "volume",
    "sentiment", "temperature_2m", "relative_humidity_2m",
    "value", "new_cases", "new_deaths",
]


def map_to_canonical(
    X: np.ndarray,
    feature_names: List[str],
    canonical: List[str] = CANONICAL_FEATURES,
) -> np.ndarray:
    """將 X 對齊到 canonical 特徵空間，缺失填 0"""
    out = np.zeros((X.shape[0], len(canonical)))
    for i, cf in enumerate(canonical):
        if cf in feature_names:
            j = feature_names.index(cf)
            out[:, i] = X[:, j]
    return out


def feature_similarity(names_a: List[str], names_b: List[str]) -> float:
    """計算兩組特徵的相似度 [0,1]"""
    set_a, set_b = set(names_a), set(names_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


class UniversalPredictor:
    """
    通用預測器：可預測未訓練過的目標

    策略優先序：
    1. 若有該 task 專用模型 -> 使用
    2. 若有通用模型（universal）-> 使用
    3. 找最相似已訓練任務 -> 使用其模型
    4. 零樣本：回傳歷史均值或簡單外推
    """

    def __init__(
        self,
        model_dir: str = "models",
        schema_path: str = "config/prediction_schema.yaml",
    ) -> None:
        self.model_dir = model_dir
        self.schema_path = schema_path
        self._task_models: Dict[str, Any] = {}
        self._universal_model: Optional[Any] = None
        self._universal_scaler: Optional[Any] = None
        self._task_feature_names: Dict[str, List[str]] = {}
        self._canonical_features: List[str] = CANONICAL_FEATURES
        self._load_models()

    def _load_models(self) -> None:
        """載入已儲存的模型"""
        if not os.path.exists(self.model_dir):
            return
        for f in os.listdir(self.model_dir):
            if f.endswith(".pkl"):
                path = os.path.join(self.model_dir, f)
                try:
                    import pickle
                    with open(path, "rb") as fp:
                        state = pickle.load(fp)  # noqa: S301
                    model = state.get("model")
                    name = state.get("model_name", "linear")
                    task_id = f.replace(".pkl", "").replace("task_", "").replace("universal_", "")
                    if "universal" in f:
                        self._universal_model = model
                        self._universal_scaler = state.get("scaler")
                    else:
                        self._task_models[task_id] = {"model": model, "scaler": state.get("scaler")}
                        self._task_feature_names[task_id] = state.get("feature_names", [])
                except Exception as e:
                    logger.debug("載入 %s 失敗: %s", f, e)

    def _get_predictor_for_task(self, task_id: str) -> Optional[Any]:
        """取得任務對應的 UnifiedPredictor 或原始 model"""
        if task_id in self._task_models:
            return self._task_models[task_id]
        return None

    def _find_nearest_task(
        self,
        feature_names: List[str],
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        """找特徵最相似的已訓練任務"""
        best_task, best_sim = None, 0.0
        for tid, names in self._task_feature_names.items():
            if tid == exclude:
                continue
            sim = feature_similarity(feature_names, names)
            if sim > best_sim:
                best_sim = sim
                best_task = tid
        return best_task

    def predict(
        self,
        X: np.ndarray,
        task_id: str,
        feature_names: List[str],
        domain: str = "custom",
    ) -> Dict[str, Any]:
        """
        預測，支援未訓練過的 task_id。

        若 task_id 未訓練過，會嘗試：
        1. 通用模型
        2. 最相似任務的模型
        3. 零樣本啟發式
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 1. 專用模型
        pred_info = self._get_predictor_for_task(task_id)
        if pred_info:
            model, scaler = pred_info["model"], pred_info.get("scaler")
            X_in = X
            if scaler is not None:
                try:
                    X_in = scaler.transform(X)
                except Exception:
                    pass
            if hasattr(model, "predict"):
                preds = model.predict(X_in)
            else:
                coef = np.array(model.get("coef", []))
                mean_y = model.get("mean_y", 0.0)
                if coef.size == X_in.shape[1]:
                    preds = X_in @ coef
                else:
                    preds = np.full(X_in.shape[0], float(mean_y))
            return {
                "task_id": task_id,
                "source": "task_model",
                "prediction": np.asarray(preds).reshape(-1, 1).tolist(),
                "confidence": 0.8,
            }

        # 2. 通用模型
        if self._universal_model is not None:
            X_can = map_to_canonical(X, feature_names)
            if self._universal_scaler is not None:
                try:
                    X_can = self._universal_scaler.transform(X_can)
                except Exception:
                    pass
            if hasattr(self._universal_model, "predict"):
                preds = self._universal_model.predict(X_can)
            else:
                coef = np.array(self._universal_model.get("coef", []))
                mean_y = self._universal_model.get("mean_y", 0.0)
                if len(coef) == X_can.shape[1]:
                    preds = X_can @ coef
                else:
                    preds = np.full(X_can.shape[0], float(mean_y))
            return {
                "task_id": task_id,
                "source": "universal_model",
                "prediction": np.asarray(preds).reshape(-1, 1).tolist(),
                "confidence": 0.6,
            }

        # 3. 最近任務
        nearest = self._find_nearest_task(feature_names)
        if nearest:
            pred_info = self._get_predictor_for_task(nearest)
            if pred_info:
                model, scaler = pred_info["model"], pred_info.get("scaler")
                # 對齊特徵：nearest 的 feature_names 與當前 X 的 feature_names
                names_nearest = self._task_feature_names.get(nearest, [])
                X_aligned = np.zeros((X.shape[0], len(names_nearest)))
                for i, n in enumerate(names_nearest):
                    if n in feature_names:
                        j = feature_names.index(n)
                        X_aligned[:, i] = X[:, j]
                if scaler is not None:
                    try:
                        X_aligned = scaler.transform(X_aligned)
                    except Exception:
                        pass
                if hasattr(model, "predict"):
                    preds = model.predict(X_aligned)
                else:
                    coef = np.array(model.get("coef", []))
                    mean_y = model.get("mean_y", 0.0)
                    if len(coef) == X_aligned.shape[1]:
                        preds = X_aligned @ coef
                    else:
                        preds = np.full(X_aligned.shape[0], float(mean_y))
                return {
                    "task_id": task_id,
                    "source": f"nearest_task:{nearest}",
                    "prediction": np.asarray(preds).reshape(-1, 1).tolist(),
                    "confidence": 0.5,
                }

        # 4. 零樣本：簡單均值
        mean_val = float(np.mean(X)) if X.size > 0 else 0.0
        preds = np.full(X.shape[0], mean_val)
        return {
            "task_id": task_id,
            "source": "zero_shot_mean",
            "prediction": preds.reshape(-1, 1).tolist(),
            "confidence": 0.2,
        }

    def list_loaded_models(self) -> Dict[str, Any]:
        """回傳已載入的模型摘要"""
        return {
            "task_models": list(self._task_models.keys()),
            "has_universal": self._universal_model is not None,
        }


def main() -> None:
    """CLI：用通用預測器預測（支援未訓練過的目標）"""
    import argparse
    parser = argparse.ArgumentParser(description="通用預測器：預測未訓練過的目標")
    parser.add_argument("task_id", help="任務 ID（可為未訓練過的）")
    parser.add_argument("--features", nargs="+", default=["open", "high", "low", "close", "volume"],
                        help="特徵欄位名稱")
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="特徵值（若無則用隨機）")
    args = parser.parse_args()

    up = UniversalPredictor()
    info = up.list_loaded_models()
    print("已載入模型:", info)

    if args.values:
        X = np.array([args.values], dtype=float)
    else:
        X = np.random.randn(1, len(args.features)).astype(float)

    result = up.predict(X, args.task_id, args.features)
    print(f"\n預測結果 (source={result['source']}):")
    print(f"  預測值: {result['prediction']}")
    print(f"  置信度: {result['confidence']}")


if __name__ == "__main__":
    main()
