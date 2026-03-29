#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料前處理與增強管線（Data Pipeline）。

模組：
1. TimeSeriesAugmentor  — 時序資料增強（Jitter、縮放、時間翹曲、Mixup）
2. FeaturePipeline      — 特徵工程管線（插補、縮放、類別編碼、lag 特徵、技術指標）
3. DataValidator        — 資料品質與洩漏檢測
4. DataVersioner        — 雜湊型資料版本管理（追蹤資料快照）
5. WalkForwardDataset   — 產生滾動視窗訓練批次（適合時序）

依賴：numpy（核心）；pandas / scipy（選配，自動降級）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    from scipy import stats as _scipy_stats  # type: ignore
except ImportError:
    _scipy_stats = None  # type: ignore


# ============================================================================
# 1. TimeSeriesAugmentor
# ============================================================================

class TimeSeriesAugmentor:
    """
    時序表格資料增強（訓練時使用，驗證/推論時關閉）。

    支援方法：
    - jitter     : 加入高斯噪音
    - scaling    : 隨機振幅縮放
    - time_warp  : 時間軸輕微扭曲（重複/插值）
    - window_slice: 隨機截取子序列
    - mixup      : 兩樣本線性混合
    - magnitude_warp: 以平滑隨機曲線縮放振幅

    用法：
        aug = TimeSeriesAugmentor(seed=42)
        X_aug, y_aug = aug.augment(X, y, methods=["jitter", "scaling"], factor=2)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def jitter(self, X: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """加入均值 0、標準差 sigma × std(X) 的高斯噪音。"""
        std = np.std(X, axis=0, keepdims=True)
        std[std == 0] = 1.0
        noise = self.rng.normal(0, sigma, size=X.shape) * std
        return X + noise

    def scaling(self, X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """每條序列整體隨機縮放（log-normal 分佈）。"""
        scale = np.exp(self.rng.normal(0, sigma, size=(X.shape[0], 1)))
        return X * scale

    def time_warp(self, X: np.ndarray, *, n_knots: int = 4, sigma: float = 0.2) -> np.ndarray:
        """以隨機樣條對時間軸輕微扭曲（近似 Dynamic Time Warping 擴充）。"""
        n, f = X.shape
        # 隨機生成扭曲路徑（以線性插值近似）
        knot_x = np.linspace(0, n - 1, n_knots + 2)
        knot_y = knot_x + self.rng.normal(0, sigma * n, size=knot_x.shape)
        knot_y = np.clip(knot_y, 0, n - 1)
        knot_y[0] = 0.0
        knot_y[-1] = float(n - 1)
        warp_path = np.interp(np.arange(n), knot_x, knot_y).astype(int)
        warp_path = np.clip(warp_path, 0, n - 1)
        return X[warp_path]

    def window_slice(self, X: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """隨機截取 crop_ratio 比例的子序列，再線性插值回原長度。"""
        n, f = X.shape
        crop_len = max(2, int(n * crop_ratio))
        start = self.rng.integers(0, max(1, n - crop_len + 1))
        sliced = X[start : start + crop_len]
        # 插值回 n
        new_idx = np.linspace(0, len(sliced) - 1, n)
        out = np.zeros_like(X)
        for j in range(f):
            out[:, j] = np.interp(new_idx, np.arange(len(sliced)), sliced[:, j])
        return out

    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup：隨機兩條樣本的線性混合。"""
        n = X.shape[0]
        idx = self.rng.permutation(n)
        lam = self.rng.beta(alpha, alpha, size=(n, 1))
        X_mix = lam * X + (1 - lam) * X[idx]
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_mix = lam * y_arr + (1 - lam) * y_arr[idx]
        return X_mix, y_mix.ravel() if y_mix.shape[1] == 1 else y_mix

    def magnitude_warp(self, X: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
        """以光滑隨機曲線逐點縮放振幅。"""
        n, f = X.shape
        knot_x = np.linspace(0, n - 1, n_knots + 2)
        curve = np.exp(self.rng.normal(0, sigma, size=knot_x.shape))
        warper = np.interp(np.arange(n), knot_x, curve).reshape(-1, 1)
        return X * warper

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: Sequence[str] = ("jitter", "scaling"),
        factor: int = 1,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批次增強：以 methods 中的方法各生成 factor 份，拼接在原始資料後。

        Returns
        -------
        X_aug, y_aug （原始 + 增強，樣本數 = (1 + factor * len(methods)) * N）
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xs = [X]
        ys = [y]

        for _ in range(factor):
            for m in methods:
                m = m.lower()
                if m == "jitter":
                    Xs.append(self.jitter(X, **{k: v for k, v in method_kwargs.items() if k == "sigma"}))
                    ys.append(y)
                elif m == "scaling":
                    Xs.append(self.scaling(X))
                    ys.append(y)
                elif m == "time_warp":
                    Xs.append(self.time_warp(X))
                    ys.append(y)
                elif m == "window_slice":
                    Xs.append(self.window_slice(X))
                    ys.append(y)
                elif m == "mixup":
                    X_m, y_m = self.mixup(X, y)
                    Xs.append(X_m)
                    ys.append(y_m)
                elif m == "magnitude_warp":
                    Xs.append(self.magnitude_warp(X))
                    ys.append(y)
                else:
                    logger.warning("未知增強方法：%s", m)

        return np.vstack(Xs), np.concatenate(ys)


# ============================================================================
# 2. FeaturePipeline
# ============================================================================

class FeaturePipeline:
    """
    特徵工程管線（fit → transform 模式，可持久化）。

    步驟（依序）：
    1. 缺失值插補（median / mean / zero / forward_fill）
    2. 標準化（standard / minmax / robust / none）
    3. 技術指標衍生（RSI、MACD、BB 等，需 pandas）
    4. 類別欄編碼（label encoding → 整數）
    5. Lag + 滾動統計（已由 feature_expansion 提供，此處為獨立實作）
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        scaling: str = "standard",
        add_technical_indicators: bool = False,
        add_lag_features: bool = False,
        lag_steps: Sequence[int] = (1, 2, 3),
        rolling_windows: Sequence[int] = (5, 10),
        clip_outliers_sigma: float = 0.0,
    ) -> None:
        self.impute_strategy = impute_strategy
        self.scaling = scaling
        self.add_technical_indicators = add_technical_indicators
        self.add_lag_features = add_lag_features
        self.lag_steps = list(lag_steps)
        self.rolling_windows = list(rolling_windows)
        self.clip_outliers_sigma = float(clip_outliers_sigma)
        # Fit 後填充的統計量
        self._medians: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._scale_params: Optional[Dict[str, np.ndarray]] = None
        self.feature_names_out_: Optional[List[str]] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeaturePipeline":
        X = np.asarray(X, dtype=float)
        n, f = X.shape
        self._medians = np.nanmedian(X, axis=0)
        self._means = np.nanmean(X, axis=0)
        self._medians = np.where(np.isnan(self._medians), 0.0, self._medians)
        self._means = np.where(np.isnan(self._means), 0.0, self._means)

        # 插補後計算縮放統計
        X_imp = self._impute(X)

        sc = self.scaling.lower()
        if sc == "standard":
            mu = X_imp.mean(axis=0)
            sig = X_imp.std(axis=0)
            sig[sig == 0] = 1.0
            self._scale_params = {"mu": mu, "sig": sig}
        elif sc == "minmax":
            lo = X_imp.min(axis=0)
            hi = X_imp.max(axis=0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            self._scale_params = {"lo": lo, "rng": rng}
        elif sc == "robust":
            med = np.median(X_imp, axis=0)
            q75 = np.percentile(X_imp, 75, axis=0)
            q25 = np.percentile(X_imp, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            self._scale_params = {"med": med, "iqr": iqr}
        else:
            self._scale_params = {}

        fn = list(feature_names or [f"f{i}" for i in range(f)])
        self.feature_names_out_ = fn
        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Returns (X_transformed, feature_names_out)
        """
        if not self._fitted:
            raise RuntimeError("請先呼叫 fit()。")
        X = np.asarray(X, dtype=float)
        fn = list(feature_names or self.feature_names_out_ or [f"f{i}" for i in range(X.shape[1])])

        # 離群值裁切
        if self.clip_outliers_sigma > 0:
            X = self._clip_outliers(X)

        # 插補
        X = self._impute(X)

        # 縮放
        X = self._scale(X)

        # Lag 特徵
        if self.add_lag_features and pd is not None:
            X, fn = self._add_lag(X, fn)

        # 技術指標（需 pandas）
        if self.add_technical_indicators and pd is not None:
            X, fn = self._add_technical(X, fn)

        return X, fn

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        return self.fit(X, feature_names).transform(X, feature_names)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        nan_mask = np.isnan(X)
        if not nan_mask.any():
            return X
        s = self.impute_strategy.lower()
        for j in range(X.shape[1]):
            col_nan = nan_mask[:, j]
            if not col_nan.any():
                continue
            if s == "median":
                fill = float(self._medians[j]) if self._medians is not None else 0.0
            elif s == "mean":
                fill = float(self._means[j]) if self._means is not None else 0.0
            elif s == "zero":
                fill = 0.0
            elif s == "forward_fill":
                vals = X[:, j].copy()
                last = 0.0
                for i in range(len(vals)):
                    if np.isnan(vals[i]):
                        vals[i] = last
                    else:
                        last = vals[i]
                X[:, j] = vals
                continue
            else:
                fill = 0.0
            X[col_nan, j] = fill
        return X

    def _scale(self, X: np.ndarray) -> np.ndarray:
        sp = self._scale_params or {}
        sc = self.scaling.lower()
        if sc == "standard" and "mu" in sp:
            return (X - sp["mu"]) / sp["sig"]
        if sc == "minmax" and "lo" in sp:
            return (X - sp["lo"]) / sp["rng"]
        if sc == "robust" and "med" in sp:
            return (X - sp["med"]) / sp["iqr"]
        return X

    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=0)
        sig = np.nanstd(X, axis=0)
        sig[sig == 0] = 1.0
        lo = mu - self.clip_outliers_sigma * sig
        hi = mu + self.clip_outliers_sigma * sig
        return np.clip(X, lo, hi)

    def _add_lag(self, X: np.ndarray, fn: List[str]) -> Tuple[np.ndarray, List[str]]:
        df = pd.DataFrame(X, columns=fn)
        new_cols: List[str] = []
        for lag in self.lag_steps:
            for col in fn:
                name = f"{col}_lag{lag}"
                df[name] = df[col].shift(lag)
                new_cols.append(name)
        for win in self.rolling_windows:
            for col in fn:
                df[f"{col}_rm{win}"] = df[col].rolling(win, min_periods=1).mean()
                df[f"{col}_rs{win}"] = df[col].rolling(win, min_periods=1).std()
                new_cols.extend([f"{col}_rm{win}", f"{col}_rs{win}"])
        df = df.fillna(0.0)
        all_cols = fn + new_cols
        return df[all_cols].to_numpy(dtype=np.float64), all_cols

    def _add_technical(self, X: np.ndarray, fn: List[str]) -> Tuple[np.ndarray, List[str]]:
        """加入 RSI(14)、EMA(12/26) 和 Bollinger Band（若有 close 欄位）。"""
        df = pd.DataFrame(X, columns=fn)
        new_cols: List[str] = []
        close_col = next((c for c in fn if "close" in c.lower()), None)
        if close_col is None:
            return X, fn
        s = df[close_col].copy()

        # RSI
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta).clip(lower=0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)
        new_cols.append("rsi_14")

        # EMA
        df["ema_12"] = s.ewm(span=12, adjust=False).mean()
        df["ema_26"] = s.ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        new_cols.extend(["ema_12", "ema_26", "macd"])

        # Bollinger Bands
        df["bb_mid"] = s.rolling(20, min_periods=1).mean()
        df["bb_std"] = s.rolling(20, min_periods=1).std().fillna(0.0)
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        new_cols.extend(["bb_mid", "bb_upper", "bb_lower"])

        all_cols = fn + new_cols
        return df[all_cols].fillna(0.0).to_numpy(dtype=np.float64), all_cols


# ============================================================================
# 3. DataValidator
# ============================================================================

class DataValidator:
    """
    資料品質與洩漏偵測。

    validate() 回傳 report dict，包含：
    - nan_counts, inf_counts（每特徵）
    - constant_features（零方差）
    - high_correlation_pairs（相關係數 > threshold）
    - target_leakage_risk（特徵與目標相關 > leak_threshold）
    - temporal_leakage（時間序反向排序偵測）
    - distribution_summary（均值、標準差、偏度、峰度）
    """

    def validate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        *,
        correlation_threshold: float = 0.95,
        leak_threshold: float = 0.99,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=float)
        n, f = X.shape
        fn = list(feature_names or [f"f{i}" for i in range(f)])
        report: Dict[str, Any] = {"n_samples": n, "n_features": f, "issues": []}

        # NaN / Inf
        nan_counts = np.isnan(X).sum(axis=0).tolist()
        inf_counts = np.isinf(X).sum(axis=0).tolist()
        report["nan_counts"] = dict(zip(fn, nan_counts))
        report["inf_counts"] = dict(zip(fn, inf_counts))
        n_nan_total = int(np.isnan(X).sum())
        if n_nan_total > 0:
            report["issues"].append(f"含有 {n_nan_total} 個 NaN 值")

        # 常數特徵
        stds = np.nanstd(X, axis=0)
        const_feats = [fn[j] for j in range(f) if stds[j] < 1e-12]
        report["constant_features"] = const_feats
        if const_feats:
            report["issues"].append(f"常數特徵（零方差）：{const_feats}")

        # 高相關
        X_clean = np.nan_to_num(X, nan=0.0)
        if f < 200:
            corr = np.corrcoef(X_clean.T)
            high_corr: List[str] = []
            for i in range(f):
                for j in range(i + 1, f):
                    if abs(corr[i, j]) >= correlation_threshold:
                        high_corr.append(f"{fn[i]} ↔ {fn[j]} ({corr[i, j]:.3f})")
            report["high_correlation_pairs"] = high_corr
            if high_corr:
                report["issues"].append(f"{len(high_corr)} 對高相關特徵（≥ {correlation_threshold}）")

        # 目標洩漏
        if y is not None:
            y_arr = np.asarray(y, dtype=float).ravel()
            m = min(len(y_arr), n)
            y_c = y_arr[:m]
            X_c = X_clean[:m]
            leaky: List[str] = []
            for j in range(f):
                if np.std(X_c[:, j]) < 1e-12:
                    continue
                c = float(np.corrcoef(X_c[:, j], y_c)[0, 1])
                if abs(c) >= leak_threshold:
                    leaky.append(f"{fn[j]} (corr={c:.4f})")
            report["target_leakage_risk"] = leaky
            if leaky:
                report["issues"].append(f"疑似目標洩漏（corr≥{leak_threshold}）：{leaky}")

        # 分佈摘要
        dist: Dict[str, Any] = {}
        for j in range(min(f, 50)):
            col = X_clean[:, j]
            dist[fn[j]] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }
            if _scipy_stats is not None:
                if float(np.std(col)) < 1e-15:
                    dist[fn[j]]["skewness"] = 0.0
                    dist[fn[j]]["kurtosis"] = 0.0
                else:
                    with np.errstate(all="ignore"):
                        dist[fn[j]]["skewness"] = float(_scipy_stats.skew(col))
                        dist[fn[j]]["kurtosis"] = float(_scipy_stats.kurtosis(col))
        report["distribution_summary"] = dist
        report["is_valid"] = len(report["issues"]) == 0
        return report


# ============================================================================
# 4. DataVersioner
# ============================================================================

class DataVersioner:
    """
    雜湊型資料版本管理（追蹤資料快照，無需 DVC）。

    用法：
        dv = DataVersioner("data/versions")
        ver_id = dv.snapshot(X, y, meta={"source": "yahoo"})
        X2, y2, meta2 = dv.load(ver_id)
    """

    def __init__(self, root: str = "data/versions") -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._index_path = os.path.join(self.root, "index.json")
        self._index: List[Dict[str, Any]] = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def _hash_array(self, arr: np.ndarray) -> str:
        a = np.ascontiguousarray(arr)
        return hashlib.sha256(a.view(np.uint8).tobytes()[:1_000_000]).hexdigest()[:16]

    def snapshot(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> str:
        """儲存資料快照，回傳 version_id（雜湊前綴）。"""
        X = np.asarray(X, dtype=float)
        h_x = self._hash_array(X)
        h_y = self._hash_array(np.asarray(y, dtype=float)) if y is not None else "none"
        ver_id = f"{h_x[:8]}_{h_y[:8]}"

        # 避免重複儲存
        existing = [v for v in self._index if v.get("version_id") == ver_id]
        if existing:
            return ver_id

        ver_dir = os.path.join(self.root, ver_id)
        os.makedirs(ver_dir, exist_ok=True)
        np.save(os.path.join(ver_dir, "X.npy"), X)
        if y is not None:
            np.save(os.path.join(ver_dir, "y.npy"), np.asarray(y, dtype=float))

        from datetime import datetime, timezone
        entry: Dict[str, Any] = {
            "version_id": ver_id,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "has_y": y is not None,
            "description": description,
            "meta": meta or {},
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._index.append(entry)
        self._save_index()
        logger.info("資料快照已儲存：%s（%d×%d）", ver_id, X.shape[0], X.shape[1])
        return ver_id

    def load(self, version_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """載入版本資料（X, y, meta）。"""
        ver_dir = os.path.join(self.root, version_id)
        if not os.path.isdir(ver_dir):
            raise FileNotFoundError(f"版本不存在：{version_id}")
        X = np.load(os.path.join(ver_dir, "X.npy"))
        y_path = os.path.join(ver_dir, "y.npy")
        y = np.load(y_path) if os.path.isfile(y_path) else None
        entry = next((v for v in self._index if v.get("version_id") == version_id), {})
        return X, y, dict(entry.get("meta", {}))

    def list_versions(self) -> List[Dict[str, Any]]:
        return list(self._index)


# ============================================================================
# 5. WalkForwardDataset
# ============================================================================

def walk_forward_batches(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_size: int,
    val_size: int,
    step: int = 1,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    產生滾動窗口訓練批次（Walk-Forward）。

    Parameters
    ----------
    train_size : 每個訓練窗口的樣本數
    val_size   : 每個驗證窗口的樣本數
    step       : 每次滑動的步長
    gap        : 訓練集結尾與驗證集開頭之間的間隔（防洩漏）

    Yields (X_train, y_train, X_val, y_val) tuples。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    result: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        tr_end = start + train_size
        va_start = tr_end + gap
        va_end = va_start + val_size
        if va_end > n:
            break
        result.append((X[start:tr_end], y[start:tr_end], X[va_start:va_end], y[va_start:va_end]))
        start += step
    return result
