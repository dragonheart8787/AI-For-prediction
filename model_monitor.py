#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生產環境模型監控（Model Monitor）。

模組：
1. DataDriftDetector    — 資料分佈漂移偵測（KS 檢驗、PSI、Wasserstein 距離）
2. PredictionMonitor    — 預測分佈監控（均值 / 標準差追蹤，異常告警）
3. PerformanceMonitor   — 滾動性能指標（RMSE / MAE 趨勢）
4. ConceptDriftDetector — ADWIN 概念漂移（輕量實作，無需 River）
5. MonitoringDashboard  — 聚合報告

依賴：numpy（核心）；scipy（KS 檢驗，選配）
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as _scipy_stats  # type: ignore
except ImportError:
    _scipy_stats = None  # type: ignore


# ============================================================================
# 1. DataDriftDetector
# ============================================================================

class DataDriftDetector:
    """
    資料分佈漂移偵測。

    以參考分佈（訓練集）為基準，對每個特徵計算：
    - KS 統計量與 p-value（雙樣本 Kolmogorov-Smirnov）
    - PSI（Population Stability Index）
    - Wasserstein 距離（Earth Mover's Distance）

    use_scipy=False 時用純 numpy 近似 KS 統計量（不計算 p-value）。
    """

    def __init__(
        self,
        drift_pvalue_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        n_bins: int = 10,
    ) -> None:
        self.drift_pvalue_threshold = drift_pvalue_threshold
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self._reference: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, X_reference: np.ndarray, feature_names: Optional[List[str]] = None) -> "DataDriftDetector":
        """設定參考分佈。"""
        self._reference = np.asarray(X_reference, dtype=float)
        n, f = self._reference.shape
        self._feature_names = list(feature_names or [f"f{i}" for i in range(f)])
        return self

    def detect(self, X_current: np.ndarray) -> Dict[str, Any]:
        """
        偵測漂移。

        Returns
        -------
        {
            "overall_drift": bool,
            "drifted_features": [...],
            "feature_reports": {feature: {ks, psi, wasserstein, is_drifted}},
            "drift_score": float  (0~1，漂移特徵佔比)
        }
        """
        if self._reference is None:
            raise RuntimeError("請先呼叫 fit() 設定參考分佈。")
        X_cur = np.asarray(X_current, dtype=float)
        n, f = self._reference.shape
        fn = self._feature_names or [f"f{i}" for i in range(f)]
        drifted: List[str] = []
        reports: Dict[str, Any] = {}

        for j in range(min(f, X_cur.shape[1])):
            ref_col = self._reference[:, j]
            cur_col = X_cur[:, j]
            ref_col = ref_col[np.isfinite(ref_col)]
            cur_col = cur_col[np.isfinite(cur_col)]

            ks_stat, p_val = self._ks_test(ref_col, cur_col)
            psi = self._psi(ref_col, cur_col)
            wass = self._wasserstein(ref_col, cur_col)

            is_drifted = (p_val is not None and p_val < self.drift_pvalue_threshold) or (psi > self.psi_threshold)
            reports[fn[j]] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val) if p_val is not None else None,
                "psi": float(psi),
                "wasserstein": float(wass),
                "is_drifted": is_drifted,
            }
            if is_drifted:
                drifted.append(fn[j])

        drift_score = len(drifted) / max(1, f)
        return {
            "overall_drift": len(drifted) > 0,
            "drifted_features": drifted,
            "n_drifted": len(drifted),
            "drift_score": drift_score,
            "feature_reports": reports,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _ks_test(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[float, Optional[float]]:
        if len(ref) < 2 or len(cur) < 2:
            return 0.0, None
        if _scipy_stats is not None:
            result = _scipy_stats.ks_2samp(ref, cur)
            return float(result.statistic), float(result.pvalue)
        # Pure numpy 近似
        all_vals = np.concatenate([ref, cur])
        all_vals.sort()
        cdf_ref = np.searchsorted(np.sort(ref), all_vals, side="right") / len(ref)
        cdf_cur = np.searchsorted(np.sort(cur), all_vals, side="right") / len(cur)
        ks = float(np.max(np.abs(cdf_ref - cdf_cur)))
        return ks, None

    def _psi(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Population Stability Index（PSI）。"""
        bins = np.linspace(
            min(ref.min(), cur.min()) - 1e-9,
            max(ref.max(), cur.max()) + 1e-9,
            self.n_bins + 1,
        )
        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)
        ref_pct = (ref_counts + 1e-9) / (len(ref) + 1e-9 * self.n_bins)
        cur_pct = (cur_counts + 1e-9) / (len(cur) + 1e-9 * self.n_bins)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(0.0, psi)

    def _wasserstein(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Wasserstein-1 距離（1-D 閉合公式）。"""
        if len(ref) < 1 or len(cur) < 1:
            return 0.0
        ref_s = np.sort(ref)
        cur_s = np.sort(cur)
        n = max(len(ref_s), len(cur_s))
        # 插值到同一長度再計算
        idx = np.linspace(0, len(ref_s) - 1, n)
        ref_i = np.interp(idx, np.arange(len(ref_s)), ref_s)
        idx2 = np.linspace(0, len(cur_s) - 1, n)
        cur_i = np.interp(idx2, np.arange(len(cur_s)), cur_s)
        return float(np.mean(np.abs(ref_i - cur_i)))


# ============================================================================
# 2. PredictionMonitor
# ============================================================================

class PredictionMonitor:
    """
    預測分佈監控。

    追蹤每批預測的均值、標準差，當偏移超過閾值時觸發警報。
    """

    def __init__(
        self,
        window_size: int = 50,
        mean_shift_sigma: float = 3.0,
        std_change_ratio: float = 2.0,
    ) -> None:
        self.window_size = window_size
        self.mean_shift_sigma = mean_shift_sigma
        self.std_change_ratio = std_change_ratio
        self._history: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None

    def set_baseline(self, predictions: Sequence[float]) -> None:
        """以初始預測集設定基準線。"""
        arr = np.asarray(predictions, dtype=float)
        self._baseline_mean = float(np.mean(arr))
        self._baseline_std = float(np.std(arr)) or 1.0
        logger.info("Baseline 設定：mean=%.4f  std=%.4f", self._baseline_mean, self._baseline_std)

    def record(self, predictions: Sequence[float]) -> Dict[str, Any]:
        """
        紀錄一批預測，回傳是否觸發告警。

        Returns
        -------
        {"alert": bool, "mean": ..., "std": ..., "reason": str}
        """
        arr = np.asarray(predictions, dtype=float)
        if len(arr) == 0:
            return {"alert": False, "mean": None, "std": None, "reason": "empty"}

        batch_mean = float(np.mean(arr))
        batch_std = float(np.std(arr)) or 0.0
        self._history.append({"mean": batch_mean, "std": batch_std, "ts": time.time()})

        alerts: List[str] = []
        if self._baseline_mean is not None and self._baseline_std is not None:
            z = abs(batch_mean - self._baseline_mean) / (self._baseline_std + 1e-9)
            if z > self.mean_shift_sigma:
                alerts.append(f"均值偏移 z={z:.2f}（閾值={self.mean_shift_sigma}）")
            if self._baseline_std > 0 and batch_std / self._baseline_std > self.std_change_ratio:
                alerts.append(f"標準差暴增 ratio={batch_std/self._baseline_std:.2f}")

        return {
            "alert": len(alerts) > 0,
            "reasons": alerts,
            "mean": batch_mean,
            "std": batch_std,
            "n_samples": len(arr),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def summary(self) -> Dict[str, Any]:
        """回傳歷史統計摘要。"""
        if not self._history:
            return {"n_batches": 0}
        means = [h["mean"] for h in self._history]
        stds = [h["std"] for h in self._history]
        return {
            "n_batches": len(self._history),
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "mean_of_stds": float(np.mean(stds)),
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std,
        }


# ============================================================================
# 3. PerformanceMonitor
# ============================================================================

class PerformanceMonitor:
    """
    滾動性能監控（追蹤 RMSE / MAE 趨勢，偵測性能劣化）。
    """

    def __init__(
        self,
        window_size: int = 20,
        rmse_degradation_ratio: float = 1.5,
    ) -> None:
        self.window_size = window_size
        self.rmse_degradation_ratio = rmse_degradation_ratio
        self._history: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self._baseline_rmse: Optional[float] = None

    def set_baseline(self, baseline_rmse: float) -> None:
        self._baseline_rmse = float(baseline_rmse)

    def record(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        *,
        label: str = "",
    ) -> Dict[str, Any]:
        """
        紀錄一批真實值與預測值，計算並儲存指標。

        Returns
        -------
        {"rmse": ..., "mae": ..., "r2": ..., "alert": bool}
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        n = min(len(yt), len(yp))
        yt, yp = yt[:n], yp[:n]

        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        entry = {"rmse": rmse, "mae": mae, "r2": r2, "n": n, "ts": time.time()}
        self._history.append(entry)

        alert = False
        reason = ""
        if self._baseline_rmse is not None and rmse > self._baseline_rmse * self.rmse_degradation_ratio:
            alert = True
            reason = f"RMSE={rmse:.4f} 超過基準 {self._baseline_rmse:.4f}×{self.rmse_degradation_ratio}"
            logger.warning("性能劣化告警（%s）：%s", label, reason)

        return {"rmse": rmse, "mae": mae, "r2": r2, "alert": alert, "reason": reason, "n_samples": n}

    def trend(self) -> Dict[str, Any]:
        """回傳滾動窗口內的指標趨勢。"""
        if not self._history:
            return {"n_batches": 0}
        rmse_list = [h["rmse"] for h in self._history]
        mae_list = [h["mae"] for h in self._history]
        return {
            "n_batches": len(self._history),
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "rmse_trend": "worsening" if len(rmse_list) > 2 and rmse_list[-1] > rmse_list[0] else "stable",
            "mae_mean": float(np.mean(mae_list)),
            "baseline_rmse": self._baseline_rmse,
        }


# ============================================================================
# 4. ConceptDriftDetector（輕量 ADWIN 近似）
# ============================================================================

class ConceptDriftDetector:
    """
    輕量概念漂移偵測（ADWIN 窗口均值偏移近似）。

    每次呼叫 update(value) 後自動縮小窗口直到統計均一，
    當縮小發生時回傳 drift=True（概念漂移）。

    無需 River 等外部套件。
    """

    def __init__(self, delta: float = 0.002, max_window: int = 300) -> None:
        self.delta = delta
        self.max_window = max_window
        self._window: Deque[float] = deque(maxlen=max_window)
        self.drift_detected = False
        self.n_drifts = 0

    def update(self, value: float) -> bool:
        """
        更新一筆觀察值。

        Returns True 若偵測到概念漂移（窗口縮小）。
        """
        self._window.append(float(value))
        w = list(self._window)
        n = len(w)
        if n < 10:
            self.drift_detected = False
            return False

        arr = np.array(w)
        # 滑動切分：比較前半 vs 後半
        best_split = -1
        for i in range(n // 4, 3 * n // 4):
            w1 = arr[:i]
            w2 = arr[i:]
            m1, m2 = w1.mean(), w2.mean()
            s1 = w1.var() / len(w1) if len(w1) > 0 else 0
            s2 = w2.var() / len(w2) if len(w2) > 0 else 0
            epsilon_cut = np.sqrt((s1 + s2) * np.log(4 * n / self.delta) / 2)
            if abs(m1 - m2) > epsilon_cut:
                best_split = i
                break

        if best_split > 0:
            # 縮小窗口（保留後半）
            new_w = list(arr[best_split:])
            self._window.clear()
            self._window.extend(new_w)
            self.drift_detected = True
            self.n_drifts += 1
            logger.info("概念漂移偵測（ADWIN）：窗口縮小至 %d 個觀察值", len(new_w))
            return True

        self.drift_detected = False
        return False

    def reset(self) -> None:
        self._window.clear()
        self.drift_detected = False

    @property
    def window_mean(self) -> Optional[float]:
        return float(np.mean(list(self._window))) if self._window else None


# ============================================================================
# 5. MonitoringDashboard
# ============================================================================

class MonitoringDashboard:
    """
    聚合所有監控器的報告，並支援將報告序列化為 JSON。
    """

    def __init__(
        self,
        data_drift_detector: Optional[DataDriftDetector] = None,
        prediction_monitor: Optional[PredictionMonitor] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        concept_drift_detector: Optional[ConceptDriftDetector] = None,
        log_path: str = "data/monitoring_log.jsonl",
    ) -> None:
        self.data_drift = data_drift_detector
        self.pred_monitor = prediction_monitor
        self.perf_monitor = performance_monitor
        self.concept_drift = concept_drift_detector
        self.log_path = log_path

    def report(
        self,
        X_current: Optional[np.ndarray] = None,
        predictions: Optional[Sequence[float]] = None,
        y_true: Optional[Sequence[float]] = None,
        y_pred: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        執行所有已配置的偵測器，彙整報告。

        Parameters
        ----------
        X_current   : 當前批次特徵（DataDriftDetector 用）
        predictions : 模型預測值（PredictionMonitor 用）
        y_true      : 真實值（PerformanceMonitor 用）
        y_pred      : 對應預測值（PerformanceMonitor 用）
        """
        rep: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alerts": [],
        }

        if self.data_drift is not None and X_current is not None:
            drift_rep = self.data_drift.detect(X_current)
            rep["data_drift"] = drift_rep
            if drift_rep.get("overall_drift"):
                rep["alerts"].append(f"資料漂移：{drift_rep['drifted_features']}")

        if self.pred_monitor is not None and predictions is not None:
            pred_rep = self.pred_monitor.record(predictions)
            rep["prediction_monitor"] = pred_rep
            if pred_rep.get("alert"):
                rep["alerts"].extend(pred_rep.get("reasons", []))

        if self.perf_monitor is not None and y_true is not None and y_pred is not None:
            perf_rep = self.perf_monitor.record(y_true, y_pred)
            rep["performance_monitor"] = perf_rep
            if perf_rep.get("alert"):
                rep["alerts"].append(f"性能劣化：{perf_rep.get('reason','')}")

        rep["has_alert"] = len(rep["alerts"]) > 0

        # 寫入 log
        self._append_log(rep)
        return rep

    def _append_log(self, rec: Dict[str, Any]) -> None:
        try:
            d = os.path.dirname(os.path.abspath(self.log_path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as ex:
            logger.warning("監控日誌寫入失敗：%s", ex)

    def read_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        ap = os.path.abspath(self.log_path)
        if not os.path.isfile(ap):
            return []
        recs: List[Dict[str, Any]] = []
        with open(ap, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return recs[-limit:]
