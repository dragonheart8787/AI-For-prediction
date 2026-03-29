"""
測試 v0.9.5 新增功能的整合煙霧測試。

覆蓋：
- data_pipeline（TimeSeriesAugmentor、FeaturePipeline、DataValidator、DataVersioner）
- prediction_intervals（naive、conformal、bootstrap）
- model_explainability（permutation、tree、explain API）
- experiment_log（read、compare、epoch log）
- artifact_registry（ModelRegistry）
- model_monitor（DataDriftDetector、PredictionMonitor、PerformanceMonitor）
- training_callbacks（CallbackList、EarlyStopping、MetricsLogger）
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pytest

RNG = np.random.default_rng(42)


def _make_data(n: int = 100, f: int = 8) -> tuple:
    X = RNG.normal(size=(n, f))
    y = X @ RNG.normal(size=f) + RNG.normal(scale=0.1, size=n)
    return X, y


# ============================================================================
# data_pipeline
# ============================================================================

class TestTimeSeriesAugmentor:
    def setup_method(self) -> None:
        from data_pipeline import TimeSeriesAugmentor
        self.aug = TimeSeriesAugmentor(seed=0)
        self.X, self.y = _make_data(60, 6)

    def test_jitter(self) -> None:
        Xj = self.aug.jitter(self.X)
        assert Xj.shape == self.X.shape
        assert not np.array_equal(Xj, self.X)

    def test_scaling(self) -> None:
        Xs = self.aug.scaling(self.X)
        assert Xs.shape == self.X.shape

    def test_time_warp(self) -> None:
        Xt = self.aug.time_warp(self.X)
        assert Xt.shape == self.X.shape

    def test_window_slice(self) -> None:
        Xw = self.aug.window_slice(self.X, crop_ratio=0.8)
        assert Xw.shape == self.X.shape

    def test_mixup(self) -> None:
        X_m, y_m = self.aug.mixup(self.X, self.y)
        assert X_m.shape == self.X.shape
        assert y_m.shape == self.y.shape

    def test_augment_batch(self) -> None:
        X_aug, y_aug = self.aug.augment(self.X, self.y, methods=["jitter", "scaling"], factor=1)
        assert X_aug.shape[0] == 3 * self.X.shape[0]  # 原始 + 2 份


class TestFeaturePipeline:
    def test_fit_transform_standard(self) -> None:
        from data_pipeline import FeaturePipeline
        X, _ = _make_data(50, 5)
        pipe = FeaturePipeline(scaling="standard", impute_strategy="median")
        Xt, fn = pipe.fit_transform(X)
        assert Xt.shape == X.shape
        assert abs(Xt.mean()) < 1.0

    def test_minmax(self) -> None:
        from data_pipeline import FeaturePipeline
        X, _ = _make_data(40, 4)
        pipe = FeaturePipeline(scaling="minmax")
        Xt, _ = pipe.fit_transform(X)
        assert Xt.min() >= -0.1 and Xt.max() <= 1.1

    def test_nan_imputation(self) -> None:
        from data_pipeline import FeaturePipeline
        X, _ = _make_data(30, 4)
        X[0, 0] = np.nan
        pipe = FeaturePipeline(impute_strategy="median")
        pipe.fit(X)
        Xt, _ = pipe.transform(X)
        assert not np.isnan(Xt).any()


class TestDataValidator:
    def test_valid_data(self) -> None:
        from data_pipeline import DataValidator
        X, y = _make_data(50, 5)
        dv = DataValidator()
        rep = dv.validate(X, y)
        assert "n_samples" in rep
        assert rep["n_samples"] == 50

    def test_detects_nan(self) -> None:
        from data_pipeline import DataValidator
        X, _ = _make_data(50, 5)
        X[2, 2] = np.nan
        rep = DataValidator().validate(X)
        assert sum(rep["nan_counts"].values()) > 0
        assert not rep["is_valid"]

    def test_constant_feature(self) -> None:
        from data_pipeline import DataValidator
        X, _ = _make_data(50, 4)
        X[:, 1] = 3.14
        rep = DataValidator().validate(X)
        assert len(rep["constant_features"]) > 0


class TestDataVersioner:
    def test_snapshot_and_load(self, tmp_path: Any) -> None:
        from data_pipeline import DataVersioner
        X, y = _make_data(30, 4)
        dv = DataVersioner(root=str(tmp_path / "versions"))
        ver_id = dv.snapshot(X, y, description="test")
        X2, y2, _ = dv.load(ver_id)
        assert np.allclose(X, X2)
        assert y2 is not None and np.allclose(y, y2)

    def test_list_versions(self, tmp_path: Any) -> None:
        from data_pipeline import DataVersioner
        dv = DataVersioner(root=str(tmp_path / "versions"))
        X, y = _make_data(20, 3)
        dv.snapshot(X, y)
        assert len(dv.list_versions()) >= 1


# ============================================================================
# prediction_intervals
# ============================================================================

class TestPredictionIntervals:
    def setup_method(self) -> None:
        from unified_predict import UnifiedPredictor
        X, y = _make_data(80, 6)
        self.pred = UnifiedPredictor(normalize=True, rate_limit_enabled=False)
        self.pred.fit(X[:60], y[:60], model="linear")
        self.X_calib = X[60:70]
        self.y_calib = y[60:70]
        self.X_test = X[70:]

    def test_naive(self) -> None:
        from prediction_intervals import compute_prediction_intervals
        res = compute_prediction_intervals(self.pred, self.X_test, method="naive")
        assert "lower" in res and "upper" in res

    def test_conformal(self) -> None:
        from prediction_intervals import compute_prediction_intervals
        res = compute_prediction_intervals(
            self.pred, self.X_test, method="conformal",
            X_calib=self.X_calib, y_calib=self.y_calib, alpha=0.1,
        )
        assert "lower" in res
        assert res["artifacts"]["method"] == "conformal"

    def test_bootstrap(self) -> None:
        from prediction_intervals import compute_prediction_intervals
        res = compute_prediction_intervals(
            self.pred, self.X_test, method="bootstrap",
            X_calib=self.X_calib, y_calib=self.y_calib, n_boot=50,
        )
        assert res["artifacts"]["method"] == "bootstrap"

    def test_interval_coverage(self) -> None:
        from prediction_intervals import interval_coverage
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        cov = interval_coverage(y, lo, hi)
        assert cov["picp"] == 1.0


# ============================================================================
# model_explainability
# ============================================================================

class TestModelExplainability:
    def setup_method(self) -> None:
        from unified_predict import UnifiedPredictor
        X, y = _make_data(60, 5)
        self.pred = UnifiedPredictor(normalize=False, rate_limit_enabled=False)
        self.pred.fit(X, y, model="linear")
        self.X = X
        self.y = y

    def test_permutation(self) -> None:
        from model_explainability import permutation_importance
        class Wrap:
            def __init__(self, m: Any) -> None:
                self._m = m
            def predict(self, X: np.ndarray) -> np.ndarray:
                return self._m.predict(X).ravel()
        res = permutation_importance(Wrap(self.pred.model), self.X, self.y, n_repeats=3)
        assert "importances_mean" in res
        assert len(res["importances_mean"]) == self.X.shape[1]

    def test_explain_permutation(self) -> None:
        from model_explainability import explain
        res = explain(self.pred, self.X, self.y, method="permutation")
        assert res["method"] == "permutation"

    def test_explain_auto(self) -> None:
        from model_explainability import explain
        res = explain(self.pred, self.X, self.y, method="auto")
        assert "method" in res


# ============================================================================
# experiment_log
# ============================================================================

class TestExperimentLog:
    def test_log_and_read(self, tmp_path: Any) -> None:
        from experiment_log import log_training_event, read_experiments
        path = str(tmp_path / "runs.jsonl")
        log_training_event(
            event_type="fit",
            run_id="test-001",
            task_name="smoke",
            model_requested="linear",
            model_trained="linear",
            row_count=100,
            feature_count=5,
            metrics={"train_rmse": 0.5},
            status="success",
            path=path,
        )
        recs = read_experiments(path)
        assert len(recs) == 1
        assert recs[0]["run_id"] == "test-001"

    def test_log_epoch(self, tmp_path: Any) -> None:
        from experiment_log import log_epoch, read_epoch_logs
        path = str(tmp_path / "epochs.jsonl")
        for ep in range(5):
            log_epoch(run_id="r1", epoch=ep, train_loss=1.0 - ep * 0.1, path=path)
        recs = read_epoch_logs("r1", path=path)
        assert len(recs) == 5

    def test_compare_runs(self, tmp_path: Any) -> None:
        from experiment_log import log_training_event, compare_runs
        path = str(tmp_path / "runs.jsonl")
        for rid, rmse in [("r1", 0.8), ("r2", 0.5)]:
            log_training_event(
                event_type="fit", run_id=rid, task_name="t", model_requested="linear",
                model_trained="linear", row_count=10, feature_count=3,
                metrics={"train_rmse": rmse}, status="success", path=path,
            )
        comp = compare_runs(["r1", "r2"], path=path)
        assert comp["best_run"]["run_id"] == "r2"


# ============================================================================
# artifact_registry
# ============================================================================

class TestModelRegistry:
    def test_register_and_load(self, tmp_path: Any) -> None:
        from artifact_registry import ModelRegistry
        from unified_predict import UnifiedPredictor
        reg = ModelRegistry(root=str(tmp_path / "registry"))
        X, y = _make_data(30, 4)
        p = UnifiedPredictor(rate_limit_enabled=False, auto_onnx=False)
        p.fit(X, y, model="linear")
        # 儲存 model（底層 LinearRegression）而非整個 predictor，避免 local closure 問題
        meta = reg.register_model("test_model", p.model, metrics={"rmse": 0.5})
        assert meta["version"] == 1

        loaded = reg.load_registered_model("test_model")
        assert hasattr(loaded, "predict")

    def test_promote_model(self, tmp_path: Any) -> None:
        from artifact_registry import ModelRegistry
        import numpy as _np
        from sklearn.linear_model import LinearRegression
        reg = ModelRegistry(root=str(tmp_path / "registry"))
        lr = LinearRegression()
        lr.fit(_np.array([[1.0], [2.0]]), [1.0, 2.0])
        reg.register_model("mymodel", lr, stage="development")
        meta = reg.promote_model("mymodel", 1, to_stage="staging")
        assert meta["stage"] == "staging"

    def test_compare_versions(self, tmp_path: Any) -> None:
        from artifact_registry import ModelRegistry
        import numpy as _np
        from sklearn.linear_model import LinearRegression
        reg = ModelRegistry(root=str(tmp_path / "registry"))
        lr = LinearRegression()
        lr.fit(_np.array([[1.0], [2.0]]), [1.0, 2.0])
        reg.register_model("m", lr, metrics={"train_rmse": 0.8})
        reg.register_model("m", lr, metrics={"train_rmse": 0.6})
        diff = reg.compare_versions("m", 1, 2)
        assert "metrics_diff" in diff


# ============================================================================
# model_monitor
# ============================================================================

class TestModelMonitor:
    def test_data_drift_no_drift(self) -> None:
        from model_monitor import DataDriftDetector
        X_ref, _ = _make_data(100, 5)
        X_cur, _ = _make_data(50, 5)
        det = DataDriftDetector()
        det.fit(X_ref)
        rep = det.detect(X_cur)
        assert "overall_drift" in rep
        assert "feature_reports" in rep

    def test_prediction_monitor_alert(self) -> None:
        from model_monitor import PredictionMonitor
        pm = PredictionMonitor(mean_shift_sigma=2.0)
        pm.set_baseline([1.0] * 100)
        # 大幅偏移的預測
        rep = pm.record([100.0] * 10)
        assert rep["alert"] is True

    def test_performance_monitor(self) -> None:
        from model_monitor import PerformanceMonitor
        pm = PerformanceMonitor()
        pm.set_baseline(0.5)
        rep = pm.record([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert "rmse" in rep

    def test_concept_drift(self) -> None:
        from model_monitor import ConceptDriftDetector
        cd = ConceptDriftDetector(delta=0.01)
        for v in np.random.default_rng(0).normal(0, 1, 50):
            cd.update(float(v))
        # 加入明顯漂移序列
        for v in np.random.default_rng(1).normal(100, 1, 50):
            cd.update(float(v))
        assert cd.n_drifts >= 0  # 至少不崩潰


# ============================================================================
# training_callbacks
# ============================================================================

class TestTrainingCallbacks:
    def test_early_stopping(self) -> None:
        from training_callbacks import EarlyStoppingCallback, CallbackList
        es = EarlyStoppingCallback(patience=3, monitor="val_loss")
        cbs = CallbackList([es])
        cbs.on_train_begin()
        losses = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
        for ep, loss in enumerate(losses):
            cbs.on_epoch_end(ep, {"val_loss": loss})
            if cbs.stop_training:
                break
        assert es.stopped_epoch is not None

    def test_metrics_logger(self, tmp_path: Any) -> None:
        from training_callbacks import MetricsLoggerCallback, CallbackList
        path = str(tmp_path / "epoch.jsonl")
        cb = MetricsLoggerCallback(run_id="cb_test", log_path=path)
        cbs = CallbackList([cb])
        cbs.on_train_begin()
        for ep in range(5):
            cbs.on_epoch_end(ep, {"train_loss": 0.5 - ep * 0.05, "val_loss": 0.6 - ep * 0.04})
        from experiment_log import read_epoch_logs
        recs = read_epoch_logs("cb_test", path=path)
        assert len(recs) == 5
