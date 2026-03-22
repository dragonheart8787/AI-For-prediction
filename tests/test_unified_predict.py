"""
UnifiedPredictor 完整測試套件
覆蓋：fit, predict, predict_many, evaluate, save/load, normalize,
      快取、限流、多地平線、ONNX 包裝、輸入驗證、dict/DataFrame 輸入等。
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_predict import UnifiedPredictor, _LRUCache, _TokenBucket

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def sample_data(rng):
    X = rng.normal(size=(200, 4))
    w = rng.normal(size=(4,))
    y = X @ w + rng.normal(scale=0.1, size=(200,))
    return X, y


@pytest.fixture()
def multi_horizon_data(rng):
    X = rng.normal(size=(200, 4))
    w = rng.normal(size=(4,))
    y1 = X @ w + rng.normal(scale=0.1, size=(200,))
    y2 = X @ (w * 0.5) + rng.normal(scale=0.15, size=(200,))
    y = np.stack([y1, y2], axis=1)
    return X, y


@pytest.fixture()
def fitted_predictor(sample_data):
    X, y = sample_data
    p = UnifiedPredictor(auto_onnx=False)
    p.fit(X, y, model="linear")
    return p, X, y


# ---------------------------------------------------------------------------
# 基本 fit / predict
# ---------------------------------------------------------------------------

class TestFitPredict:

    def test_linear_fit_returns_metrics(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        metrics = p.fit(X, y, model="linear")
        assert "train_rmse" in metrics
        assert "train_mae" in metrics
        assert "train_r2" in metrics
        assert metrics["train_r2"] > 0.9

    def test_predict_returns_dict(self, fitted_predictor):
        p, X, _ = fitted_predictor
        result = p.predict(X[:5])
        assert isinstance(result, dict)
        for key in ("domain", "horizons", "model", "prediction", "confidence", "n_samples"):
            assert key in result

    def test_predict_shape(self, fitted_predictor):
        p, X, _ = fitted_predictor
        result = p.predict(X[:10])
        preds = result["prediction"]
        assert len(preds) == 10
        assert all(isinstance(row, list) for row in preds)

    def test_predict_before_fit_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(RuntimeError, match="尚未訓練"):
            p.predict([[1.0, 2.0]])

    def test_fit_dimension_mismatch_raises(self, rng):
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=(40,))
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(ValueError, match="樣本數不一致"):
            p.fit(X, y)

    def test_fit_empty_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(ValueError, match="不可為空"):
            p.fit(np.array([]).reshape(0, 3), np.array([]))


# ---------------------------------------------------------------------------
# 多地平線
# ---------------------------------------------------------------------------

class TestMultiHorizon:

    def test_multi_horizon_fit_predict(self, multi_horizon_data):
        X, y = multi_horizon_data
        p = UnifiedPredictor(auto_onnx=False)
        metrics = p.fit(X[:150], y[:150], model="linear")
        assert metrics["train_r2"] > 0.5

        result = p.predict(X[150:160], domain="financial")
        preds = result["prediction"]
        assert len(preds) == 10
        assert len(preds[0]) == 2


# ---------------------------------------------------------------------------
# 批次預測
# ---------------------------------------------------------------------------

class TestPredictMany:

    def test_predict_many_basic(self, fitted_predictor):
        p, X, _ = fitted_predictor
        result = p.predict_many(X[:100], batch_size=32)
        assert result["n_samples"] == 100
        assert len(result["prediction"]) == 100

    def test_predict_many_single_batch(self, fitted_predictor):
        p, X, _ = fitted_predictor
        result = p.predict_many(X[:10], batch_size=1024)
        assert len(result["prediction"]) == 10

    def test_predict_many_before_fit_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(RuntimeError, match="尚未訓練"):
            p.predict_many([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# 輸入格式支援
# ---------------------------------------------------------------------------

class TestInputFormats:

    def test_list_of_lists(self, fitted_predictor):
        p, _, _ = fitted_predictor
        result = p.predict([[1.0, 2.0, 3.0, 4.0]])
        assert len(result["prediction"]) == 1

    def test_dict_input(self, rng):
        rows_train = [{"a": float(i), "b": float(i * 2)} for i in range(50)]
        y_train = [float(i) for i in range(50)]
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(rows_train, y_train, model="linear")

        rows_test = [{"a": 10.0, "b": 20.0}, {"a": 20.0, "b": 40.0}]
        result = p.predict(rows_test)
        assert len(result["prediction"]) == 2

    def test_1d_array_reshaped(self, fitted_predictor):
        p, _, _ = fitted_predictor
        result = p.predict(np.array([1.0, 2.0, 3.0, 4.0]))
        assert len(result["prediction"]) == 1

    def test_pandas_dataframe(self, sample_data):
        pd = pytest.importorskip("pandas")
        X, y = sample_data
        df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(df, y, model="linear")
        result = p.predict(df.iloc[:5])
        assert len(result["prediction"]) == 5


# ---------------------------------------------------------------------------
# 資料驗證
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_nan_replaced(self, rng):
        X = rng.normal(size=(100, 3))
        y = rng.normal(size=(100,))
        X[0, 0] = np.nan
        X[5, 2] = np.inf
        p = UnifiedPredictor(auto_onnx=False)
        metrics = p.fit(X, y)
        assert metrics["train_rmse"] >= 0

    def test_predict_with_nan(self, fitted_predictor):
        p, _, _ = fitted_predictor
        X_test = np.array([[1.0, np.nan, 3.0, 4.0]])
        result = p.predict(X_test)
        assert len(result["prediction"]) == 1


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_normalize_improves_or_maintains(self, sample_data):
        X, y = sample_data
        p_raw = UnifiedPredictor(auto_onnx=False, normalize=False)
        m_raw = p_raw.fit(X, y, model="linear")

        p_norm = UnifiedPredictor(auto_onnx=False, normalize=True)
        m_norm = p_norm.fit(X, y, model="linear")

        assert m_norm["train_r2"] > 0.8
        assert m_raw["train_r2"] > 0.8

    def test_scaler_applied_on_predict(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False, normalize=True)
        p.fit(X[:150], y[:150], model="linear")
        result = p.predict(X[150:160])
        assert len(result["prediction"]) == 10


# ---------------------------------------------------------------------------
# 評估
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_returns_metrics(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(X[:150], y[:150], model="linear")
        ev = p.evaluate(X[150:], y[150:])
        assert "rmse" in ev and "mae" in ev and "r2" in ev
        assert ev["rmse"] >= 0
        assert ev["mae"] >= 0

    def test_evaluate_before_fit_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(RuntimeError, match="尚未訓練"):
            p.evaluate([[1.0]], [1.0])


# ---------------------------------------------------------------------------
# 模型儲存/載入
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False, normalize=True)
        p.fit(X[:150], y[:150], model="linear")
        pred_before = p.predict(X[150:155])

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.pkl")
            p.save_model(path)
            assert os.path.exists(path)

            p2 = UnifiedPredictor(auto_onnx=False)
            p2.load_model(path)
            pred_after = p2.predict(X[150:155])

        np.testing.assert_allclose(
            pred_before["prediction"],
            pred_after["prediction"],
            rtol=1e-5,
        )

    def test_save_before_fit_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(RuntimeError, match="尚未訓練"):
            p.save_model("dummy.pkl")

    def test_load_missing_file_raises(self):
        p = UnifiedPredictor(auto_onnx=False)
        with pytest.raises(FileNotFoundError):
            p.load_model("nonexistent_path_12345.pkl")


# ---------------------------------------------------------------------------
# 快取
# ---------------------------------------------------------------------------

class TestCaching:

    def test_cache_returns_same_result(self, fitted_predictor):
        p, X, _ = fitted_predictor
        r1 = p.predict(X[:5])
        r2 = p.predict(X[:5])
        assert r1["prediction"] == r2["prediction"]
        assert r1["confidence"] == r2["confidence"]

    def test_cache_cleared_on_fit(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(X[:100], y[:100])
        _ = p.predict(X[:5])
        assert p._cache.size > 0
        p.fit(X[:100], y[:100])
        assert p._cache.size == 0


# ---------------------------------------------------------------------------
# LRU Cache 單元測試
# ---------------------------------------------------------------------------

class TestLRUCache:

    def test_get_set(self):
        c = _LRUCache(max_items=3)
        c.set("a", 1)
        c.set("b", 2)
        assert c.get("a") == 1
        assert c.get("b") == 2
        assert c.get("c") is None

    def test_eviction(self):
        c = _LRUCache(max_items=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_clear(self):
        c = _LRUCache(max_items=5)
        c.set("x", 10)
        c.clear()
        assert c.size == 0
        assert c.get("x") is None


# ---------------------------------------------------------------------------
# Token Bucket 單元測試
# ---------------------------------------------------------------------------

class TestTokenBucket:

    def test_allow_within_capacity(self):
        tb = _TokenBucket(capacity=5, refill_per_sec=0)
        for _ in range(5):
            assert tb.allow()
        assert not tb.allow()

    def test_refill(self):
        import time
        tb = _TokenBucket(capacity=2, refill_per_sec=100)
        assert tb.allow()
        assert tb.allow()
        assert not tb.allow()
        time.sleep(0.05)
        assert tb.allow()


# ---------------------------------------------------------------------------
# 置信度
# ---------------------------------------------------------------------------

class TestConfidence:

    def test_confidence_between_0_and_1(self, fitted_predictor):
        p, X, _ = fitted_predictor
        result = p.predict(X[:20])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_constant_prediction(self):
        p = UnifiedPredictor(auto_onnx=False)
        preds = np.full((10, 1), 5.0)
        conf = p._compute_confidence(preds)
        assert conf == pytest.approx(1.0)

    def test_confidence_high_variance(self):
        p = UnifiedPredictor(auto_onnx=False)
        preds = np.array([[0.0], [1000.0]])
        conf = p._compute_confidence(preds)
        assert conf < 0.01


# ---------------------------------------------------------------------------
# 自動 ONNX 包裝
# ---------------------------------------------------------------------------

class TestAutoONNX:

    def test_auto_onnx_linear(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=True)
        p.fit(X, y, model="linear")
        assert p.onnx_runner is not None
        assert "onnx" in p.model_name

        result = p.predict(X[:5])
        assert len(result["prediction"]) == 5

    def test_auto_onnx_disabled(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(X, y, model="linear")
        assert p.onnx_runner is None
        assert "onnx" not in p.model_name


# ---------------------------------------------------------------------------
# 模型資訊
# ---------------------------------------------------------------------------

class TestModelInfo:

    def test_info_before_fit(self):
        p = UnifiedPredictor(auto_onnx=False)
        info = p.get_model_info()
        assert info["is_fitted"] is False
        assert info["input_dim"] is None

    def test_info_after_fit(self, fitted_predictor):
        p, _, _ = fitted_predictor
        info = p.get_model_info()
        assert info["is_fitted"] is True
        assert info["input_dim"] == 4
        assert "fit_metrics" in info


# ---------------------------------------------------------------------------
# 領域預設地平線
# ---------------------------------------------------------------------------

class TestDefaultHorizons:

    def test_financial_horizons(self):
        p = UnifiedPredictor(auto_onnx=False)
        h = p._default_horizons("financial")
        assert h == [1, 5, 10, 20]

    def test_unknown_domain_fallback(self):
        p = UnifiedPredictor(auto_onnx=False)
        h = p._default_horizons("unknown_domain_xyz")
        assert isinstance(h, list)
        assert len(h) >= 1


# ---------------------------------------------------------------------------
# 回退模型（無 sklearn）
# ---------------------------------------------------------------------------

class TestFallbackModel:

    def test_fallback_with_unknown_model(self, sample_data):
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(X, y, model="unknown_model_type")
        assert isinstance(p.model, dict)
        assert "mean_y" in p.model
        result = p.predict(X[:3])
        assert len(result["prediction"]) == 3


# ---------------------------------------------------------------------------
# 端到端整合
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, rng):
        """訓練 -> 評估 -> 儲存 -> 載入 -> 預測 完整流程"""
        X = rng.normal(size=(300, 8))
        w = rng.normal(size=(8,))
        y = X @ w + rng.normal(scale=0.2, size=(300,))

        p = UnifiedPredictor(auto_onnx=False, normalize=True)
        train_metrics = p.fit(X[:200], y[:200], model="linear")
        assert train_metrics["train_r2"] > 0.9

        eval_metrics = p.evaluate(X[200:], y[200:])
        assert eval_metrics["r2"] > 0.8

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "e2e.pkl")
            p.save_model(path)

            p2 = UnifiedPredictor(auto_onnx=False)
            p2.load_model(path)
            assert p2.normalize is True

            r1 = p.predict(X[200:205])
            r2 = p2.predict(X[200:205])
            np.testing.assert_allclose(
                r1["prediction"], r2["prediction"], rtol=1e-5
            )

    def test_predict_many_consistency(self, sample_data):
        """predict 和 predict_many 對相同資料應回傳相同結果"""
        X, y = sample_data
        p = UnifiedPredictor(auto_onnx=False)
        p.fit(X[:150], y[:150])

        r_single = p.predict(X[150:160])
        r_batch = p.predict_many(X[150:160], batch_size=3)

        np.testing.assert_allclose(
            r_single["prediction"],
            r_batch["prediction"],
            rtol=1e-10,
        )
