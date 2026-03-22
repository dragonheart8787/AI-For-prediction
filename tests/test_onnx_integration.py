"""
ONNX／線性包裝推論整合測試（對齊 UnifiedPredictor 現行 API）
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from unified_predict import UnifiedPredictor


class TestONNXIntegration:
    """ONNX 與手動線性 ONNX 包裝一致性"""

    def setup_method(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
                "target": 2 * np.random.randn(50) + 1,
            }
        )
        self.X = self.df[["feature1", "feature2", "feature3"]].values.astype(float)
        self.y = self.df["target"].values.astype(float)

    def test_linear_onnx_consistency(self):
        original = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        original.fit(self.X, self.y, model="linear")
        orig_pred = original.predict(self.X[:10], domain="financial")

        onnx_p = UnifiedPredictor(auto_onnx=True, rate_limit_enabled=False)
        onnx_p.fit(self.X, self.y, model="linear")
        onnx_pred = onnx_p.predict(self.X[:10], domain="financial")

        if "onnx_" in onnx_p.model_name.lower():
            a = np.asarray(orig_pred["prediction"])
            b = np.asarray(onnx_pred["prediction"])
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    def test_predict_returns_dict_with_prediction_and_confidence(self):
        p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        p.fit(self.X, self.y, model="linear")
        out = p.predict(self.X[:5], domain="financial")
        assert isinstance(out, dict)
        assert "prediction" in out
        assert "confidence" in out
        assert isinstance(out["confidence"], (int, float))
        assert len(out["prediction"]) == 5

    def test_predict_many_returns_batch_dict(self):
        p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        p.fit(self.X, self.y, model="linear")
        out = p.predict_many(self.X, domain="financial", batch_size=16)
        assert isinstance(out, dict)
        assert len(out["prediction"]) == len(self.X)

    def test_onnx_model_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            p = UnifiedPredictor(auto_onnx=True, rate_limit_enabled=False)
            p.fit(self.X, self.y, model="linear")
            p.save_model(path)
            q = UnifiedPredictor(auto_onnx=True, rate_limit_enabled=False)
            q.load_model(path)
            r1 = p.predict(self.X[:5], domain="financial")
            r2 = q.predict(self.X[:5], domain="financial")
            np.testing.assert_allclose(
                np.asarray(r1["prediction"]),
                np.asarray(r2["prediction"]),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_onnx_error_empty_input(self):
        p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        p.fit(self.X, self.y, model="linear")
        with pytest.raises(ValueError):
            p.predict(np.zeros((0, 3)), domain="financial")
