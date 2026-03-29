"""UnifiedPredictor HTTP 服務（WSGI）邏輯測試。"""
import json
from pathlib import Path

import numpy as np
import pytest

from unified_predict import UnifiedPredictor


@pytest.fixture
def model_pkl(tmp_path: Path) -> str:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False, normalize=False)
    p.fit(X, y, model="linear")
    mp = tmp_path / "t.pkl"
    p.save_model(str(mp))
    return str(mp)


def test_http_root_landing(model_pkl: str):
    from model_serving.unified_http_service import UnifiedPredictHTTPApp

    app = UnifiedPredictHTTPApp(model_pkl)
    st, headers, b = app.handle_request("GET", "/", b"")
    assert st == 200
    assert any(h[0].lower() == "content-type" and "text/html" in h[1] for h in headers)
    text = b.decode("utf-8")
    assert "v1/predict" in text
    assert "/v1/models" in text
    assert "操作介面" in text


def test_http_health_and_predict(model_pkl: str):
    from model_serving.unified_http_service import UnifiedPredictHTTPApp

    app = UnifiedPredictHTTPApp(model_pkl)
    st, _h, b = app.handle_request("GET", "/health", b"")
    assert st == 200
    assert json.loads(b.decode())["status"] == "ok"

    st2, _h2, b2 = app.handle_request(
        "POST",
        "/v1/predict",
        json.dumps({"X": [[0.1, 0.2, 0.3]], "domain": "financial"}).encode("utf-8"),
    )
    assert st2 == 200
    out = json.loads(b2.decode())
    assert "prediction" in out
    assert len(out["prediction"]) == 1


def test_http_404(model_pkl: str):
    from model_serving.unified_http_service import UnifiedPredictHTTPApp

    app = UnifiedPredictHTTPApp(model_pkl)
    st, _, b = app.handle_request("GET", "/nope", b"")
    assert st == 404


def test_http_v1_models_and_load(tmp_path: Path):
    from model_serving.unified_http_service import UnifiedPredictHTTPApp

    models = tmp_path / "models"
    models.mkdir()
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False, normalize=False)
    p.fit(X, y, model="linear")
    p._feature_names = ["a", "b"]
    p.save_model(str(models / "task_demo_task.pkl"))

    app = UnifiedPredictHTTPApp(model_path=None, models_root=str(models), auto_load=False)
    st, _, b = app.handle_request("GET", "/v1/models", b"")
    assert st == 200
    j = json.loads(b.decode())
    assert any(t.get("task_id") == "demo_task" for t in j.get("tasks", []))

    st2, _, b2 = app.handle_request(
        "POST",
        "/v1/load_model",
        json.dumps({"task_id": "demo_task"}).encode("utf-8"),
    )
    assert st2 == 200
    st3, _, b3 = app.handle_request("GET", "/v1/model/info", b"")
    assert st3 == 200
    info = json.loads(b3.decode())
    assert info.get("feature_names") == ["a", "b"]
