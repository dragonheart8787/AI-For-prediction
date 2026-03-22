"""
效能測試套件
測試系統整體性能與資源使用
"""
import pytest
import numpy as np
import time
import os
from unified_predict import UnifiedPredictor


class TestPerformance:
    """效能測試"""

    def setup_method(self):
        """每個測試前的設置"""
        np.random.seed(42)
        n_samples = 1000
        self.X = np.random.randn(n_samples, 4).astype(float)
        self.y = (2 * np.random.randn(n_samples) + 1).astype(float)

    def test_training_time_under_2s(self):
        """測試訓練時間小於 2 秒（smoke test）"""
        predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)

        start_time = time.time()
        predictor.fit(self.X, self.y, model="linear")
        training_time = time.time() - start_time

        assert training_time < 2.0, f"訓練耗時 {training_time:.2f}s，期望 < 2s"

    def test_prediction_throughput(self):
        """測試預測吞吐量"""
        predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        predictor.fit(self.X, self.y, model="linear")

        # 單次預測
        start_time = time.time()
        for _ in range(100):
            predictor.predict(self.X[:10])
        single_time = time.time() - start_time

        # 批次預測
        start_time = time.time()
        predictor.predict_many(self.X[:1000])
        batch_time = time.time() - start_time

        # 批次應較快（per sample）
        samples_single = 100 * 10
        samples_batch = 1000
        throughput_single = samples_single / single_time
        throughput_batch = samples_batch / batch_time
        assert throughput_batch >= throughput_single * 0.5, "批次預測吞吐量應接近或高於重複單次"

    def test_memory_usage(self):
        """測試記憶體使用"""
        try:
            import psutil
        except ImportError:
            pytest.skip("需安裝 psutil")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        predictors = []
        for _ in range(5):
            p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
            p.fit(self.X, self.y, model="linear")
            predictors.append(p)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 500, f"記憶體增加 {memory_increase:.1f}MB 過高"

    def test_concurrent_prediction(self):
        """測試並發預測"""
        import threading
        import queue

        predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        predictor.fit(self.X, self.y, model="linear")

        results_queue = queue.Queue()

        def predict_worker(worker_id, num_predictions):
            for _ in range(num_predictions):
                out = predictor.predict(self.X[:5])
                results_queue.put((worker_id, len(out["prediction"])))

        threads = []
        num_workers = 4
        predictions_per_worker = 25

        start_time = time.time()
        for i in range(num_workers):
            t = threading.Thread(target=predict_worker, args=(i, predictions_per_worker))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start_time

        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == num_workers * predictions_per_worker
        assert total_time < 10.0, f"並發預測耗時 {total_time:.2f}s，期望 < 10s"

    def test_large_batch_performance(self):
        """測試大批量性能"""
        predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        predictor.fit(self.X, self.y, model="linear")

        large_batch = self.X[:1000]
        start_time = time.time()
        result = predictor.predict_many(large_batch)
        batch_time = time.time() - start_time

        assert "prediction" in result
        assert len(result["prediction"]) == 1000
        samples_per_second = 1000 / batch_time
        assert samples_per_second > 100, f"批次吞吐量 {samples_per_second:.1f} samples/s 過低"

    def test_model_switching_performance(self):
        """測試模型切換性能"""
        models = ["linear", "xgboost", "lightgbm"]
        for model in models:
            start_time = time.time()
            predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
            predictor.fit(self.X[:500], self.y[:500], model=model)
            fit_time = time.time() - start_time
            assert fit_time < 5.0, f"模型 {model} 訓練耗時 {fit_time:.2f}s，期望 < 5s"

    def test_predict_returns_dict(self):
        """驗證 predict 回傳格式"""
        predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        predictor.fit(self.X, self.y, model="linear")

        out = predictor.predict(self.X[:3])
        assert isinstance(out, dict)
        assert "prediction" in out
        assert "confidence" in out
        assert isinstance(out["prediction"], list)
        assert len(out["prediction"]) == 3

    def test_resource_cleanup(self):
        """測試資源清理"""
        try:
            import psutil
            import gc
        except ImportError:
            pytest.skip("需安裝 psutil")

        initial_handles = psutil.Process().num_handles()

        predictors = []
        for _ in range(10):
            p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
            p.fit(self.X[:100], self.y[:100], model="linear")
            predictors.append(p)

        del predictors
        gc.collect()

        final_handles = psutil.Process().num_handles()
        handle_increase = final_handles - initial_handles

        assert handle_increase < 100, f"句柄增加 {handle_increase} 過高"
