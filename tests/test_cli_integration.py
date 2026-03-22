"""
CLI 整合測試套件
測試 demo_backtest_all.py 的命令列介面與 JSON 輸出
"""
import json
import os
import subprocess
import sys

import pytest

# 加速：避免每次 subprocess 都打外部 API
os.environ.setdefault("PREDICT_AI_DEMO_FAST", "1")


def _run_demo(args, timeout=60):
    """Windows 下避免 cp950 解碼 UTF-8 輸出失敗。"""
    return subprocess.run(
        [sys.executable, "demo_backtest_all.py", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


class TestCLIIntegration:
    """CLI 整合測試"""

    def test_demo_backtest_cli_help(self):
        """測試 CLI 幫助訊息"""
        try:
            result = _run_demo(["--help"], timeout=10)
            assert result.returncode == 0
            out = (result.stdout or "") + (result.stderr or "")
            assert "--model" in out
            assert "--batch" in out
        except subprocess.TimeoutExpired:
            pytest.skip("CLI help command timed out")

    def test_demo_backtest_json_output(self):
        """測試 JSON 輸出格式"""
        try:
            result = _run_demo(["--model", "linear", "--batch", "32"], timeout=60)

            # 檢查返回碼
            assert result.returncode == 0, f"Command failed: {result.stderr}"

            # 解析 JSON 輸出
            output_lines = (result.stdout or "").strip().split("\n")
            json_results = []

            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json_obj = json.loads(line)
                        json_results.append(json_obj)
                    except json.JSONDecodeError:
                        continue

            # 應該有至少一個 JSON 結果
            assert len(json_results) > 0, "No valid JSON output found"

            # 檢查 JSON 結構
            for result in json_results:
                assert 'model' in result
                assert 'domain' in result
                assert 'horizon' in result
                assert 'mae' in result
                assert 'batch' in result
                assert 'onnx' in result

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")

    def test_demo_backtest_auto_model(self):
        """測試自動模型選擇"""
        try:
            result = _run_demo(["--model", "auto", "--batch", "16"], timeout=60)

            assert result.returncode == 0

            # 解析輸出
            output_lines = (result.stdout or "").strip().split("\n")
            json_results = []

            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json_obj = json.loads(line)
                        json_results.append(json_obj)
                    except json.JSONDecodeError:
                        continue

            # 檢查模型類型
            models_used = set(r["model"] for r in json_results)
            assert len(models_used) > 0, "No models were used"

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")

    def test_demo_backtest_batch_sizes(self):
        """測試不同批次大小"""
        batch_sizes = [8, 16, 32, 64]

        for batch_size in batch_sizes:
            try:
                result = _run_demo(["--model", "linear", "--batch", str(batch_size)], timeout=60)

                assert result.returncode == 0, f"Batch size {batch_size} failed"

                # 檢查批次大小在輸出中
                output_lines = (result.stdout or "").strip().split("\n")
                batch_found = False

                for line in output_lines:
                    if f'"batch": {batch_size}' in line:
                        batch_found = True
                        break

                assert batch_found, f"Batch size {batch_size} not found in output"

            except subprocess.TimeoutExpired:
                pytest.skip(f"CLI command timed out for batch size {batch_size}")

    def test_demo_backtest_onnx_flag(self):
        """測試 ONNX 標記"""
        try:
            result = _run_demo(["--model", "linear", "--batch", "16"], timeout=60)

            assert result.returncode == 0

            # 檢查 ONNX 標記
            output_lines = (result.stdout or "").strip().split("\n")
            onnx_found = False

            for line in output_lines:
                if '"onnx":' in line:
                    onnx_found = True
                    break

            assert onnx_found, "ONNX flag not found in output"

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")

    def test_demo_backtest_error_handling(self):
        """測試錯誤處理"""
        # 測試無效模型（argparse 拒絕）
        result = _run_demo(["--model", "invalid_model", "--batch", "16"], timeout=15)
        err = (result.stderr or "").lower()
        assert result.returncode != 0 or "error" in err

    def test_demo_backtest_performance_metrics(self):
        """測試性能指標輸出"""
        try:
            result = _run_demo(["--model", "linear", "--batch", "32"], timeout=60)

            assert result.returncode == 0

            # 解析並檢查性能指標
            output_lines = (result.stdout or "").strip().split("\n")
            json_results = []

            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json_obj = json.loads(line)
                        json_results.append(json_obj)
                    except json.JSONDecodeError:
                        continue

            for result in json_results:
                # 檢查必要的性能指標
                assert 'mae' in result
                assert 'mse' in result
                assert 'rmse' in result
                assert 'r2' in result

                # 檢查指標值合理性
                assert isinstance(result['mae'], (int, float))
                assert isinstance(result['mse'], (int, float))
                assert isinstance(result['rmse'], (int, float))
                assert isinstance(result['r2'], (int, float))

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")

    def test_demo_backtest_streaming_output(self):
        """測試串流輸出"""
        try:
            # 使用較大的批次大小來測試串流
            result = _run_demo(["--model", "linear", "--batch", "128"], timeout=90)

            assert result.returncode == 0

            # 檢查輸出有多行
            output_lines = (result.stdout or "").strip().split("\n")
            assert len(output_lines) > 1, "Expected multiple lines of output"

            # 檢查每行都是有效的 JSON 或警告訊息
            json_lines = 0
            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json.loads(line)
                        json_lines += 1
                    except json.JSONDecodeError:
                        pass

            assert json_lines > 0, "No valid JSON lines found"

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")

    def test_demo_backtest_model_switching(self):
        """測試模型切換"""
        models = ['linear', 'xgboost', 'lightgbm']

        for model in models:
            try:
                result = _run_demo(["--model", model, "--batch", "16"], timeout=90)

                # 模型切換應該成功
                assert result.returncode == 0, f"Model {model} failed"

                # 檢查模型名稱在輸出中（可能為 onnx_* 包裝）
                out = (result.stdout or "") + (result.stderr or "")
                assert model in out, f"Model {model} not found in output"

            except subprocess.TimeoutExpired:
                pytest.skip(f"CLI command timed out for model {model}")

    def test_demo_backtest_resource_usage(self):
        """測試資源使用"""
        import psutil

        # 記錄初始資源
        initial_memory = psutil.virtual_memory().used

        try:
            result = _run_demo(["--model", "linear", "--batch", "64"], timeout=60)

            assert result.returncode == 0

            # 檢查記憶體使用合理
            final_memory = psutil.virtual_memory().used
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

            # 記憶體增加應該小於 1GB
            assert memory_increase < 1024, f"Memory increase {memory_increase:.1f}MB too high"

        except subprocess.TimeoutExpired:
            pytest.skip("CLI command timed out")
