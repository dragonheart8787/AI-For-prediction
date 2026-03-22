#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ CPU 多核優化模組
整合 Modin, Numba, IPEX, ONNX 等 CPU 優化技術
"""

import os
import numpy as np
import pandas as pd
import modin.pandas as mpd
from numba import njit, prange, jit
import torch
import torch.nn as nn
import onnxruntime as ort
import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 設置環境變數以優化 CPU 使用
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())

logger = logging.getLogger(__name__)

class CPUOptimizer:
    """CPU 優化器"""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_jobs)
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        
        # 設置 PyTorch 執行緒
        torch.set_num_threads(self.n_jobs)
        torch.set_num_interop_threads(self.n_jobs)
        
        logger.info(f"CPU 優化器初始化完成，使用 {self.n_jobs} 個核心")
    
    def optimize_pandas_operations(self, df: pd.DataFrame) -> mpd.DataFrame:
        """使用 Modin 優化 Pandas 操作"""
        try:
            # 轉換為 Modin DataFrame
            modin_df = mpd.DataFrame(df)
            
            # 示例：複雜的數據處理操作
            modin_df['feature_1'] = modin_df['feature_1'] * 0.3 + modin_df['feature_2'].shift(1).fillna(0)
            modin_df['feature_2'] = modin_df['feature_2'].rolling(window=5).mean()
            modin_df['feature_3'] = modin_df['feature_1'] + modin_df['feature_2']
            
            logger.info("✅ Modin 優化完成")
            return modin_df
            
        except Exception as e:
            logger.error(f"❌ Modin 優化失敗: {e}")
            return df
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def numba_vectorized_operation(arr: np.ndarray) -> np.ndarray:
        """使用 Numba 進行向量化操作"""
        out = np.empty_like(arr)
        for i in prange(1, arr.shape[0]):
            out[i] = 0.7 * arr[i] + 0.3 * arr[i-1]
        out[0] = arr[0]
        return out
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def numba_matrix_operations(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """使用 Numba 進行矩陣操作"""
        result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        for i in prange(matrix_a.shape[0]):
            for j in prange(matrix_b.shape[1]):
                for k in prange(matrix_a.shape[1]):
                    result[i, j] += matrix_a[i, k] * matrix_b[k, j]
        return result
    
    def optimize_sklearn_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """優化 sklearn 模型"""
        results = {}
        
        # Random Forest 優化
        print("🌲 優化 Random Forest...")
        start_time = time.time()
        rf = RandomForestRegressor(
            n_estimators=400,
            n_jobs=self.n_jobs,
            random_state=42
        )
        rf.fit(X, y)
        rf_time = time.time() - start_time
        results['random_forest'] = {
            'model': rf,
            'training_time': rf_time,
            'score': rf.score(X, y)
        }
        
        # Ridge 回歸優化
        print("📈 優化 Ridge 回歸...")
        start_time = time.time()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        ridge_time = time.time() - start_time
        results['ridge'] = {
            'model': ridge,
            'training_time': ridge_time,
            'score': ridge.score(X, y)
        }
        
        logger.info(f"✅ sklearn 模型優化完成，Random Forest: {rf_time:.2f}s, Ridge: {ridge_time:.2f}s")
        return results
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """優化 XGBoost"""
        print("🚀 優化 XGBoost...")
        start_time = time.time()
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",  # CPU 友善
            nthread=self.n_jobs,
            random_state=42
        )
        
        xgb_model.fit(X, y)
        xgb_time = time.time() - start_time
        
        results = {
            'model': xgb_model,
            'training_time': xgb_time,
            'score': xgb_model.score(X, y)
        }
        
        logger.info(f"✅ XGBoost 優化完成，訓練時間: {xgb_time:.2f}s")
        return results
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """優化 LightGBM"""
        print("💡 優化 LightGBM...")
        start_time = time.time()
        
        lgb_model = lgb.LGBMRegressor(
            num_leaves=255,
            n_estimators=2000,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=42
        )
        
        lgb_model.fit(X, y)
        lgb_time = time.time() - start_time
        
        results = {
            'model': lgb_model,
            'training_time': lgb_time,
            'score': lgb_model.score(X, y)
        }
        
        logger.info(f"✅ LightGBM 優化完成，訓練時間: {lgb_time:.2f}s")
        return results
    
    def optimize_pytorch_cpu(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """優化 PyTorch CPU 訓練"""
        print("🔥 優化 PyTorch CPU 訓練...")
        
        # 嘗試使用 Intel IPEX
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            logger.info("✅ 使用 Intel IPEX 優化")
        except ImportError:
            logger.warning("Intel IPEX 未安裝，使用標準 PyTorch")
        
        # 設置模型為評估模式
        model.eval()
        
        # 轉換數據為張量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # 測試推理性能
        start_time = time.time()
        with torch.no_grad():
            predictions = model(X_tensor)
        inference_time = time.time() - start_time
        
        # 計算損失
        criterion = nn.MSELoss()
        loss = criterion(predictions.squeeze(), y_tensor).item()
        
        results = {
            'model': model,
            'inference_time': inference_time,
            'loss': loss,
            'predictions': predictions.numpy()
        }
        
        logger.info(f"✅ PyTorch CPU 優化完成，推理時間: {inference_time:.4f}s")
        return results
    
    def convert_to_onnx(self, model: Any, X_sample: np.ndarray, 
                       model_name: str = "model") -> str:
        """將模型轉換為 ONNX 格式"""
        print(f"🔄 轉換 {model_name} 為 ONNX...")
        
        try:
            if hasattr(model, 'predict'):
                # sklearn 模型
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                
                onnx_path = f"{model_name}.onnx"
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
            elif isinstance(model, nn.Module):
                # PyTorch 模型
                dummy_input = torch.tensor(X_sample, dtype=torch.float32)
                onnx_path = f"{model_name}.onnx"
                
                torch.onnx.export(
                    model, dummy_input, onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    opset_version=17
                )
            
            else:
                raise ValueError("不支援的模型類型")
            
            logger.info(f"✅ ONNX 轉換完成: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"❌ ONNX 轉換失敗: {e}")
            return None
    
    def optimize_onnx_inference(self, onnx_path: str, X: np.ndarray) -> Dict[str, Any]:
        """優化 ONNX 推理"""
        print(f"⚡ 優化 ONNX 推理: {onnx_path}")
        
        try:
            # 設置 ONNX Runtime 會話選項
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.n_jobs
            sess_options.inter_op_num_threads = self.n_jobs
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 創建推理會話
            session = ort.InferenceSession(onnx_path, sess_options=sess_options, 
                                         providers=["CPUExecutionProvider"])
            
            # 準備輸入數據
            input_name = session.get_inputs()[0].name
            input_data = X.astype(np.float32)
            
            # 測試推理性能
            start_time = time.time()
            predictions = session.run(None, {input_name: input_data})[0]
            inference_time = time.time() - start_time
            
            results = {
                'session': session,
                'inference_time': inference_time,
                'predictions': predictions,
                'throughput': len(X) / inference_time
            }
            
            logger.info(f"✅ ONNX 推理優化完成，推理時間: {inference_time:.4f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ ONNX 推理優化失敗: {e}")
            return {}
    
    def parallel_data_processing(self, data_list: List[np.ndarray], 
                               processing_func: callable) -> List[Any]:
        """並行數據處理"""
        print(f"🔄 並行處理 {len(data_list)} 個數據批次...")
        
        start_time = time.time()
        
        # 使用進程池進行並行處理
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(processing_func, data_list))
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ 並行處理完成，處理時間: {processing_time:.2f}s")
        return results
    
    def batch_inference(self, model: Any, X: np.ndarray, 
                       batch_size: int = 32) -> np.ndarray:
        """批量推理優化"""
        print(f"📦 批量推理優化，批次大小: {batch_size}")
        
        predictions = []
        start_time = time.time()
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            if hasattr(model, 'predict'):
                batch_pred = model.predict(batch)
            elif isinstance(model, nn.Module):
                with torch.no_grad():
                    batch_tensor = torch.tensor(batch, dtype=torch.float32)
                    batch_pred = model(batch_tensor).numpy()
            else:
                raise ValueError("不支援的模型類型")
            
            predictions.append(batch_pred)
        
        predictions = np.concatenate(predictions, axis=0)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ 批量推理完成，總時間: {inference_time:.4f}s")
        return predictions
    
    def get_cpu_utilization_info(self) -> Dict[str, Any]:
        """獲取 CPU 使用情況信息"""
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_info.percent,
            'memory_available_gb': memory_info.available / (1024**3),
            'n_jobs_configured': self.n_jobs
        }
    
    def benchmark_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """性能基準測試"""
        print("🏁 開始性能基準測試...")
        
        benchmark_results = {}
        
        # 測試 Numba 向量化操作
        print("測試 Numba 向量化操作...")
        start_time = time.time()
        numba_result = self.numba_vectorized_operation(X[:, 0])
        numba_time = time.time() - start_time
        benchmark_results['numba_vectorized'] = {
            'time': numba_time,
            'throughput': len(X) / numba_time
        }
        
        # 測試 sklearn 模型
        print("測試 sklearn 模型...")
        sklearn_results = self.optimize_sklearn_models(X, y)
        for model_name, result in sklearn_results.items():
            benchmark_results[f'sklearn_{model_name}'] = {
                'training_time': result['training_time'],
                'score': result['score']
            }
        
        # 測試 XGBoost
        print("測試 XGBoost...")
        xgb_results = self.optimize_xgboost(X, y)
        benchmark_results['xgboost'] = {
            'training_time': xgb_results['training_time'],
            'score': xgb_results['score']
        }
        
        # 測試 LightGBM
        print("測試 LightGBM...")
        lgb_results = self.optimize_lightgbm(X, y)
        benchmark_results['lightgbm'] = {
            'training_time': lgb_results['training_time'],
            'score': lgb_results['score']
        }
        
        # 測試批量推理
        print("測試批量推理...")
        batch_times = []
        for batch_size in [16, 32, 64, 128]:
            start_time = time.time()
            self.batch_inference(sklearn_results['random_forest']['model'], X, batch_size)
            batch_time = time.time() - start_time
            batch_times.append((batch_size, batch_time))
        
        benchmark_results['batch_inference'] = {
            'batch_sizes': [t[0] for t in batch_times],
            'times': [t[1] for t in batch_times]
        }
        
        logger.info("✅ 性能基準測試完成")
        return benchmark_results

# 使用示例
if __name__ == "__main__":
    # 創建示例數據
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
    
    # 創建 CPU 優化器
    optimizer = CPUOptimizer(n_jobs=-1)
    
    print("🖥️ CPU 多核優化演示")
    print("=" * 50)
    
    # 獲取 CPU 信息
    cpu_info = optimizer.get_cpu_utilization_info()
    print(f"CPU 核心數: {cpu_info['cpu_count']}")
    print(f"CPU 使用率: {cpu_info['cpu_percent']}%")
    print(f"記憶體使用率: {cpu_info['memory_percent']}%")
    print(f"配置的並行數: {cpu_info['n_jobs_configured']}")
    
    # 測試 Numba 向量化操作
    print("\n🔢 測試 Numba 向量化操作...")
    numba_result = optimizer.numba_vectorized_operation(X[:, 0])
    print(f"✅ Numba 向量化操作完成，結果形狀: {numba_result.shape}")
    
    # 測試矩陣操作
    print("\n🔢 測試 Numba 矩陣操作...")
    matrix_a = np.random.randn(100, 100)
    matrix_b = np.random.randn(100, 100)
    matrix_result = optimizer.numba_matrix_operations(matrix_a, matrix_b)
    print(f"✅ 矩陣操作完成，結果形狀: {matrix_result.shape}")
    
    # 優化 sklearn 模型
    print("\n🌲 優化 sklearn 模型...")
    sklearn_results = optimizer.optimize_sklearn_models(X, y)
    for model_name, result in sklearn_results.items():
        print(f"{model_name}: 訓練時間 {result['training_time']:.2f}s, 分數 {result['score']:.4f}")
    
    # 優化 XGBoost
    print("\n🚀 優化 XGBoost...")
    xgb_results = optimizer.optimize_xgboost(X, y)
    print(f"XGBoost: 訓練時間 {xgb_results['training_time']:.2f}s, 分數 {xgb_results['score']:.4f}")
    
    # 優化 LightGBM
    print("\n💡 優化 LightGBM...")
    lgb_results = optimizer.optimize_lightgbm(X, y)
    print(f"LightGBM: 訓練時間 {lgb_results['training_time']:.2f}s, 分數 {lgb_results['score']:.4f}")
    
    # 轉換為 ONNX
    print("\n🔄 轉換模型為 ONNX...")
    onnx_path = optimizer.convert_to_onnx(sklearn_results['random_forest']['model'], X[:100])
    if onnx_path:
        print(f"✅ ONNX 模型已保存: {onnx_path}")
        
        # 優化 ONNX 推理
        print("\n⚡ 優化 ONNX 推理...")
        onnx_results = optimizer.optimize_onnx_inference(onnx_path, X[:1000])
        if onnx_results:
            print(f"ONNX 推理時間: {onnx_results['inference_time']:.4f}s")
            print(f"吞吐量: {onnx_results['throughput']:.0f} 樣本/秒")
    
    # 測試批量推理
    print("\n📦 測試批量推理...")
    batch_predictions = optimizer.batch_inference(sklearn_results['random_forest']['model'], X[:1000], batch_size=64)
    print(f"✅ 批量推理完成，預測形狀: {batch_predictions.shape}")
    
    # 性能基準測試
    print("\n🏁 性能基準測試...")
    benchmark_results = optimizer.benchmark_performance(X, y)
    
    print("\n📊 基準測試結果:")
    for test_name, result in benchmark_results.items():
        if 'time' in result:
            print(f"{test_name}: {result['time']:.4f}s")
        elif 'training_time' in result:
            print(f"{test_name}: 訓練時間 {result['training_time']:.2f}s, 分數 {result['score']:.4f}")
    
    print("\n🎉 CPU 多核優化演示完成！")
