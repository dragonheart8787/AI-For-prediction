#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真實AI預測系統啟動腳本
簡化版本，專注於核心功能
"""

import os
import sys
import json
import time
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_ai_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleRealModelDownloader:
    """簡化真實模型下載器"""
    
    def __init__(self, models_dir: str = "./real_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 真實模型配置
        self.real_models = {
            'timesfm': {
                'name': 'TimesFM',
                'description': 'Google零訓練時間序列預測模型',
                'type': 'zero_shot',
                'source': 'huggingface',
                'model_id': 'google/timesfm-base'
            },
            'chronos': {
                'name': 'Chronos-Bolt',
                'description': 'Amazon基於T5的時間序列模型',
                'type': 'zero_shot',
                'source': 'huggingface',
                'model_id': 'amazon/chronos-t5-small'
            },
            'patchtst': {
                'name': 'PatchTST',
                'description': 'IBM基於Patch的時間序列Transformer',
                'type': 'deep_learning',
                'source': 'huggingface',
                'model_id': 'ibm/patchtst'
            },
            'itransformer': {
                'name': 'iTransformer',
                'description': '反轉Transformer時間序列模型',
                'type': 'deep_learning',
                'source': 'huggingface',
                'model_id': 'thuml/iTransformer'
            },
            'informer': {
                'name': 'Informer',
                'description': '高效Transformer時間序列模型',
                'type': 'deep_learning',
                'source': 'huggingface',
                'model_id': 'thuml/Informer'
            },
            'dlinear': {
                'name': 'DLinear',
                'description': '分解線性時間序列模型',
                'type': 'deep_learning',
                'source': 'huggingface',
                'model_id': 'thuml/DLinear'
            }
        }
    
    async def download_model(self, model_key: str) -> bool:
        """下載單個模型"""
        try:
            model_info = self.real_models[model_key]
            model_dir = self.models_dir / model_key
            model_dir.mkdir(exist_ok=True)
            
            logger.info(f"開始下載模型: {model_info['name']}")
            
            # 嘗試下載配置文件
            config_url = f"https://huggingface.co/{model_info['model_id']}/resolve/main/config.json"
            response = requests.get(config_url, timeout=30)
            
            if response.status_code == 200:
                # 保存配置文件
                config_file = model_dir / 'config.json'
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # 創建狀態文件
                status_data = {
                    'model_key': model_key,
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'type': model_info['type'],
                    'source': model_info['source'],
                    'model_id': model_info['model_id'],
                    'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'config_downloaded',
                    'note': '配置文件已下載，完整模型需要使用huggingface_hub'
                }
                
                status_file = model_dir / 'status.json'
                with open(status_file, 'w', encoding='utf-8') as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"成功下載模型配置: {model_key}")
                return True
                
            else:
                logger.warning(f"無法下載模型配置: {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"下載模型失敗 {model_key}: {e}")
            return False
    
    async def download_all_models(self) -> Dict[str, Any]:
        """下載所有模型"""
        logger.info("開始下載所有真實模型...")
        
        results = {}
        for model_key in self.real_models.keys():
            logger.info(f"下載模型: {model_key}")
            result = await self.download_model(model_key)
            results[model_key] = result
            await asyncio.sleep(1)  # 避免過於頻繁的請求
        
        # 統計結果
        successful_downloads = sum(1 for r in results.values() if r)
        total_models = len(self.real_models)
        
        logger.info(f"下載完成: {successful_downloads}/{total_models} 個模型成功")
        
        return {
            'total_models': total_models,
            'successful_downloads': successful_downloads,
            'failed_downloads': total_models - successful_downloads,
            'results': results
        }
    
    def get_available_models(self) -> List[str]:
        """獲取可用模型列表"""
        available = []
        for model_key in self.real_models.keys():
            model_dir = self.models_dir / model_key
            status_file = model_dir / 'status.json'
            if status_file.exists():
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status_data = json.load(f)
                    if status_data.get('status') == 'config_downloaded':
                        available.append(model_key)
                except Exception as e:
                    logger.warning(f"讀取模型狀態失敗 {model_key}: {e}")
        
        return available


class SimpleDataCollector:
    """簡化數據收集器"""
    
    def __init__(self, data_dir: str = "./collected_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據源配置
        self.data_sources = {
            'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CADUSD=X'],
            'commodities': ['GC=F', 'CL=F', 'SI=F', 'NG=F'],
            'indices': ['^GSPC', '^DJI', '^IXIC', '^FTSE']
        }
    
    def collect_sample_data(self) -> Dict[str, pd.DataFrame]:
        """收集樣本數據（模擬）"""
        logger.info("開始收集樣本數據...")
        
        all_data = {}
        
        # 創建模擬數據
        for data_type, symbols in self.data_sources.items():
            data_type_dir = self.data_dir / data_type
            data_type_dir.mkdir(exist_ok=True)
            
            for symbol in symbols:
                try:
                    # 創建模擬時間序列數據
                    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
                    np.random.seed(hash(symbol) % 2**32)  # 為每個符號設置不同的隨機種子
                    
                    # 生成模擬價格數據
                    base_price = 100 + hash(symbol) % 1000
                    trend = np.linspace(0, 50, len(dates))
                    noise = np.random.normal(0, 5, len(dates))
                    prices = base_price + trend + noise
                    
                    # 創建DataFrame
                    data = pd.DataFrame({
                        'Date': dates,
                        'Open': prices + np.random.normal(0, 2, len(dates)),
                        'High': prices + np.random.normal(5, 3, len(dates)),
                        'Low': prices - np.random.normal(5, 3, len(dates)),
                        'Close': prices,
                        'Volume': np.random.randint(1000000, 10000000, len(dates))
                    })
                    
                    # 添加技術指標
                    data['SMA_20'] = data['Close'].rolling(window=20).mean()
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['RSI'] = self._calculate_rsi(data['Close'])
                    data['MACD'] = self._calculate_macd(data['Close'])
                    data['Volatility'] = data['Close'].rolling(window=20).std()
                    
                    # 保存數據
                    data_file = data_type_dir / f"{symbol.replace('=', '_').replace('-', '_')}.csv"
                    data.to_csv(data_file, index=False)
                    all_data[symbol] = data
                    
                    logger.info(f"創建模擬數據: {symbol} -> {data_file}")
                    
                except Exception as e:
                    logger.error(f"創建數據失敗 {symbol}: {e}")
        
        logger.info(f"數據收集完成，共創建 {len(all_data)} 個數據集")
        return all_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """計算MACD指標"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd


class SimpleModelTrainer:
    """簡化模型訓練器"""
    
    def __init__(self, models_dir: str = "./real_models"):
        self.models_dir = Path(models_dir)
    
    def train_simple_model(self, model_key: str, data: pd.DataFrame) -> Dict[str, Any]:
        """訓練簡單預測模型"""
        try:
            logger.info(f"開始訓練簡單模型: {model_key}")
            
            # 準備數據
            series = data['Close'].dropna()
            
            if len(series) < 100:
                return {'error': f'數據點不足: {len(series)} < 100'}
            
            # 簡單的移動平均預測
            ma_20 = series.rolling(window=20).mean().iloc[-1]
            ma_50 = series.rolling(window=50).mean().iloc[-1]
            
            # 趨勢預測
            recent_trend = (series.iloc[-1] - series.iloc[-20]) / 20
            
            # 生成30天預測
            forecast = []
            current_price = series.iloc[-1]
            
            for i in range(30):
                # 結合移動平均和趨勢
                predicted = (ma_20 * 0.6 + ma_50 * 0.4) + (recent_trend * (i + 1))
                forecast.append(max(predicted, 0))  # 確保價格非負
            
            result = {
                'model_type': f'Simple_{model_key}',
                'forecast': forecast,
                'training_data_points': len(series),
                'ma_20': float(ma_20),
                'ma_50': float(ma_50),
                'trend': float(recent_trend),
                'last_price': float(current_price)
            }
            
            logger.info(f"簡單模型訓練成功: {model_key}")
            return result
            
        except Exception as e:
            logger.error(f"訓練簡單模型失敗 {model_key}: {e}")
            return {'error': str(e)}


class SimpleModelFusion:
    """簡化模型融合系統"""
    
    def __init__(self):
        pass
    
    def create_fusion_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """創建融合模型"""
        try:
            logger.info("開始創建融合模型...")
            
            # 收集所有預測結果
            forecasts = []
            model_names = []
            
            for model_name, result in model_results.items():
                if 'error' not in result and 'forecast' in result:
                    forecasts.append(result['forecast'])
                    model_names.append(model_name)
            
            if not forecasts:
                return {'error': '沒有可用的預測結果'}
            
            # 計算集成預測
            forecasts_array = np.array(forecasts)
            ensemble_forecast = np.mean(forecasts_array, axis=0)
            
            # 計算預測置信區間
            std_forecast = np.std(forecasts_array, axis=0)
            confidence_interval = {
                'lower': ensemble_forecast - 1.96 * std_forecast,
                'upper': ensemble_forecast + 1.96 * std_forecast
            }
            
            result = {
                'fusion_type': 'ensemble',
                'base_models': model_names,
                'ensemble_forecast': ensemble_forecast.tolist(),
                'confidence_interval': {
                    'lower': confidence_interval['lower'].tolist(),
                    'upper': confidence_interval['upper'].tolist()
                },
                'model_weights': [1.0/len(model_names)] * len(model_names),
                'fusion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"融合模型創建成功，包含 {len(model_names)} 個基礎模型")
            return result
            
        except Exception as e:
            logger.error(f"創建融合模型失敗: {e}")
            return {'error': str(e)}


class SimpleRealAISystem:
    """簡化真實AI預測系統"""
    
    def __init__(self):
        self.downloader = SimpleRealModelDownloader()
        self.collector = SimpleDataCollector()
        self.trainer = SimpleModelTrainer()
        self.fusion = SimpleModelFusion()
    
    async def run_system(self):
        """運行系統"""
        logger.info("啟動簡化真實AI預測系統...")
        
        try:
            # 步驟1: 下載模型
            logger.info("步驟1: 下載真實模型...")
            download_results = await self.downloader.download_all_models()
            
            # 步驟2: 收集數據
            logger.info("步驟2: 收集預測數據...")
            all_data = self.collector.collect_sample_data()
            
            # 步驟3: 訓練模型
            logger.info("步驟3: 訓練預測模型...")
            training_results = {}
            
            available_models = self.downloader.get_available_models()
            for model_key in available_models:
                if all_data:
                    first_data = list(all_data.values())[0]
                    result = self.trainer.train_simple_model(model_key, first_data)
                    training_results[model_key] = result
            
            # 步驟4: 創建融合模型
            logger.info("步驟4: 創建融合模型...")
            fusion_result = self.fusion.create_fusion_model(training_results)
            
            # 步驟5: 生成最終預測
            logger.info("步驟5: 生成最終預測...")
            final_prediction = self._generate_final_prediction(fusion_result)
            
            # 步驟6: 保存結果
            logger.info("步驟6: 保存系統結果...")
            self._save_system_results(
                download_results, 
                all_data, 
                training_results, 
                fusion_result, 
                final_prediction
            )
            
            logger.info("簡化真實AI預測系統運行完成！")
            
        except Exception as e:
            logger.error(f"系統運行失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_final_prediction(self, fusion_result: Dict) -> Dict[str, Any]:
        """生成最終預測"""
        try:
            if 'error' in fusion_result:
                return fusion_result
            
            final_forecast = fusion_result['ensemble_forecast']
            forecast_array = np.array(final_forecast)
            
            result = {
                'final_forecast': final_forecast,
                'prediction_horizon': len(final_forecast),
                'statistics': {
                    'mean': float(np.mean(forecast_array)),
                    'std': float(np.std(forecast_array)),
                    'min': float(np.min(forecast_array)),
                    'max': float(np.max(forecast_array))
                },
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fusion_model_used': fusion_result['fusion_type'],
                'base_models_count': len(fusion_result.get('base_models', []))
            }
            
            logger.info(f"最終預測生成成功，預測點數: {len(final_forecast)}")
            return result
            
        except Exception as e:
            logger.error(f"生成最終預測失敗: {e}")
            return {'error': str(e)}
    
    def _save_system_results(self, download_results: Dict, all_data: Dict, 
                           training_results: Dict, fusion_result: Dict, 
                           final_prediction: Dict):
        """保存系統結果"""
        try:
            results_dir = Path("./system_results")
            results_dir.mkdir(exist_ok=True)
            
            # 保存所有結果
            results = {
                'download_results': download_results,
                'training_results': training_results,
                'fusion_results': fusion_result,
                'final_prediction': final_prediction,
                'data_summary': {
                    'total_datasets': len(all_data),
                    'dataset_names': list(all_data.keys()),
                    'data_points_per_dataset': {name: len(data) for name, data in all_data.items()},
                    'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'system_run_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(results_dir / 'complete_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"系統結果已保存到: {results_dir}")
            
        except Exception as e:
            logger.error(f"保存系統結果失敗: {e}")


async def main():
    """主函數"""
    print("啟動簡化真實AI預測系統...")
    
    try:
        system = SimpleRealAISystem()
        await system.run_system()
    except Exception as e:
        logger.error(f"系統運行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
