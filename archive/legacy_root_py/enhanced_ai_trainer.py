#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版AI訓練器
支持融合和訓練真實AI模型，使用爬取的數據進行實際訓練
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
import sqlite3
import pickle

# 機器學習庫
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 時間序列庫
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trainer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedAITrainer:
    """增強版AI訓練器"""
    
    def __init__(self, data_path: str = "enhanced_financial_data.db"):
        self.data_path = data_path
        self.models_path = Path("trained_models")
        self.models_path.mkdir(exist_ok=True)
        self.scalers_path = Path("scalers")
        self.scalers_path.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_configs = {
            'arima': {'order': (1, 1, 1)},
            'ets': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
            'svr': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            'mlp': {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42}
        }
        
        self.scalers = {}
        self.trained_models = {}
    
    def get_training_data(self, symbols: List[str] = None, 
                         data_type: str = None,
                         min_data_points: int = 100) -> Dict[str, pd.DataFrame]:
        """獲取訓練數據"""
        conn = sqlite3.connect(self.data_path)
        
        try:
            query = "SELECT * FROM financial_data WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            query += " ORDER BY symbol, timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning("沒有找到訓練數據")
                return {}
            
            # 按符號分組並過濾數據量不足的符號
            grouped_data = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data.set_index('timestamp', inplace=True)
                symbol_data.sort_index(inplace=True)
                
                if len(symbol_data) >= min_data_points:
                    grouped_data[symbol] = symbol_data
                else:
                    logger.warning(f"符號 {symbol} 數據不足: {len(symbol_data)} < {min_data_points}")
            
            logger.info(f"✅ 獲取到 {len(grouped_data)} 個符號的訓練數據")
            return grouped_data
            
        except Exception as e:
            logger.error(f"獲取訓練數據失敗: {e}")
            return {}
        finally:
            conn.close()
    
    def engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """特徵工程"""
        try:
            df = data.copy()
            
            # 基本價格特徵
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['close'].shift(1)
            
            # 技術指標
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # 滯後特徵
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
            # 季節性特徵
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # 清理數據
            df = df.dropna()
            
            logger.info(f"✅ {symbol} 特徵工程完成: {df.shape[1]} 個特徵, {df.shape[0]} 個樣本")
            return df
            
        except Exception as e:
            logger.error(f"特徵工程失敗 {symbol}: {e}")
            return data
    
    def train_classical_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """訓練經典統計模型"""
        try:
            results = {}
            
            # ARIMA模型
            try:
                logger.info(f"🚀 訓練ARIMA模型: {symbol}")
                arima_model = ARIMA(data['close'], order=self.model_configs['arima']['order'])
                arima_fitted = arima_model.fit()
                
                # 預測
                arima_forecast = arima_fitted.forecast(steps=30)
                
                results['arima'] = {
                    'model': arima_fitted,
                    'forecast': arima_forecast
                }
                logger.info(f"✅ ARIMA訓練完成: {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ ARIMA訓練失敗: {e}")
            
            # ETS模型
            try:
                logger.info(f"🚀 訓練ETS模型: {symbol}")
                ets_model = ExponentialSmoothing(
                    data['close'],
                    trend=self.model_configs['ets']['trend'],
                    seasonal=self.model_configs['ets']['seasonal'],
                    seasonal_periods=self.model_configs['ets']['seasonal_periods']
                )
                ets_fitted = ets_model.fit()
                
                # 預測
                ets_forecast = ets_fitted.forecast(steps=30)
                
                results['ets'] = {
                    'model': ets_fitted,
                    'forecast': ets_forecast
                }
                logger.info(f"✅ ETS訓練完成: {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ ETS訓練失敗: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"經典模型訓練失敗 {symbol}: {e}")
            return {}
    
    def train_ml_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """訓練機器學習模型"""
        try:
            # 準備特徵和目標
            feature_cols = [col for col in data.columns if col != 'close' and not col.startswith('target_')]
            X = data[feature_cols].values
            y = data['close'].values
            
            # 標準化
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # 保存標準化器
            self.scalers[f"{symbol}_X"] = scaler_X
            self.scalers[f"{symbol}_y"] = scaler_y
            
            # 分割數據
            split_idx = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:split_idx]
            y_train = y_scaled[:split_idx]
            X_test = X_scaled[split_idx:]
            y_test = y_scaled[split_idx:]
            
            results = {}
            
            # 隨機森林
            try:
                logger.info(f"🚀 訓練隨機森林: {symbol}")
                rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
                rf_model.fit(X_train, y_train)
                
                rf_pred = rf_model.predict(X_test)
                rf_mse = mean_squared_error(y_test, rf_pred)
                rf_r2 = r2_score(y_test, rf_pred)
                
                results['random_forest'] = {
                    'model': rf_model,
                    'mse': rf_mse,
                    'r2': rf_r2,
                    'predictions': rf_pred
                }
                logger.info(f"✅ 隨機森林訓練完成: MSE={rf_mse:.4f}, R²={rf_r2:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ 隨機森林訓練失敗: {e}")
            
            # 梯度提升
            try:
                logger.info(f"🚀 訓練梯度提升: {symbol}")
                gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
                gb_model.fit(X_train, y_train)
                
                gb_pred = gb_model.predict(X_test)
                gb_mse = mean_squared_error(y_test, gb_pred)
                gb_r2 = r2_score(y_test, gb_pred)
                
                results['gradient_boosting'] = {
                    'model': gb_model,
                    'mse': gb_mse,
                    'r2': gb_r2,
                    'predictions': gb_pred
                }
                logger.info(f"✅ 梯度提升訓練完成: MSE={gb_mse:.4f}, R²={gb_r2:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ 梯度提升訓練失敗: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"機器學習模型訓練失敗 {symbol}: {e}")
            return {}
    
    def save_models(self, symbol: str, models: Dict[str, Any]):
        """保存訓練好的模型"""
        try:
            symbol_path = self.models_path / symbol
            symbol_path.mkdir(exist_ok=True)
            
            for model_name, model_data in models.items():
                model_path = symbol_path / f"{model_name}.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data['model'], f)
                
                logger.info(f"✅ 模型保存成功: {symbol}/{model_name}")
            
            # 保存標準化器
            scaler_path = self.scalers_path / f"{symbol}_scalers.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"✅ 模型保存完成: {symbol}")
            
        except Exception as e:
            logger.error(f"保存模型失敗 {symbol}: {e}")
    
    def train_all_models_for_symbol(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """為單個符號訓練所有模型"""
        try:
            logger.info(f"🚀 開始為 {symbol} 訓練所有模型...")
            
            # 特徵工程
            engineered_data = self.engineer_features(data, symbol)
            
            all_results = {}
            
            # 訓練經典統計模型
            logger.info(f"📊 訓練經典統計模型: {symbol}")
            classical_results = self.train_classical_models(engineered_data, symbol)
            all_results.update(classical_results)
            
            # 訓練機器學習模型
            logger.info(f"🤖 訓練機器學習模型: {symbol}")
            ml_results = self.train_ml_models(engineered_data, symbol)
            all_results.update(ml_results)
            
            # 保存模型
            if all_results:
                self.save_models(symbol, all_results)
                logger.info(f"✅ {symbol} 所有模型訓練完成: {len(all_results)} 個模型")
            else:
                logger.warning(f"⚠️ {symbol} 沒有成功訓練的模型")
            
            return all_results
            
        except Exception as e:
            logger.error(f"訓練所有模型失敗 {symbol}: {e}")
            return {}
    
    def train_models_for_multiple_symbols(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """為多個符號訓練模型"""
        logger.info(f"🚀 開始為 {len(symbols_data)} 個符號訓練模型...")
        
        all_results = {}
        
        for symbol, data in symbols_data.items():
            try:
                symbol_results = self.train_all_models_for_symbol(symbol, data)
                if symbol_results:
                    all_results[symbol] = symbol_results
                
            except Exception as e:
                logger.error(f"訓練失敗 {symbol}: {e}")
        
        logger.info(f"✅ 批量訓練完成: {len(all_results)}/{len(symbols_data)} 個符號成功")
        return all_results

def main():
    """主函數"""
    trainer = EnhancedAITrainer()
    
    try:
        # 獲取訓練數據
        logger.info("🔍 獲取訓練數據...")
        training_data = trainer.get_training_data(
            data_type="stocks",  # 只訓練股票數據
            min_data_points=200  # 至少200個數據點
        )
        
        if not training_data:
            logger.error("❌ 沒有可用的訓練數據")
            return
        
        logger.info(f"📊 獲取到 {len(training_data)} 個符號的數據")
        
        # 選擇前3個符號進行訓練（避免過長時間）
        selected_symbols = list(training_data.keys())[:3]
        selected_data = {symbol: training_data[symbol] for symbol in selected_symbols}
        
        logger.info(f"🎯 選擇訓練符號: {selected_symbols}")
        
        # 開始訓練
        results = trainer.train_models_for_multiple_symbols(selected_data)
        
        # 顯示結果摘要
        print("\n🎯 訓練結果摘要:")
        for symbol, symbol_results in results.items():
            print(f"\n📊 {symbol}:")
            for model_name, model_data in symbol_results.items():
                if 'mse' in model_data:
                    print(f"   {model_name}: MSE={model_data['mse']:.4f}")
                else:
                    print(f"   {model_name}: 訓練成功")
        
        print(f"\n✅ 訓練完成！共 {len(results)} 個符號，{sum(len(symbol_results) for symbol_results in results.values())} 個模型")
        
    except Exception as e:
        logger.error(f"訓練過程失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
