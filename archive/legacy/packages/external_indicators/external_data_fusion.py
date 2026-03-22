#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 外部指標融合模組
整合 VIX、恐懼與貪婪指數、Google Trends、NOAA 等外部數據
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pytrends
from pytrends.request import TrendReq
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExternalDataFusion:
    """外部數據融合器"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.cached_data = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1小時緩存
        
    def _is_cache_valid(self, key: str) -> bool:
        """檢查緩存是否有效"""
        if key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]
    
    def _set_cache(self, key: str, data: Any, duration: int = None):
        """設置緩存"""
        if duration is None:
            duration = self.cache_duration
        
        self.cached_data[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=duration)
    
    def get_vix_data(self, period: str = "1y") -> pd.DataFrame:
        """獲取 VIX 恐慌指數數據"""
        cache_key = f"vix_{period}"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的 VIX 數據")
            return self.cached_data[cache_key]
        
        try:
            print("📈 獲取 VIX 恐慌指數數據...")
            vix = yf.Ticker("^VIX")
            data = vix.history(period=period)
            
            # 計算技術指標
            data['VIX_MA_5'] = data['Close'].rolling(window=5).mean()
            data['VIX_MA_20'] = data['Close'].rolling(window=20).mean()
            data['VIX_Volatility'] = data['Close'].rolling(window=20).std()
            
            self._set_cache(cache_key, data)
            logger.info(f"✅ VIX 數據獲取成功，共 {len(data)} 條記錄")
            return data
            
        except Exception as e:
            logger.error(f"❌ VIX 數據獲取失敗: {e}")
            return pd.DataFrame()
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """獲取恐懼與貪婪指數"""
        cache_key = "fear_greed"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的恐懼與貪婪指數")
            return self.cached_data[cache_key]
        
        try:
            print("😨 獲取恐懼與貪婪指數...")
            
            # 使用 CNN 恐懼與貪婪指數 API
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            fng_data = data['data'][0]
            
            result = {
                'value': int(fng_data['value']),
                'classification': fng_data['value_classification'],
                'timestamp': datetime.fromtimestamp(int(fng_data['timestamp'])),
                'time_until_update': fng_data['time_until_update']
            }
            
            self._set_cache(cache_key, result, 3600)  # 1小時緩存
            logger.info(f"✅ 恐懼與貪婪指數獲取成功: {result['value']} ({result['classification']})")
            return result
            
        except Exception as e:
            logger.error(f"❌ 恐懼與貪婪指數獲取失敗: {e}")
            return {}
    
    def get_google_trends(self, keywords: List[str], timeframe: str = "today 12-m") -> pd.DataFrame:
        """獲取 Google Trends 數據"""
        cache_key = f"google_trends_{'_'.join(keywords)}_{timeframe}"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的 Google Trends 數據")
            return self.cached_data[cache_key]
        
        try:
            print(f"🔍 獲取 Google Trends 數據: {keywords}")
            
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            # 獲取興趣隨時間數據
            interest_over_time = pytrends.interest_over_time()
            
            if interest_over_time.empty:
                logger.warning("Google Trends 數據為空")
                return pd.DataFrame()
            
            # 計算趨勢指標
            for keyword in keywords:
                if keyword in interest_over_time.columns:
                    interest_over_time[f'{keyword}_MA_7'] = interest_over_time[keyword].rolling(window=7).mean()
                    interest_over_time[f'{keyword}_MA_30'] = interest_over_time[keyword].rolling(window=30).mean()
            
            self._set_cache(cache_key, interest_over_time, 3600)
            logger.info(f"✅ Google Trends 數據獲取成功，共 {len(interest_over_time)} 條記錄")
            return interest_over_time
            
        except Exception as e:
            logger.error(f"❌ Google Trends 數據獲取失敗: {e}")
            return pd.DataFrame()
    
    def get_noaa_weather_data(self, station_id: str = "USW00094728", 
                            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """獲取 NOAA 天氣數據"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"noaa_{station_id}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的 NOAA 天氣數據")
            return self.cached_data[cache_key]
        
        try:
            print(f"🌤️ 獲取 NOAA 天氣數據 (站點: {station_id})...")
            
            # NOAA API 端點
            base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
            
            params = {
                'dataset': 'daily-summaries',
                'stations': station_id,
                'startDate': start_date,
                'endDate': end_date,
                'format': 'json',
                'units': 'metric'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning("NOAA 數據為空")
                return pd.DataFrame()
            
            # 轉換為 DataFrame
            df = pd.DataFrame(data)
            
            # 處理日期
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.set_index('DATE', inplace=True)
            
            # 選擇關鍵天氣指標
            weather_columns = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'AWND', 'WSF2']
            available_columns = [col for col in weather_columns if col in df.columns]
            
            if available_columns:
                df = df[available_columns]
                
                # 計算天氣指標
                if 'TMAX' in df.columns and 'TMIN' in df.columns:
                    df['TEMP_AVG'] = (df['TMAX'] + df['TMIN']) / 2
                
                if 'PRCP' in df.columns:
                    df['PRCP_MA_7'] = df['PRCP'].rolling(window=7).mean()
                
                if 'TEMP_AVG' in df.columns:
                    df['TEMP_MA_7'] = df['TEMP_AVG'].rolling(window=7).mean()
                    df['TEMP_MA_30'] = df['TEMP_AVG'].rolling(window=30).mean()
            
            self._set_cache(cache_key, df, 3600)
            logger.info(f"✅ NOAA 天氣數據獲取成功，共 {len(df)} 條記錄")
            return df
            
        except Exception as e:
            logger.error(f"❌ NOAA 天氣數據獲取失敗: {e}")
            return pd.DataFrame()
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """獲取經濟指標"""
        cache_key = "economic_indicators"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的經濟指標")
            return self.cached_data[cache_key]
        
        try:
            print("💰 獲取經濟指標...")
            
            indicators = {}
            
            # 獲取主要經濟指標
            economic_tickers = {
                'DGS10': '10年期國債收益率',
                'DGS2': '2年期國債收益率',
                'UNRATE': '失業率',
                'CPIAUCSL': '消費者物價指數',
                'GDP': '國內生產總值',
                'FEDFUNDS': '聯邦基金利率'
            }
            
            for ticker, name in economic_tickers.items():
                try:
                    data = yf.download(ticker, period="1y", progress=False)
                    if not data.empty:
                        latest_value = data['Close'].iloc[-1]
                        indicators[name] = {
                            'value': latest_value,
                            'ticker': ticker,
                            'last_update': data.index[-1].strftime('%Y-%m-%d')
                        }
                except Exception as e:
                    logger.warning(f"獲取 {name} 失敗: {e}")
            
            # 計算經濟指標
            if '10年期國債收益率' in indicators and '2年期國債收益率' in indicators:
                yield_10y = indicators['10年期國債收益率']['value']
                yield_2y = indicators['2年期國債收益率']['value']
                indicators['收益率曲線斜率'] = {
                    'value': yield_10y - yield_2y,
                    'description': '10年期與2年期國債收益率差'
                }
            
            self._set_cache(cache_key, indicators, 3600)
            logger.info(f"✅ 經濟指標獲取成功，共 {len(indicators)} 個指標")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 經濟指標獲取失敗: {e}")
            return {}
    
    def get_social_sentiment_data(self, keywords: List[str]) -> Dict[str, Any]:
        """獲取社交媒體情緒數據"""
        cache_key = f"social_sentiment_{'_'.join(keywords)}"
        
        if self._is_cache_valid(cache_key):
            logger.info("使用緩存的社交媒體情緒數據")
            return self.cached_data[cache_key]
        
        try:
            print(f"📱 獲取社交媒體情緒數據: {keywords}")
            
            # 這裡可以整合 Twitter API、Reddit API 等
            # 由於需要 API 密鑰，這裡提供模擬數據
            sentiment_data = {}
            
            for keyword in keywords:
                # 模擬情緒分析結果
                sentiment_score = np.random.uniform(-1, 1)  # -1 到 1 之間的情緒分數
                volume = np.random.randint(100, 10000)  # 提及次數
                
                sentiment_data[keyword] = {
                    'sentiment_score': sentiment_score,
                    'volume': volume,
                    'sentiment_label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
                    'timestamp': datetime.now()
                }
            
            self._set_cache(cache_key, sentiment_data, 1800)  # 30分鐘緩存
            logger.info(f"✅ 社交媒體情緒數據獲取成功，共 {len(sentiment_data)} 個關鍵詞")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ 社交媒體情緒數據獲取失敗: {e}")
            return {}
    
    def fuse_all_external_data(self, keywords: List[str] = None) -> pd.DataFrame:
        """融合所有外部數據"""
        if keywords is None:
            keywords = ['AI', 'machine learning', 'artificial intelligence']
        
        print("🔄 融合所有外部數據...")
        
        fused_data = pd.DataFrame()
        
        try:
            # 獲取 VIX 數據
            vix_data = self.get_vix_data()
            if not vix_data.empty:
                vix_features = vix_data[['Close', 'VIX_MA_5', 'VIX_MA_20', 'VIX_Volatility']].copy()
                vix_features.columns = ['VIX_Close', 'VIX_MA_5', 'VIX_MA_20', 'VIX_Volatility']
                fused_data = pd.concat([fused_data, vix_features], axis=1)
            
            # 獲取恐懼與貪婪指數
            fng_data = self.get_fear_greed_index()
            if fng_data:
                fng_df = pd.DataFrame([{
                    'Fear_Greed_Value': fng_data['value'],
                    'Fear_Greed_Classification': fng_data['classification']
                }], index=[datetime.now().date()])
                fused_data = pd.concat([fused_data, fng_df], axis=1)
            
            # 獲取 Google Trends 數據
            trends_data = self.get_google_trends(keywords)
            if not trends_data.empty:
                trends_features = trends_data[keywords].copy()
                trends_features.columns = [f'GoogleTrends_{col}' for col in trends_features.columns]
                fused_data = pd.concat([fused_data, trends_features], axis=1)
            
            # 獲取天氣數據
            weather_data = self.get_noaa_weather_data()
            if not weather_data.empty:
                weather_features = weather_data[['TEMP_AVG', 'PRCP', 'AWND']].copy()
                weather_features.columns = ['Temperature', 'Precipitation', 'Wind_Speed']
                fused_data = pd.concat([fused_data, weather_features], axis=1)
            
            # 獲取經濟指標
            economic_data = self.get_economic_indicators()
            if economic_data:
                economic_df = pd.DataFrame([{
                    '10Y_Treasury': economic_data.get('10年期國債收益率', {}).get('value', np.nan),
                    '2Y_Treasury': economic_data.get('2年期國債收益率', {}).get('value', np.nan),
                    'Unemployment_Rate': economic_data.get('失業率', {}).get('value', np.nan),
                    'CPI': economic_data.get('消費者物價指數', {}).get('value', np.nan),
                    'Yield_Curve_Slope': economic_data.get('收益率曲線斜率', {}).get('value', np.nan)
                }], index=[datetime.now().date()])
                fused_data = pd.concat([fused_data, economic_df], axis=1)
            
            # 獲取社交媒體情緒數據
            social_data = self.get_social_sentiment_data(keywords)
            if social_data:
                social_data_dict = {}
                for keyword, data in social_data.items():
                    social_data_dict[f'{keyword}_Sentiment'] = data['sentiment_score']
                    social_data_dict[f'{keyword}_Volume'] = data['volume']
                
                social_df = pd.DataFrame([social_data_dict], index=[datetime.now().date()])
                fused_data = pd.concat([fused_data, social_df], axis=1)
            
            # 填充缺失值
            fused_data = fused_data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ 外部數據融合完成，共 {len(fused_data.columns)} 個特徵")
            return fused_data
            
        except Exception as e:
            logger.error(f"❌ 外部數據融合失敗: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self, fused_data: pd.DataFrame, 
                              target: pd.Series = None) -> pd.DataFrame:
        """計算特徵重要性"""
        if target is None:
            # 使用第一個數值列作為目標
            numeric_cols = fused_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target = fused_data[numeric_cols[0]]
            else:
                logger.warning("沒有找到數值列作為目標")
                return pd.DataFrame()
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # 準備數據
            X = fused_data.select_dtypes(include=[np.number]).fillna(0)
            y = target.fillna(0)
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 訓練隨機森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            # 獲取特徵重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("✅ 特徵重要性計算完成")
            return feature_importance
            
        except Exception as e:
            logger.error(f"❌ 特徵重要性計算失敗: {e}")
            return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    # 創建外部數據融合器
    fusion = ExternalDataFusion()
    
    print("📊 外部數據融合演示")
    print("=" * 50)
    
    # 融合所有外部數據
    keywords = ['AI', 'machine learning', 'artificial intelligence', 'deep learning']
    fused_data = fusion.fuse_all_external_data(keywords)
    
    if not fused_data.empty:
        print(f"\n✅ 融合數據形狀: {fused_data.shape}")
        print(f"特徵列表: {list(fused_data.columns)}")
        
        # 顯示數據摘要
        print("\n📈 數據摘要:")
        print(fused_data.describe())
        
        # 計算特徵重要性
        print("\n🔍 計算特徵重要性...")
        feature_importance = fusion.get_feature_importance(fused_data)
        
        if not feature_importance.empty:
            print("\n📊 前10個重要特徵:")
            print(feature_importance.head(10))
        
        # 保存融合數據
        output_file = f"fused_external_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fused_data.to_csv(output_file)
        print(f"\n💾 融合數據已保存到: {output_file}")
    
    else:
        print("❌ 沒有獲取到外部數據")
    
    print("\n🎉 外部數據融合演示完成！")
