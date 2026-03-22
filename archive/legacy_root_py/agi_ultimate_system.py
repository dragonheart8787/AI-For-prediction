#!/usr/bin/env python3
"""終極AGI系統 - 整合蒙地卡羅模擬、自動API選擇、完美預測模型"""
import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import sqlite3

# 導入自定義模組
from monte_carlo_simulator import MonteCarloSimulator
from auto_api_selector import AutoAPISelector, DataRequirement
from perfect_prediction_model import PerfectPredictionModel
from agi_enhanced_real_prediction import EnhancedAGISystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateAGISystem:
    """終極AGI系統"""
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.api_selector = AutoAPISelector()
        self.prediction_model = PerfectPredictionModel()
        self.enhanced_agi = EnhancedAGISystem()
        
        # 系統狀態
        self.system_status = {
            'initialized': False,
            'models_loaded': False,
            'apis_tested': False,
            'last_update': None
        }
        
        # 預測歷史
        self.prediction_history = []
        
        # 資料庫路徑
        self.db_path = "./agi_storage/ultimate_agi.db"
        self._init_database()
    
    def _init_database(self):
        """初始化資料庫"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 創建預測歷史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    domain TEXT,
                    prediction_type TEXT,
                    input_data TEXT,
                    prediction_result TEXT,
                    confidence REAL,
                    models_used TEXT,
                    monte_carlo_results TEXT,
                    api_source TEXT,
                    execution_time REAL
                )
            ''')
            
            # 創建系統狀態表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    status_type TEXT,
                    status_value TEXT,
                    details TEXT
                )
            ''')
            
            # 創建模型性能表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    performance_metrics TEXT,
                    training_time REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 終極AGI資料庫初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 資料庫初始化失敗: {e}")
    
    async def initialize_system(self):
        """初始化系統"""
        logger.info("🚀 開始初始化終極AGI系統")
        
        try:
            # 1. 載入預測模型
            logger.info("步驟1: 載入預測模型")
            self.prediction_model.load_models()
            
            # 2. 測試API連接性
            logger.info("步驟2: 測試API連接性")
            api_results = await self.api_selector.batch_test_apis()
            
            # 3. 執行蒙地卡羅模擬
            logger.info("步驟3: 執行蒙地卡羅模擬")
            self._run_monte_carlo_simulations()
            
            # 4. 更新系統狀態
            self.system_status.update({
                'initialized': True,
                'models_loaded': True,
                'apis_tested': True,
                'last_update': datetime.now().isoformat()
            })
            
            self._save_system_status('initialization', 'completed', '系統初始化完成')
            
            logger.info("✅ 終極AGI系統初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系統初始化失敗: {e}")
            self._save_system_status('initialization', 'failed', str(e))
            return False
    
    def _run_monte_carlo_simulations(self):
        """執行蒙地卡羅模擬"""
        try:
            # 金融投資組合模擬
            financial_results = self.monte_carlo.simulate_financial_portfolio(
                num_simulations=5000
            )
            
            # 天氣模式模擬
            weather_results = self.monte_carlo.simulate_weather_patterns(
                num_simulations=3000
            )
            
            # 疾病傳播模擬
            disease_results = self.monte_carlo.simulate_disease_spread(
                num_simulations=3000
            )
            
            # 能源需求模擬
            energy_results = self.monte_carlo.simulate_energy_demand(
                num_simulations=3000
            )
            
            logger.info("✅ 蒙地卡羅模擬完成")
            
        except Exception as e:
            logger.error(f"❌ 蒙地卡羅模擬失敗: {e}")
    
    async def intelligent_prediction(self, domain: str, prediction_type: str,
                                   requirements: Dict[str, Any]) -> Dict[str, Any]:
        """智能預測"""
        start_time = datetime.now()
        logger.info(f"🧠 開始智能預測: {domain} - {prediction_type}")
        
        try:
            # 1. 自動選擇最佳API
            api_requirement = self._create_api_requirement(domain, prediction_type, requirements)
            best_api, api_info = await self.api_selector.select_best_api(api_requirement)
            
            # 2. 執行蒙地卡羅模擬
            mc_results = self._execute_domain_monte_carlo(domain, prediction_type)
            
            # 3. 使用完美預測模型
            prediction_result = await self._execute_perfect_prediction(domain, prediction_type, requirements)
            
            # 4. 整合結果
            final_result = self._integrate_prediction_results(
                domain, prediction_type, prediction_result, mc_results, best_api, api_info
            )
            
            # 5. 保存預測歷史
            execution_time = (datetime.now() - start_time).total_seconds()
            self._save_prediction_history(domain, prediction_type, requirements, final_result, execution_time)
            
            logger.info(f"✅ 智能預測完成: {domain} - {prediction_type}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 智能預測失敗: {e}")
            return self._get_fallback_prediction(domain, prediction_type, str(e))
    
    def _create_api_requirement(self, domain: str, prediction_type: str, 
                               requirements: Dict[str, Any]) -> DataRequirement:
        """創建API需求"""
        # 根據領域和預測類型設定預設值
        default_requirements = {
            'financial': {
                'accuracy_requirement': 0.9,
                'cost_constraint': 0.01,
                'latency_requirement': 500
            },
            'weather': {
                'accuracy_requirement': 0.85,
                'cost_constraint': 0.001,
                'latency_requirement': 1000
            },
            'medical': {
                'accuracy_requirement': 0.95,
                'cost_constraint': 0.0,
                'latency_requirement': 2000
            },
            'energy': {
                'accuracy_requirement': 0.88,
                'cost_constraint': 0.0,
                'latency_requirement': 1500
            }
        }
        
        defaults = default_requirements.get(domain, {
            'accuracy_requirement': 0.8,
            'cost_constraint': 0.01,
            'latency_requirement': 1000
        })
        
        return DataRequirement(
            domain=domain,
            data_type=prediction_type,
            time_range=requirements.get('time_range', '1d'),
            geographic_scope=requirements.get('geographic_scope', 'global'),
            update_frequency=requirements.get('update_frequency', 'real_time'),
            accuracy_requirement=requirements.get('accuracy_requirement', defaults['accuracy_requirement']),
            cost_constraint=requirements.get('cost_constraint', defaults['cost_constraint']),
            latency_requirement=requirements.get('latency_requirement', defaults['latency_requirement'])
        )
    
    def _execute_domain_monte_carlo(self, domain: str, prediction_type: str) -> Dict[str, Any]:
        """執行領域特定的蒙地卡羅模擬"""
        try:
            if domain == 'financial':
                return self.monte_carlo.simulate_financial_portfolio(num_simulations=1000)
            elif domain == 'weather':
                return self.monte_carlo.simulate_weather_patterns(num_simulations=1000)
            elif domain == 'medical':
                return self.monte_carlo.simulate_disease_spread(num_simulations=1000)
            elif domain == 'energy':
                return self.monte_carlo.simulate_energy_demand(num_simulations=1000)
            else:
                # 通用模擬
                return self.monte_carlo.simulate_financial_portfolio(num_simulations=500)
        except Exception as e:
            logger.error(f"❌ 蒙地卡羅模擬失敗: {e}")
            return {}
    
    async def _execute_perfect_prediction(self, domain: str, prediction_type: str,
                                        requirements: Dict[str, Any]) -> Dict[str, Any]:
        """執行完美預測"""
        try:
            # 生成模擬資料（實際應用中會從API獲取真實資料）
            data = self._generate_simulation_data(domain, prediction_type, requirements)
            
            # 使用預測模型
            if self.prediction_model.ensemble.models:
                # 準備資料
                X, y = self.prediction_model.prepare_data(data, 'target_value')
                
                if len(X) > 0:
                    # 進行預測
                    predictions = self.prediction_model.predict(X)
                    
                    # 計算信心度
                    confidence = self._calculate_prediction_confidence(predictions, y)
                    
                    return {
                        'predictions': predictions.tolist(),
                        'confidence': confidence,
                        'model_summary': self.prediction_model.get_model_summary()
                    }
            
            # 如果沒有模型，使用增強AGI系統
            return await self.enhanced_agi.smart_predict_with_real_data(domain, prediction_type)
            
        except Exception as e:
            logger.error(f"❌ 完美預測執行失敗: {e}")
            return {'predictions': [[0.0]], 'confidence': 0.5, 'error': str(e)}
    
    def _generate_simulation_data(self, domain: str, prediction_type: str,
                                 requirements: Dict[str, Any]) -> pd.DataFrame:
        """生成模擬資料"""
        np.random.seed(42)
        
        # 根據領域生成不同的資料
        if domain == 'financial':
            data = self._generate_financial_data()
        elif domain == 'weather':
            data = self._generate_weather_data()
        elif domain == 'medical':
            data = self._generate_medical_data()
        elif domain == 'energy':
            data = self._generate_energy_data()
        else:
            data = self._generate_generic_data()
        
        return data
    
    def _generate_financial_data(self) -> pd.DataFrame:
        """生成金融資料"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # 模擬股票價格
        price = 100
        prices = []
        for _ in range(500):
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(1000000, 10000000, 500),
            'volatility': np.random.uniform(0.1, 0.3, 500),
            'target_value': np.random.normal(0, 1, 500)
        })
    
    def _generate_weather_data(self) -> pd.DataFrame:
        """生成天氣資料"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # 季節性溫度變化
        seasonal_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(500) / 365)
        temperatures = seasonal_temp + np.random.normal(0, 3, 500)
        
        return pd.DataFrame({
            'timestamp': dates,
            'temperature': temperatures,
            'humidity': np.random.uniform(30, 90, 500),
            'pressure': np.random.uniform(1000, 1030, 500),
            'wind_speed': np.random.uniform(0, 20, 500),
            'target_value': np.random.normal(0, 1, 500)
        })
    
    def _generate_medical_data(self) -> pd.DataFrame:
        """生成醫療資料"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # 模擬病例數變化
        base_cases = 100
        cases = []
        for _ in range(500):
            change = np.random.normal(0, 0.1)
            base_cases = max(0, base_cases * (1 + change))
            cases.append(base_cases)
        
        return pd.DataFrame({
            'timestamp': dates,
            'cases': cases,
            'deaths': [max(0, c * 0.02 + np.random.normal(0, 2)) for c in cases],
            'recovered': [max(0, c * 0.8 + np.random.normal(0, 5)) for c in cases],
            'target_value': np.random.normal(0, 1, 500)
        })
    
    def _generate_energy_data(self) -> pd.DataFrame:
        """生成能源資料"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # 模擬能源需求
        base_demand = 1000
        demands = []
        for _ in range(500):
            change = np.random.normal(0, 0.05)
            base_demand = max(0, base_demand * (1 + change))
            demands.append(base_demand)
        
        return pd.DataFrame({
            'timestamp': dates,
            'demand': demands,
            'supply': [d * (1 + np.random.normal(0, 0.1)) for d in demands],
            'price': [50 + np.random.normal(0, 5) for _ in demands],
            'target_value': np.random.normal(0, 1, 500)
        })
    
    def _generate_generic_data(self) -> pd.DataFrame:
        """生成通用資料"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        return pd.DataFrame({
            'timestamp': dates,
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(0, 1, 500),
            'feature3': np.random.normal(0, 1, 500),
            'feature4': np.random.normal(0, 1, 500),
            'target_value': np.random.normal(0, 1, 500)
        })
    
    def _calculate_prediction_confidence(self, predictions: np.ndarray, 
                                       actual_values: np.ndarray) -> float:
        """計算預測信心度"""
        try:
            if len(predictions) == 0 or len(actual_values) == 0:
                return 0.5
            
            # 計算預測準確度
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            if len(actual_values.shape) > 1:
                actual_values = actual_values.flatten()
            
            # 計算相關係數
            correlation = np.corrcoef(predictions, actual_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # 轉換為信心度分數
            confidence = (correlation + 1) / 2  # 將[-1,1]轉換為[0,1]
            confidence = max(0.1, min(0.95, confidence))  # 限制範圍
            
            return confidence
            
        except Exception as e:
            logger.error(f"❌ 計算信心度失敗: {e}")
            return 0.5
    
    def _integrate_prediction_results(self, domain: str, prediction_type: str,
                                    prediction_result: Dict[str, Any],
                                    mc_results: Dict[str, Any],
                                    best_api: Any,
                                    api_info: Dict[str, Any]) -> Dict[str, Any]:
        """整合預測結果"""
        try:
            # 基礎預測結果
            integrated_result = {
                'timestamp': datetime.now().isoformat(),
                'domain': domain,
                'prediction_type': prediction_type,
                'prediction': prediction_result.get('predictions', [[0.0]]),
                'confidence': prediction_result.get('confidence', 0.5),
                'api_source': best_api.name if best_api else 'unknown',
                'api_score': api_info.get('score', 0.0),
                'monte_carlo_insights': self._extract_mc_insights(mc_results),
                'risk_assessment': self._assess_risk(mc_results),
                'recommendations': self._generate_recommendations(domain, prediction_type, prediction_result, mc_results)
            }
            
            # 添加模型摘要
            if 'model_summary' in prediction_result:
                integrated_result['model_summary'] = prediction_result['model_summary']
            
            # 添加API資訊
            if best_api:
                integrated_result['api_details'] = {
                    'name': best_api.name,
                    'url': best_api.url,
                    'data_quality': best_api.data_quality.value,
                    'reliability': best_api.reliability_score
                }
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"❌ 整合預測結果失敗: {e}")
            return self._get_fallback_prediction(domain, prediction_type, str(e))
    
    def _extract_mc_insights(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取蒙地卡羅模擬的洞察"""
        insights = {}
        
        try:
            if 'statistics' in mc_results:
                stats = mc_results['statistics']
                
                # 提取關鍵統計量
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        insights[key] = value
                    elif isinstance(value, tuple) and len(value) == 2:
                        insights[f"{key}_min"] = value[0]
                        insights[f"{key}_max"] = value[1]
            
            # 計算風險指標
            if 'final_values' in mc_results:
                final_values = np.array(mc_results['final_values'])
                insights['var_95'] = np.percentile(final_values, 5)
                insights['var_99'] = np.percentile(final_values, 1)
                insights['expected_shortfall'] = np.mean(final_values[final_values <= insights['var_95']])
            
        except Exception as e:
            logger.error(f"❌ 提取蒙地卡羅洞察失敗: {e}")
        
        return insights
    
    def _assess_risk(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """評估風險"""
        risk_assessment = {
            'risk_level': 'medium',
            'risk_score': 0.5,
            'key_risks': [],
            'mitigation_strategies': []
        }
        
        try:
            if 'statistics' in mc_results:
                stats = mc_results['statistics']
                
                # 根據統計量評估風險
                if 'var_95' in stats:
                    var_95 = stats['var_95']
                    if var_95 < 0:
                        risk_assessment['risk_level'] = 'high'
                        risk_assessment['risk_score'] = 0.8
                        risk_assessment['key_risks'].append('高下行風險')
                        risk_assessment['mitigation_strategies'].append('考慮對沖策略')
                
                if 'volatility' in stats:
                    volatility = stats['volatility']
                    if volatility > 0.3:
                        risk_assessment['risk_level'] = 'high'
                        risk_assessment['risk_score'] = max(risk_assessment['risk_score'], 0.7)
                        risk_assessment['key_risks'].append('高波動性')
                        risk_assessment['mitigation_strategies'].append('分散投資組合')
            
        except Exception as e:
            logger.error(f"❌ 風險評估失敗: {e}")
        
        return risk_assessment
    
    def _generate_recommendations(self, domain: str, prediction_type: str,
                                 prediction_result: Dict[str, Any],
                                 mc_results: Dict[str, Any]) -> List[str]:
        """生成建議"""
        recommendations = []
        
        try:
            confidence = prediction_result.get('confidence', 0.5)
            
            # 基於信心度的建議
            if confidence < 0.6:
                recommendations.append("預測信心度較低，建議謹慎決策")
                recommendations.append("考慮收集更多資料或使用多個資料來源")
            elif confidence > 0.8:
                recommendations.append("預測信心度較高，可以考慮執行預測結果")
            else:
                recommendations.append("預測信心度中等，建議結合其他分析工具")
            
            # 基於領域的建議
            if domain == 'financial':
                recommendations.append("建議關注市場風險和流動性")
                recommendations.append("考慮技術分析和基本面分析的結合")
            elif domain == 'weather':
                recommendations.append("建議關注極端天氣事件的影響")
                recommendations.append("考慮季節性模式的影響")
            elif domain == 'medical':
                recommendations.append("建議關注公共衛生指標")
                recommendations.append("考慮醫療資源的可用性")
            elif domain == 'energy':
                recommendations.append("建議關注供需平衡")
                recommendations.append("考慮可再生能源的影響")
            
            # 基於蒙地卡羅結果的建議
            if mc_results and 'statistics' in mc_results:
                stats = mc_results['statistics']
                if 'var_95' in stats and stats['var_95'] < 0:
                    recommendations.append("蒙地卡羅模擬顯示下行風險，建議謹慎")
            
        except Exception as e:
            logger.error(f"❌ 生成建議失敗: {e}")
            recommendations.append("無法生成具體建議")
        
        return recommendations
    
    def _save_prediction_history(self, domain: str, prediction_type: str,
                                requirements: Dict[str, Any], result: Dict[str, Any],
                                execution_time: float):
        """保存預測歷史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_history 
                (timestamp, domain, prediction_type, input_data, prediction_result, 
                 confidence, models_used, monte_carlo_results, api_source, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'],
                domain,
                prediction_type,
                json.dumps(requirements),
                json.dumps(result['prediction']),
                result['confidence'],
                json.dumps(result.get('model_summary', {})),
                json.dumps(result.get('monte_carlo_insights', {})),
                result.get('api_source', 'unknown'),
                execution_time
            ))
            
            conn.commit()
            conn.close()
            
            # 更新記憶列表
            self.prediction_history.append(result)
            
        except Exception as e:
            logger.error(f"❌ 保存預測歷史失敗: {e}")
    
    def _save_system_status(self, status_type: str, status_value: str, details: str = ""):
        """保存系統狀態"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_status (timestamp, status_type, status_value, details)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                status_type,
                status_value,
                details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存系統狀態失敗: {e}")
    
    def _get_fallback_prediction(self, domain: str, prediction_type: str, error: str) -> Dict[str, Any]:
        """獲取備用預測"""
        return {
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'prediction_type': prediction_type,
            'prediction': [[0.0]],
            'confidence': 0.3,
            'error': error,
            'api_source': 'fallback',
            'api_score': 0.0,
            'monte_carlo_insights': {},
            'risk_assessment': {
                'risk_level': 'unknown',
                'risk_score': 0.5,
                'key_risks': ['系統錯誤'],
                'mitigation_strategies': ['檢查系統狀態']
            },
            'recommendations': ['系統出現錯誤，建議檢查日誌', '使用備用預測模式']
        }
    
    def get_prediction_history(self, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取預測歷史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if domain:
                cursor.execute('''
                    SELECT * FROM prediction_history 
                    WHERE domain = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (domain, limit))
            else:
                cursor.execute('''
                    SELECT * FROM prediction_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                history.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'domain': row[2],
                    'prediction_type': row[3],
                    'input_data': json.loads(row[4]),
                    'prediction_result': json.loads(row[5]),
                    'confidence': row[6],
                    'models_used': json.loads(row[7]),
                    'monte_carlo_results': json.loads(row[8]),
                    'api_source': row[9],
                    'execution_time': row[10]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"❌ 獲取預測歷史失敗: {e}")
            return []
    
    def get_system_summary(self) -> Dict[str, Any]:
        """獲取系統摘要"""
        return {
            'system_status': self.system_status,
            'total_predictions': len(self.prediction_history),
            'available_models': len(self.prediction_model.ensemble.models),
            'available_apis': len(self.api_selector.available_apis),
            'monte_carlo_simulations': len(self.monte_carlo.simulation_results),
            'last_update': self.system_status.get('last_update'),
            'database_path': self.db_path
        }
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """執行綜合演示"""
        logger.info("🚀 開始終極AGI系統綜合演示")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'steps': {},
            'summary': {}
        }
        
        # 步驟1: 系統初始化
        logger.info("步驟1: 系統初始化")
        init_success = await self.initialize_system()
        results['steps']['initialization'] = {
            'success': init_success,
            'status': self.system_status
        }
        
        if not init_success:
            results['summary'] = {'status': 'failed', 'error': '系統初始化失敗'}
            return results
        
        # 步驟2: 測試各領域預測
        logger.info("步驟2: 測試各領域預測")
        domains = ['financial', 'weather', 'medical', 'energy']
        prediction_types = ['股票預測', '天氣預報', '疾病傳播', '能源需求']
        
        step2_results = {}
        for domain, pred_type in zip(domains, prediction_types):
            result = await self.intelligent_prediction(domain, pred_type, {})
            step2_results[f"{domain}_{pred_type}"] = result
        
        results['steps']['predictions'] = step2_results
        
        # 步驟3: 蒙地卡羅模擬分析
        logger.info("步驟3: 蒙地卡羅模擬分析")
        mc_summary = {}
        for sim_type, sim_results in self.monte_carlo.simulation_results.items():
            if 'statistics' in sim_results:
                mc_summary[sim_type] = {
                    'key_metrics': list(sim_results['statistics'].keys()),
                    'sample_values': {k: v for k, v in list(sim_results['statistics'].items())[:3]}
                }
        
        results['steps']['monte_carlo'] = mc_summary
        
        # 步驟4: API性能分析
        logger.info("步驟4: API性能分析")
        api_performance = {}
        for api_name in self.api_selector.available_apis.keys():
            stats = self.api_selector.get_api_performance_stats(api_name)
            if stats:
                api_performance[api_name] = {
                    'success_rate': stats['success_rate'],
                    'average_latency': stats['average_latency_ms']
                }
        
        results['steps']['api_performance'] = api_performance
        
        # 步驟5: 模型性能分析
        logger.info("步驟5: 模型性能分析")
        model_summary = self.prediction_model.get_model_summary()
        results['steps']['model_performance'] = model_summary
        
        # 總結
        results['summary'] = {
            'total_predictions': len(step2_results),
            'successful_predictions': len([r for r in step2_results.values() if r.get('confidence', 0) > 0.5]),
            'average_confidence': np.mean([r.get('confidence', 0) for r in step2_results.values()]),
            'system_health': 'healthy' if init_success else 'unhealthy',
            'status': 'completed'
        }
        
        logger.info("✅ 終極AGI系統綜合演示完成")
        return results

async def main():
    """主函數"""
    # 創建終極AGI系統
    agi_system = UltimateAGISystem()
    
    # 執行綜合演示
    results = await agi_system.run_comprehensive_demo()
    
    # 保存結果
    with open('ultimate_agi_demo_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 顯示結果摘要
    print("🎉 終極AGI系統演示完成！")
    print(f"📊 總預測數: {results['summary']['total_predictions']}")
    print(f"✅ 成功預測數: {results['summary']['successful_predictions']}")
    print(f"📈 平均信心度: {results['summary']['average_confidence']:.2%}")
    print(f"🔧 系統健康狀態: {results['summary']['system_health']}")
    
    # 顯示系統摘要
    system_summary = agi_system.get_system_summary()
    print(f"\n🔍 系統摘要:")
    print(f"  可用模型數: {system_summary['available_models']}")
    print(f"  可用API數: {system_summary['available_apis']}")
    print(f"  蒙地卡羅模擬數: {system_summary['monte_carlo_simulations']}")
    print(f"  總預測數: {system_summary['total_predictions']}")

if __name__ == "__main__":
    asyncio.run(main())
