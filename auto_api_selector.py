#!/usr/bin/env python3
"""自動API選擇器 - 智能選擇最適合的API和資料來源"""
import asyncio
import aiohttp
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIType(Enum):
    """API類型枚舉"""
    FINANCIAL = "financial"
    WEATHER = "weather"
    MEDICAL = "medical"
    ENERGY = "energy"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    SATELLITE = "satellite"
    SENSOR = "sensor"

class DataQuality(Enum):
    """資料品質枚舉"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class APISource:
    """API來源資訊"""
    name: str
    url: str
    api_type: APIType
    data_quality: DataQuality
    reliability_score: float
    latency_ms: float
    cost_per_request: float
    rate_limit: int
    features: List[str]
    last_updated: datetime
    documentation_url: str
    api_key_required: bool = True

@dataclass
class DataRequirement:
    """資料需求定義"""
    domain: str
    data_type: str
    time_range: str
    geographic_scope: str
    update_frequency: str
    accuracy_requirement: float
    cost_constraint: float
    latency_requirement: float

class AutoAPISelector:
    """自動API選擇器"""
    
    def __init__(self):
        self.available_apis = self._initialize_apis()
        self.api_performance_history = {}
        self.cost_tracking = {}
        self.reliability_scores = {}
        
    def _initialize_apis(self) -> Dict[str, APISource]:
        """初始化可用的API列表"""
        apis = {}
        
        # 金融API
        apis['alpha_vantage'] = APISource(
            name="Alpha Vantage",
            url="https://www.alphavantage.co/",
            api_type=APIType.FINANCIAL,
            data_quality=DataQuality.EXCELLENT,
            reliability_score=0.95,
            latency_ms=150,
            cost_per_request=0.001,
            rate_limit=500,
            features=["stock_data", "forex", "crypto", "technical_indicators"],
            last_updated=datetime.now(),
            documentation_url="https://www.alphavantage.co/documentation/"
        )
        
        apis['yahoo_finance'] = APISource(
            name="Yahoo Finance",
            url="https://finance.yahoo.com/",
            api_type=APIType.FINANCIAL,
            data_quality=DataQuality.GOOD,
            reliability_score=0.90,
            latency_ms=200,
            cost_per_request=0.0,
            rate_limit=2000,
            features=["stock_data", "news", "portfolio_tracking"],
            last_updated=datetime.now(),
            documentation_url="https://finance.yahoo.com/",
            api_key_required=False
        )
        
        # 天氣API
        apis['openweather'] = APISource(
            name="OpenWeather",
            url="https://openweathermap.org/",
            api_type=APIType.WEATHER,
            data_quality=DataQuality.EXCELLENT,
            reliability_score=0.92,
            latency_ms=180,
            cost_per_request=0.0001,
            rate_limit=1000,
            features=["current_weather", "forecast", "historical_data", "air_quality"],
            last_updated=datetime.now(),
            documentation_url="https://openweathermap.org/api"
        )
        
        apis['weather_api'] = APISource(
            name="WeatherAPI",
            url="https://www.weatherapi.com/",
            api_type=APIType.WEATHER,
            data_quality=DataQuality.GOOD,
            reliability_score=0.88,
            latency_ms=220,
            cost_per_request=0.0002,
            rate_limit=1000000,
            features=["current_weather", "forecast", "astronomy", "sports"],
            last_updated=datetime.now(),
            documentation_url="https://www.weatherapi.com/docs/"
        )
        
        # 醫療API
        apis['who_api'] = APISource(
            name="WHO Health",
            url="https://www.who.int/",
            api_type=APIType.MEDICAL,
            data_quality=DataQuality.EXCELLENT,
            reliability_score=0.98,
            latency_ms=300,
            cost_per_request=0.0,
            rate_limit=100,
            features=["disease_data", "health_statistics", "outbreak_info"],
            last_updated=datetime.now(),
            documentation_url="https://www.who.int/data",
            api_key_required=False
        )
        
        apis['cdc_api'] = APISource(
            name="CDC Data",
            url="https://data.cdc.gov/",
            api_type=APIType.MEDICAL,
            data_quality=DataQuality.EXCELLENT,
            reliability_score=0.96,
            latency_ms=250,
            cost_per_request=0.0,
            rate_limit=1000,
            features=["disease_data", "vaccination", "mortality"],
            last_updated=datetime.now(),
            documentation_url="https://data.cdc.gov/",
            api_key_required=False
        )
        
        # 能源API
        apis['eia_api'] = APISource(
            name="EIA Energy",
            url="https://www.eia.gov/",
            api_type=APIType.ENERGY,
            data_quality=DataQuality.EXCELLENT,
            reliability_score=0.94,
            latency_ms=400,
            cost_per_request=0.0,
            rate_limit=500,
            features=["energy_consumption", "production", "prices", "forecasts"],
            last_updated=datetime.now(),
            documentation_url="https://www.eia.gov/opendata/",
            api_key_required=False
        )
        
        apis['iea_api'] = APISource(
            name="IEA Energy",
            url="https://www.iea.org/",
            api_type=APIType.ENERGY,
            data_quality=DataQuality.GOOD,
            reliability_score=0.89,
            latency_ms=350,
            cost_per_request=0.0,
            rate_limit=200,
            features=["energy_statistics", "renewables", "efficiency"],
            last_updated=datetime.now(),
            documentation_url="https://www.iea.org/data-and-statistics",
            api_key_required=False
        )
        
        return apis
    
    async def select_best_api(self, requirement: DataRequirement) -> Tuple[APISource, Dict[str, Any]]:
        """選擇最適合的API"""
        logger.info(f"開始為 {requirement.domain} 領域選擇最佳API")
        
        # 篩選符合條件的API
        candidate_apis = self._filter_candidate_apis(requirement)
        
        if not candidate_apis:
            logger.warning("沒有找到符合條件的API")
            return None, {}
        
        # 計算綜合評分
        api_scores = {}
        for api_name, api in candidate_apis.items():
            score = self._calculate_api_score(api, requirement)
            api_scores[api_name] = score
        
        # 選擇評分最高的API
        best_api_name = max(api_scores, key=api_scores.get)
        best_api = candidate_apis[best_api_name]
        best_score = api_scores[best_api_name]
        
        logger.info(f"選擇API: {best_api.name} (評分: {best_score:.3f})")
        
        # 生成選擇理由
        selection_reason = self._generate_selection_reason(best_api, requirement, best_score)
        
        return best_api, {
            'score': best_score,
            'reason': selection_reason,
            'alternatives': self._get_alternative_apis(candidate_apis, api_scores, best_api_name)
        }
    
    def _filter_candidate_apis(self, requirement: DataRequirement) -> Dict[str, APISource]:
        """篩選候選API"""
        candidates = {}
        
        for api_name, api in self.available_apis.items():
            # 檢查API類型是否匹配
            if not self._api_type_matches(api.api_type, requirement.domain):
                continue
            
            # 檢查資料品質是否滿足要求
            if not self._quality_satisfies_requirement(api.data_quality, requirement.accuracy_requirement):
                continue
            
            # 檢查成本約束
            if api.cost_per_request > requirement.cost_constraint:
                continue
            
            # 檢查延遲要求
            if api.latency_ms > requirement.latency_requirement:
                continue
            
            candidates[api_name] = api
        
        return candidates
    
    def _api_type_matches(self, api_type: APIType, domain: str) -> bool:
        """檢查API類型是否匹配領域"""
        mapping = {
            'financial': [APIType.FINANCIAL],
            'weather': [APIType.WEATHER],
            'medical': [APIType.MEDICAL],
            'energy': [APIType.ENERGY],
            'news': [APIType.NEWS],
            'social_media': [APIType.SOCIAL_MEDIA]
        }
        
        return api_type in mapping.get(domain, [])
    
    def _quality_satisfies_requirement(self, api_quality: DataQuality, required_accuracy: float) -> bool:
        """檢查API品質是否滿足準確度要求"""
        quality_scores = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.85,
            DataQuality.FAIR: 0.70,
            DataQuality.POOR: 0.50
        }
        
        return quality_scores[api_quality] >= required_accuracy
    
    def _calculate_api_score(self, api: APISource, requirement: DataRequirement) -> float:
        """計算API綜合評分"""
        # 基礎評分
        base_score = api.reliability_score * 0.4
        
        # 資料品質評分
        quality_score = self._get_quality_score(api.data_quality) * 0.25
        
        # 成本效益評分
        cost_score = self._calculate_cost_score(api, requirement) * 0.15
        
        # 延遲評分
        latency_score = self._calculate_latency_score(api, requirement) * 0.10
        
        # 功能豐富度評分
        feature_score = self._calculate_feature_score(api, requirement) * 0.10
        
        # 綜合評分
        total_score = base_score + quality_score + cost_score + latency_score + feature_score
        
        return total_score
    
    def _get_quality_score(self, quality: DataQuality) -> float:
        """獲取品質分數"""
        return {
            DataQuality.EXCELLENT: 1.0,
            DataQuality.GOOD: 0.8,
            DataQuality.FAIR: 0.6,
            DataQuality.POOR: 0.4
        }[quality]
    
    def _calculate_cost_score(self, api: APISource, requirement: DataRequirement) -> float:
        """計算成本分數"""
        if requirement.cost_constraint == 0:
            return 1.0 if api.cost_per_request == 0 else 0.0
        
        cost_ratio = api.cost_per_request / requirement.cost_constraint
        return max(0, 1 - cost_ratio)
    
    def _calculate_latency_score(self, api: APISource, requirement: DataRequirement) -> float:
        """計算延遲分數"""
        if requirement.latency_requirement == 0:
            return 1.0
        
        latency_ratio = api.latency_ms / requirement.latency_requirement
        return max(0, 1 - latency_ratio)
    
    def _calculate_feature_score(self, api: APISource, requirement: DataRequirement) -> float:
        """計算功能分數"""
        # 根據需求計算功能匹配度
        required_features = self._extract_required_features(requirement)
        matched_features = sum(1 for feature in required_features if feature in api.features)
        
        if not required_features:
            return 0.5  # 中性評分
        
        return matched_features / len(required_features)
    
    def _extract_required_features(self, requirement: DataRequirement) -> List[str]:
        """提取需求中的功能要求"""
        # 根據領域和資料類型推斷所需功能
        feature_mapping = {
            ('financial', 'stock_data'): ['stock_data', 'technical_indicators'],
            ('financial', 'forex'): ['forex', 'exchange_rates'],
            ('weather', 'forecast'): ['forecast', 'current_weather'],
            ('weather', 'historical'): ['historical_data'],
            ('medical', 'disease'): ['disease_data', 'outbreak_info'],
            ('medical', 'statistics'): ['health_statistics'],
            ('energy', 'consumption'): ['energy_consumption', 'production'],
            ('energy', 'prices'): ['prices', 'forecasts']
        }
        
        key = (requirement.domain, requirement.data_type)
        return feature_mapping.get(key, [])
    
    def _generate_selection_reason(self, api: APISource, requirement: DataRequirement, score: float) -> str:
        """生成選擇理由"""
        reasons = []
        
        if api.reliability_score > 0.9:
            reasons.append("高可靠性")
        
        if api.data_quality == DataQuality.EXCELLENT:
            reasons.append("資料品質優秀")
        
        if api.cost_per_request == 0:
            reasons.append("免費使用")
        elif api.cost_per_request < requirement.cost_constraint * 0.5:
            reasons.append("成本效益高")
        
        if api.latency_ms < requirement.latency_requirement * 0.5:
            reasons.append("響應速度快")
        
        if len(api.features) > 5:
            reasons.append("功能豐富")
        
        return f"選擇 {api.name} 的原因: {', '.join(reasons)} (綜合評分: {score:.3f})"
    
    def _get_alternative_apis(self, candidates: Dict[str, APISource], 
                            scores: Dict[str, float], best_api_name: str) -> List[Dict[str, Any]]:
        """獲取替代API選項"""
        alternatives = []
        
        for api_name, api in candidates.items():
            if api_name == best_api_name:
                continue
            
            alternatives.append({
                'name': api.name,
                'score': scores[api_name],
                'reason': f"評分: {scores[api_name]:.3f}, 品質: {api.data_quality.value}"
            })
        
        # 按評分排序
        alternatives.sort(key=lambda x: x['score'], reverse=True)
        return alternatives[:3]  # 返回前3個替代選項
    
    async def test_api_connectivity(self, api: APISource) -> Dict[str, Any]:
        """測試API連接性"""
        logger.info(f"測試API連接性: {api.name}")
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # 模擬API測試請求
                if api.api_type == APIType.FINANCIAL:
                    test_url = f"{api.url}test"
                elif api.api_type == APIType.WEATHER:
                    test_url = f"{api.url}test"
                else:
                    test_url = f"{api.url}test"
                
                # 這裡應該發送實際的測試請求
                # 為了演示，我們模擬結果
                await asyncio.sleep(0.1)  # 模擬網路延遲
                
                latency = (time.time() - start_time) * 1000
                success = random.random() > 0.1  # 90%成功率
                
                result = {
                    'api_name': api.name,
                    'success': success,
                    'latency_ms': latency,
                    'timestamp': datetime.now().isoformat(),
                    'error': None if success else "連接超時"
                }
                
                # 更新性能歷史
                self._update_performance_history(api.name, result)
                
                return result
                
        except Exception as e:
            result = {
                'api_name': api.name,
                'success': False,
                'latency_ms': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            
            self._update_performance_history(api.name, result)
            return result
    
    def _update_performance_history(self, api_name: str, result: Dict[str, Any]):
        """更新API性能歷史"""
        if api_name not in self.api_performance_history:
            self.api_performance_history[api_name] = []
        
        self.api_performance_history[api_name].append(result)
        
        # 只保留最近100條記錄
        if len(self.api_performance_history[api_name]) > 100:
            self.api_performance_history[api_name] = self.api_performance_history[api_name][-100:]
    
    def get_api_performance_stats(self, api_name: str) -> Dict[str, Any]:
        """獲取API性能統計"""
        if api_name not in self.api_performance_history:
            return {}
        
        history = self.api_performance_history[api_name]
        
        if not history:
            return {}
        
        successful_tests = [h for h in history if h['success']]
        failed_tests = [h for h in history if not h['success']]
        
        if successful_tests:
            avg_latency = np.mean([h['latency_ms'] for h in successful_tests])
            success_rate = len(successful_tests) / len(history)
        else:
            avg_latency = 0
            success_rate = 0
        
        return {
            'total_tests': len(history),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': success_rate,
            'average_latency_ms': avg_latency,
            'last_test': history[-1] if history else None
        }
    
    def recommend_api_updates(self) -> List[Dict[str, Any]]:
        """推薦API更新"""
        recommendations = []
        
        for api_name, api in self.available_apis.items():
            performance = self.get_api_performance_stats(api_name)
            
            if not performance:
                continue
            
            # 檢查是否需要更新
            if performance['success_rate'] < 0.8:
                recommendations.append({
                    'api_name': api_name,
                    'issue': '低成功率',
                    'current_rate': performance['success_rate'],
                    'recommendation': '考慮更換API或檢查配置'
                })
            
            if performance['average_latency_ms'] > api.latency_ms * 2:
                recommendations.append({
                    'api_name': api_name,
                    'issue': '延遲過高',
                    'current_latency': performance['average_latency_ms'],
                    'expected_latency': api.latency_ms,
                    'recommendation': '檢查網路連接或考慮更換API'
                })
        
        return recommendations
    
    async def batch_test_apis(self) -> Dict[str, Dict[str, Any]]:
        """批量測試所有API"""
        logger.info("開始批量測試所有API")
        
        results = {}
        tasks = []
        
        for api_name, api in self.available_apis.items():
            task = self.test_api_connectivity(api)
            tasks.append((api_name, task))
        
        # 並行執行測試
        for api_name, task in tasks:
            try:
                result = await task
                results[api_name] = result
            except Exception as e:
                results[api_name] = {
                    'api_name': api_name,
                    'success': False,
                    'latency_ms': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
        
        logger.info(f"批量測試完成，共測試 {len(results)} 個API")
        return results

def main():
    """測試自動API選擇器"""
    async def test():
        selector = AutoAPISelector()
        
        # 測試金融資料需求
        financial_requirement = DataRequirement(
            domain="financial",
            data_type="stock_data",
            time_range="1d",
            geographic_scope="global",
            update_frequency="real_time",
            accuracy_requirement=0.9,
            cost_constraint=0.01,
            latency_requirement=500
        )
        
        best_api, selection_info = await selector.select_best_api(financial_requirement)
        
        if best_api:
            print(f"最佳API: {best_api.name}")
            print(f"評分: {selection_info['score']:.3f}")
            print(f"選擇理由: {selection_info['reason']}")
            
            # 測試API連接性
            connectivity_result = await selector.test_api_connectivity(best_api)
            print(f"連接測試: {connectivity_result}")
        
        # 批量測試所有API
        print("\n批量測試所有API...")
        batch_results = await selector.batch_test_apis()
        
        for api_name, result in batch_results.items():
            print(f"{api_name}: {'成功' if result['success'] else '失敗'}")
        
        # 獲取性能統計
        print("\nAPI性能統計:")
        for api_name in selector.available_apis.keys():
            stats = selector.get_api_performance_stats(api_name)
            if stats:
                print(f"{api_name}: 成功率 {stats['success_rate']:.2%}, 平均延遲 {stats['average_latency_ms']:.1f}ms")
        
        # 獲取更新建議
        recommendations = selector.recommend_api_updates()
        if recommendations:
            print("\nAPI更新建議:")
            for rec in recommendations:
                print(f"- {rec['api_name']}: {rec['issue']} - {rec['recommendation']}")
    
    asyncio.run(test())

if __name__ == "__main__":
    main()
