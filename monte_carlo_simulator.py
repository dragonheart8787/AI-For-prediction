#!/usr/bin/env python3
"""蒙地卡羅模擬器 - 用於風險評估和不確定性分析"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """蒙地卡羅模擬器"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.simulation_results = {}
        
    def simulate_financial_portfolio(self, 
                                   initial_value: float = 100000,
                                   time_horizon: int = 252,
                                   num_simulations: int = 10000,
                                   volatility: float = 0.2,
                                   drift: float = 0.08) -> Dict[str, Any]:
        """模擬金融投資組合"""
        logger.info(f"開始金融投資組合蒙地卡羅模擬: {num_simulations}次模擬")
        
        # 生成隨機路徑
        returns = np.random.normal(drift/252, volatility/np.sqrt(252), 
                                 (num_simulations, time_horizon))
        
        # 計算價格路徑
        price_paths = initial_value * np.exp(np.cumsum(returns, axis=1))
        
        # 計算統計量
        final_values = price_paths[:, -1]
        max_value = np.max(price_paths, axis=1)
        min_value = np.min(price_paths, axis=1)
        
        # 風險指標
        var_95 = np.percentile(final_values, 5)  # 95% VaR
        var_99 = np.percentile(final_values, 1)  # 99% VaR
        expected_shortfall = np.mean(final_values[final_values <= var_95])
        
        results = {
            'price_paths': price_paths,
            'final_values': final_values,
            'max_values': max_value,
            'min_values': min_value,
            'statistics': {
                'mean_final': np.mean(final_values),
                'std_final': np.std(final_values),
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': expected_shortfall,
                'max_drawdown': np.mean(max_value - min_value),
                'positive_probability': np.mean(final_values > initial_value)
            }
        }
        
        self.simulation_results['financial_portfolio'] = results
        return results
    
    def simulate_weather_patterns(self,
                                base_temperature: float = 20.0,
                                temperature_volatility: float = 5.0,
                                humidity_base: float = 60.0,
                                humidity_volatility: float = 15.0,
                                time_horizon: int = 365,
                                num_simulations: int = 5000) -> Dict[str, Any]:
        """模擬天氣模式"""
        logger.info(f"開始天氣模式蒙地卡羅模擬: {num_simulations}次模擬")
        
        # 季節性趨勢
        seasonal_trend = 10 * np.sin(2 * np.pi * np.arange(time_horizon) / 365)
        
        # 溫度模擬
        temp_noise = np.random.normal(0, temperature_volatility, 
                                    (num_simulations, time_horizon))
        temperatures = base_temperature + seasonal_trend + temp_noise
        
        # 濕度模擬（與溫度相關）
        humidity_noise = np.random.normal(0, humidity_volatility, 
                                        (num_simulations, time_horizon))
        humidity = humidity_base - 0.5 * (temperatures - base_temperature) + humidity_noise
        humidity = np.clip(humidity, 0, 100)
        
        # 極端天氣事件
        extreme_events = np.random.poisson(0.1, (num_simulations, time_horizon))
        
        results = {
            'temperatures': temperatures,
            'humidity': humidity,
            'extreme_events': extreme_events,
            'statistics': {
                'mean_temp': np.mean(temperatures),
                'std_temp': np.std(temperatures),
                'mean_humidity': np.mean(humidity),
                'extreme_event_prob': np.mean(extreme_events > 0),
                'temp_range': (np.min(temperatures), np.max(temperatures))
            }
        }
        
        self.simulation_results['weather_patterns'] = results
        return results
    
    def simulate_disease_spread(self,
                              initial_cases: int = 100,
                              time_horizon: int = 100,
                              num_simulations: int = 5000,
                              infection_rate: float = 0.3,
                              recovery_rate: float = 0.1,
                              death_rate: float = 0.02) -> Dict[str, Any]:
        """模擬疾病傳播"""
        logger.info(f"開始疾病傳播蒙地卡羅模擬: {num_simulations}次模擬")
        
        # SIR模型模擬
        all_simulations = []
        
        for _ in range(num_simulations):
            # 初始化
            S = [10000 - initial_cases]  # 易感人群
            I = [initial_cases]  # 感染者
            R = [0]  # 康復者
            D = [0]  # 死亡者
            
            for t in range(time_horizon - 1):
                # 新增感染
                new_infections = int(infection_rate * S[-1] * I[-1] / 10000)
                new_infections = min(new_infections, S[-1])
                
                # 新增康復
                new_recoveries = int(recovery_rate * I[-1])
                new_recoveries = min(new_recoveries, I[-1])
                
                # 新增死亡
                new_deaths = int(death_rate * I[-1])
                new_deaths = min(new_deaths, I[-1])
                
                # 更新狀態
                S.append(S[-1] - new_infections)
                I.append(I[-1] + new_infections - new_recoveries - new_deaths)
                R.append(R[-1] + new_recoveries)
                D.append(D[-1] + new_deaths)
            
            all_simulations.append({
                'S': S, 'I': I, 'R': R, 'D': D
            })
        
        # 計算統計量
        peak_cases = [max(sim['I']) for sim in all_simulations]
        total_deaths = [sim['D'][-1] for sim in all_simulations]
        
        results = {
            'simulations': all_simulations,
            'peak_cases': peak_cases,
            'total_deaths': total_deaths,
            'statistics': {
                'mean_peak_cases': np.mean(peak_cases),
                'std_peak_cases': np.std(peak_cases),
                'mean_total_deaths': np.mean(total_deaths),
                'peak_case_95ci': np.percentile(peak_cases, [2.5, 97.5]),
                'death_95ci': np.percentile(total_deaths, [2.5, 97.5])
            }
        }
        
        self.simulation_results['disease_spread'] = results
        return results
    
    def simulate_energy_demand(self,
                             base_demand: float = 1000.0,
                             demand_volatility: float = 100.0,
                             time_horizon: int = 8760,  # 一年小時數
                             num_simulations: int = 5000,
                             seasonal_factor: float = 0.3) -> Dict[str, Any]:
        """模擬能源需求"""
        logger.info(f"開始能源需求蒙地卡羅模擬: {num_simulations}次模擬")
        
        # 季節性和日間變化
        seasonal_pattern = seasonal_factor * np.sin(2 * np.pi * np.arange(time_horizon) / 8760)
        daily_pattern = 0.2 * np.sin(2 * np.pi * np.arange(time_horizon) / 24)
        
        # 需求模擬
        demand_noise = np.random.normal(0, demand_volatility, 
                                      (num_simulations, time_horizon))
        demands = base_demand * (1 + seasonal_pattern + daily_pattern) + demand_noise
        demands = np.maximum(demands, 0)  # 需求不能為負
        
        # 峰值需求
        peak_demands = np.max(demands, axis=1)
        
        # 供應中斷模擬
        supply_interruptions = np.random.poisson(0.01, (num_simulations, time_horizon))
        
        results = {
            'demands': demands,
            'peak_demands': peak_demands,
            'supply_interruptions': supply_interruptions,
            'statistics': {
                'mean_demand': np.mean(demands),
                'std_demand': np.std(demands),
                'mean_peak': np.mean(peak_demands),
                'peak_95ci': np.percentile(peak_demands, [2.5, 97.5]),
                'interruption_prob': np.mean(supply_interruptions > 0)
            }
        }
        
        self.simulation_results['energy_demand'] = results
        return results
    
    def perform_sensitivity_analysis(self, 
                                   base_params: Dict[str, float],
                                   param_ranges: Dict[str, Tuple[float, float]],
                                   num_points: int = 10) -> Dict[str, Any]:
        """執行敏感性分析"""
        logger.info("開始敏感性分析")
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, num_points)
            output_values = []
            
            for val in param_values:
                # 更新參數
                test_params = base_params.copy()
                test_params[param_name] = val
                
                # 執行模擬（這裡簡化為示例）
                if 'volatility' in param_name:
                    output = self._quick_financial_sim(test_params)
                elif 'temperature' in param_name:
                    output = self._quick_weather_sim(test_params)
                else:
                    output = val * 100  # 預設輸出
                
                output_values.append(output)
            
            sensitivity_results[param_name] = {
                'values': param_values,
                'outputs': output_values,
                'sensitivity': np.polyfit(param_values, output_values, 1)[0]
            }
        
        return sensitivity_results
    
    def _quick_financial_sim(self, params: Dict[str, float]) -> float:
        """快速金融模擬"""
        return params.get('volatility', 0.2) * 1000
    
    def _quick_weather_sim(self, params: Dict[str, float]) -> float:
        """快速天氣模擬"""
        return params.get('temperature', 20.0) * 2
    
    def generate_confidence_intervals(self, 
                                    data: np.ndarray, 
                                    confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> Dict[str, Tuple[float, float]]:
        """生成置信區間"""
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(data, lower_percentile)
            upper_bound = np.percentile(data, upper_percentile)
            
            intervals[f'{int(level*100)}%'] = (lower_bound, upper_bound)
        
        return intervals
    
    def plot_simulation_results(self, simulation_type: str, save_path: str = None):
        """繪製模擬結果"""
        if simulation_type not in self.simulation_results:
            logger.error(f"模擬類型 {simulation_type} 不存在")
            return
        
        results = self.simulation_results[simulation_type]
        
        if simulation_type == 'financial_portfolio':
            self._plot_financial_results(results, save_path)
        elif simulation_type == 'weather_patterns':
            self._plot_weather_results(results, save_path)
        elif simulation_type == 'disease_spread':
            self._plot_disease_results(results, save_path)
        elif simulation_type == 'energy_demand':
            self._plot_energy_results(results, save_path)
    
    def _plot_financial_results(self, results: Dict[str, Any], save_path: str = None):
        """繪製金融結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 價格路徑
        price_paths = results['price_paths']
        for i in range(min(100, price_paths.shape[0])):
            axes[0, 0].plot(price_paths[i], alpha=0.1, color='blue')
        axes[0, 0].set_title('投資組合價格路徑')
        axes[0, 0].set_xlabel('時間')
        axes[0, 0].set_ylabel('價值')
        
        # 最終價值分布
        axes[0, 1].hist(results['final_values'], bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(results['statistics']['mean_final'], color='red', linestyle='--', label='平均值')
        axes[0, 1].set_title('最終價值分布')
        axes[0, 1].legend()
        
        # 最大回撤
        axes[1, 0].hist(results['max_values'] - results['min_values'], bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_title('最大回撤分布')
        
        # 風險指標
        stats_text = f"""
        平均值: {results['statistics']['mean_final']:.2f}
        標準差: {results['statistics']['std_final']:.2f}
        95% VaR: {results['statistics']['var_95']:.2f}
        99% VaR: {results['statistics']['var_99']:.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title('統計摘要')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _plot_weather_results(self, results: Dict[str, Any], save_path: str = None):
        """繪製天氣結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 溫度路徑
        temperatures = results['temperatures']
        for i in range(min(50, temperatures.shape[0])):
            axes[0, 0].plot(temperatures[i], alpha=0.1, color='red')
        axes[0, 0].set_title('溫度變化路徑')
        axes[0, 0].set_xlabel('時間 (天)')
        axes[0, 0].set_ylabel('溫度 (°C)')
        
        # 濕度路徑
        humidity = results['humidity']
        for i in range(min(50, humidity.shape[0])):
            axes[0, 1].plot(humidity[i], alpha=0.1, color='blue')
        axes[0, 1].set_title('濕度變化路徑')
        axes[0, 1].set_xlabel('時間 (天)')
        axes[0, 1].set_ylabel('濕度 (%)')
        
        # 溫度分布
        axes[1, 0].hist(temperatures.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('溫度分布')
        
        # 極端事件
        extreme_events = results['extreme_events']
        axes[1, 1].hist(np.sum(extreme_events, axis=1), bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_title('極端天氣事件頻率')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _plot_disease_results(self, results: Dict[str, Any], save_path: str = None):
        """繪製疾病結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 感染曲線
        simulations = results['simulations']
        for i in range(min(20, len(simulations))):
            axes[0, 0].plot(simulations[i]['I'], alpha=0.3, color='red')
        axes[0, 0].set_title('感染人數變化')
        axes[0, 0].set_xlabel('時間')
        axes[0, 0].set_ylabel('感染人數')
        
        # 峰值病例分布
        axes[0, 1].hist(results['peak_cases'], bins=50, alpha=0.7, color='red')
        axes[0, 1].set_title('峰值病例分布')
        
        # 死亡人數分布
        axes[1, 0].hist(results['total_deaths'], bins=50, alpha=0.7, color='black')
        axes[1, 0].set_title('總死亡人數分布')
        
        # 統計摘要
        stats_text = f"""
        平均峰值: {results['statistics']['mean_peak_cases']:.0f}
        平均死亡: {results['statistics']['mean_total_deaths']:.0f}
        95% CI峰值: {results['statistics']['peak_case_95ci'][0]:.0f} - {results['statistics']['peak_case_95ci'][1]:.0f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title('統計摘要')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _plot_energy_results(self, results: Dict[str, Any], save_path: str = None):
        """繪製能源結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 需求路徑
        demands = results['demands']
        for i in range(min(50, demands.shape[0])):
            axes[0, 0].plot(demands[i], alpha=0.1, color='blue')
        axes[0, 0].set_title('能源需求路徑')
        axes[0, 0].set_xlabel('時間 (小時)')
        axes[0, 0].set_ylabel('需求 (MW)')
        
        # 峰值需求分布
        axes[0, 1].hist(results['peak_demands'], bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_title('峰值需求分布')
        
        # 需求分布
        axes[1, 0].hist(demands.flatten(), bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('需求分布')
        
        # 統計摘要
        stats_text = f"""
        平均需求: {results['statistics']['mean_demand']:.0f}
        平均峰值: {results['statistics']['mean_peak']:.0f}
        95% CI峰值: {results['statistics']['peak_95ci'][0]:.0f} - {results['statistics']['peak_95ci'][1]:.0f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title('統計摘要')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    """測試蒙地卡羅模擬器"""
    simulator = MonteCarloSimulator()
    
    # 金融投資組合模擬
    print("=== 金融投資組合模擬 ===")
    financial_results = simulator.simulate_financial_portfolio()
    print(f"平均最終價值: ${financial_results['statistics']['mean_final']:,.2f}")
    print(f"95% VaR: ${financial_results['statistics']['var_95']:,.2f}")
    
    # 天氣模式模擬
    print("\n=== 天氣模式模擬 ===")
    weather_results = simulator.simulate_weather_patterns()
    print(f"平均溫度: {weather_results['statistics']['mean_temp']:.1f}°C")
    print(f"極端天氣概率: {weather_results['statistics']['extreme_event_prob']:.2%}")
    
    # 疾病傳播模擬
    print("\n=== 疾病傳播模擬 ===")
    disease_results = simulator.simulate_disease_spread()
    print(f"平均峰值病例: {disease_results['statistics']['mean_peak_cases']:.0f}")
    print(f"平均死亡人數: {disease_results['statistics']['mean_total_deaths']:.0f}")
    
    # 能源需求模擬
    print("\n=== 能源需求模擬 ===")
    energy_results = simulator.simulate_energy_demand()
    print(f"平均需求: {energy_results['statistics']['mean_demand']:.0f} MW")
    print(f"平均峰值: {energy_results['statistics']['mean_peak']:.0f} MW")
    
    # 敏感性分析
    print("\n=== 敏感性分析 ===")
    base_params = {'volatility': 0.2, 'drift': 0.08}
    param_ranges = {'volatility': (0.1, 0.4), 'drift': (0.05, 0.12)}
    sensitivity = simulator.perform_sensitivity_analysis(base_params, param_ranges)
    
    for param, result in sensitivity.items():
        print(f"{param} 敏感性: {result['sensitivity']:.2f}")
    
    # 繪製結果
    simulator.plot_simulation_results('financial_portfolio')
    simulator.plot_simulation_results('weather_patterns')
    simulator.plot_simulation_results('disease_spread')
    simulator.plot_simulation_results('energy_demand')

if __name__ == "__main__":
    main()
