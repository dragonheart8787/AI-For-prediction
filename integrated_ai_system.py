#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合AI系統
結合數據爬取和AI訓練，提供完整的端到端解決方案
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
import sqlite3
import time

# 導入我們的模組
from enhanced_data_crawler import EnhancedDataCrawler
from enhanced_ai_trainer import EnhancedAITrainer

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_ai_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedAISystem:
    """整合AI系統"""
    
    def __init__(self, config_path: str = "integrated_config.json"):
        self.config = self._load_config(config_path)
        self.data_crawler = EnhancedDataCrawler()
        self.ai_trainer = EnhancedAITrainer()
        self.results_path = Path("integrated_results")
        self.results_path.mkdir(exist_ok=True)
        
        # 系統狀態
        self.system_status = {
            'last_crawling': None,
            'last_training': None,
            'total_symbols': 0,
            'trained_models': 0,
            'system_health': 'healthy'
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """加載系統配置"""
        default_config = {
            "workflow": {
                "auto_crawling": True,
                "auto_training": True,
                "crawling_interval_hours": 24,
                "training_interval_hours": 48,
                "max_symbols_per_training": 10
            },
            "data_management": {
                "min_data_points": 200,
                "data_retention_days": 365,
                "backup_enabled": True
            },
            "model_management": {
                "model_evaluation": True,
                "model_selection": "best_performance",
                "ensemble_methods": ["weighted_average", "stacking"]
            },
            "performance": {
                "enable_gpu": True,
                "parallel_processing": True,
                "max_workers": 4
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合併配置
                    for key, value in user_config.items():
                        if key in default_config:
                            if isinstance(value, dict) and isinstance(default_config[key], dict):
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
            except Exception as e:
                logger.warning(f"無法加載用戶配置: {e}")
        
        return default_config
    
    async def run_complete_workflow(self) -> Dict[str, Any]:
        """運行完整工作流程"""
        logger.info("🚀 開始運行整合AI系統完整工作流程...")
        
        start_time = time.time()
        workflow_results = {
            'crawling': {},
            'training': {},
            'evaluation': {},
            'summary': {}
        }
        
        try:
            # 步驟1: 數據爬取
            logger.info("📊 步驟1: 開始數據爬取...")
            crawling_results = await self.data_crawler.start_crawling()
            workflow_results['crawling'] = crawling_results
            
            if crawling_results.get('successful', 0) == 0:
                logger.warning("⚠️ 沒有成功爬取到數據，跳過訓練步驟")
                return workflow_results
            
            # 步驟2: 數據準備和特徵工程
            logger.info("🔧 步驟2: 數據準備和特徵工程...")
            training_data = self._prepare_training_data()
            
            if not training_data:
                logger.warning("⚠️ 沒有可用的訓練數據")
                return workflow_results
            
            # 步驟3: AI模型訓練
            logger.info("🤖 步驟3: 開始AI模型訓練...")
            training_results = self._train_all_models(training_data)
            workflow_results['training'] = training_results
            
            # 步驟4: 模型評估和選擇
            logger.info("📈 步驟4: 模型評估和選擇...")
            evaluation_results = self._evaluate_and_select_models(training_results)
            workflow_results['evaluation'] = evaluation_results
            
            # 步驟5: 生成工作流程摘要
            logger.info("📋 步驟5: 生成工作流程摘要...")
            workflow_results['summary'] = self._generate_workflow_summary(
                crawling_results, training_results, evaluation_results
            )
            
            # 更新系統狀態
            self._update_system_status(workflow_results)
            
            # 保存結果
            self._save_workflow_results(workflow_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 完整工作流程完成！耗時: {elapsed_time:.2f} 秒")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ 工作流程執行失敗: {e}")
            import traceback
            traceback.print_exc()
            return workflow_results
    
    def _prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """準備訓練數據"""
        try:
            # 獲取可用的數據
            available_data = self.data_crawler.get_available_data()
            logger.info(f"📊 可用數據: {available_data}")
            
            # 選擇數據類型進行訓練
            data_types = ['stocks', 'crypto']  # 優先訓練股票和加密貨幣
            
            all_training_data = {}
            
            for data_type in data_types:
                if data_type in available_data:
                    logger.info(f"🔍 準備 {data_type} 數據...")
                    
                    # 獲取該類型的數據
                    type_data = self.data_crawler.get_data_for_training(
                        data_type=data_type,
                        min_data_points=self.config['data_management']['min_data_points']
                    )
                    
                    if type_data:
                        all_training_data.update(type_data)
                        logger.info(f"✅ {data_type}: {len(type_data)} 個符號")
                    else:
                        logger.warning(f"⚠️ {data_type}: 沒有足夠的數據")
            
            # 限制訓練符號數量
            max_symbols = self.config['workflow']['max_symbols_per_training']
            if len(all_training_data) > max_symbols:
                selected_symbols = list(all_training_data.keys())[:max_symbols]
                all_training_data = {symbol: all_training_data[symbol] for symbol in selected_symbols}
                logger.info(f"🎯 限制訓練符號數量: {max_symbols}")
            
            logger.info(f"✅ 數據準備完成: {len(all_training_data)} 個符號")
            return all_training_data
            
        except Exception as e:
            logger.error(f"數據準備失敗: {e}")
            return {}
    
    def _train_all_models(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """訓練所有模型"""
        try:
            logger.info(f"🚀 開始訓練 {len(training_data)} 個符號的模型...")
            
            training_results = {}
            successful_symbols = 0
            
            for symbol, data in training_data.items():
                try:
                    logger.info(f"🎯 訓練符號: {symbol}")
                    
                    # 訓練所有模型
                    symbol_results = self.ai_trainer.train_all_models_for_symbol(symbol, data)
                    
                    if symbol_results:
                        training_results[symbol] = symbol_results
                        successful_symbols += 1
                        logger.info(f"✅ {symbol} 訓練完成: {len(symbol_results)} 個模型")
                    else:
                        logger.warning(f"⚠️ {symbol} 沒有成功訓練的模型")
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} 訓練失敗: {e}")
            
            logger.info(f"✅ 模型訓練完成: {successful_symbols}/{len(training_data)} 個符號成功")
            
            return {
                'total_symbols': len(training_data),
                'successful_symbols': successful_symbols,
                'failed_symbols': len(training_data) - successful_symbols,
                'symbol_results': training_results
            }
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            return {}
    
    def _evaluate_and_select_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """評估和選擇模型"""
        try:
            if not training_results or 'symbol_results' not in training_results:
                return {}
            
            symbol_results = training_results['symbol_results']
            evaluation_results = {
                'model_performance': {},
                'best_models': {},
                'ensemble_recommendations': {}
            }
            
            # 評估每個符號的模型性能
            for symbol, models in symbol_results.items():
                symbol_performance = {}
                best_model = None
                best_score = float('inf')
                
                for model_name, model_data in models.items():
                    if 'mse' in model_data:
                        score = model_data['mse']
                        symbol_performance[model_name] = {
                            'mse': score,
                            'r2': model_data.get('r2', None)
                        }
                        
                        if score < best_score:
                            best_score = score
                            best_model = model_name
                
                evaluation_results['model_performance'][symbol] = symbol_performance
                
                if best_model:
                    evaluation_results['best_models'][symbol] = {
                        'model': best_model,
                        'mse': best_score,
                        'r2': symbol_performance[best_model].get('r2', None)
                    }
            
            # 生成集成建議
            for symbol in symbol_results.keys():
                if symbol in evaluation_results['best_models']:
                    evaluation_results['ensemble_recommendations'][symbol] = {
                        'primary_model': evaluation_results['best_models'][symbol]['model'],
                        'ensemble_methods': self.config['model_management']['ensemble_methods'],
                        'weight_distribution': self._calculate_ensemble_weights(
                            evaluation_results['model_performance'][symbol]
                        )
                    }
            
            logger.info(f"✅ 模型評估完成: {len(evaluation_results['best_models'])} 個符號")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"模型評估失敗: {e}")
            return {}
    
    def _calculate_ensemble_weights(self, model_performance: Dict[str, Dict]) -> Dict[str, float]:
        """計算集成權重"""
        try:
            if not model_performance:
                return {}
            
            # 基於MSE的逆權重
            total_inverse_mse = 0
            weights = {}
            
            for model_name, perf in model_performance.items():
                if 'mse' in perf and perf['mse'] > 0:
                    inverse_mse = 1 / perf['mse']
                    total_inverse_mse += inverse_mse
                    weights[model_name] = inverse_mse
            
            # 標準化權重
            if total_inverse_mse > 0:
                for model_name in weights:
                    weights[model_name] /= total_inverse_mse
            
            return weights
            
        except Exception as e:
            logger.error(f"權重計算失敗: {e}")
            return {}
    
    def _generate_workflow_summary(self, crawling_results: Dict, 
                                 training_results: Dict, 
                                 evaluation_results: Dict) -> Dict[str, Any]:
        """生成工作流程摘要"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'workflow_duration': None,  # 將在外部設置
                'data_crawling': {
                    'total_symbols': crawling_results.get('total', 0),
                    'successful_symbols': crawling_results.get('successful', 0),
                    'failed_symbols': crawling_results.get('failed', 0),
                    'success_rate': 0
                },
                'model_training': {
                    'total_symbols': training_results.get('total_symbols', 0),
                    'successful_symbols': training_results.get('successful_symbols', 0),
                    'failed_symbols': training_results.get('failed_symbols', 0),
                    'success_rate': 0
                },
                'model_evaluation': {
                    'evaluated_symbols': len(evaluation_results.get('best_models', {})),
                    'best_models': evaluation_results.get('best_models', {}),
                    'ensemble_recommendations': len(evaluation_results.get('ensemble_recommendations', {}))
                },
                'system_health': self.system_status['system_health']
            }
            
            # 計算成功率
            if summary['data_crawling']['total_symbols'] > 0:
                summary['data_crawling']['success_rate'] = (
                    summary['data_crawling']['successful_symbols'] / 
                    summary['data_crawling']['total_symbols']
                )
            
            if summary['model_training']['total_symbols'] > 0:
                summary['model_training']['success_rate'] = (
                    summary['model_training']['successful_symbols'] / 
                    summary['model_training']['total_symbols']
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"生成摘要失敗: {e}")
            return {}
    
    def _update_system_status(self, workflow_results: Dict[str, Any]):
        """更新系統狀態"""
        try:
            self.system_status.update({
                'last_crawling': datetime.now().isoformat(),
                'last_training': datetime.now().isoformat(),
                'total_symbols': workflow_results.get('crawling', {}).get('total', 0),
                'trained_models': sum(
                    len(symbol_results) 
                    for symbol_results in workflow_results.get('training', {}).get('symbol_results', {}).values()
                )
            })
            
            # 評估系統健康狀態
            crawling_success_rate = workflow_results.get('summary', {}).get('data_crawling', {}).get('success_rate', 0)
            training_success_rate = workflow_results.get('summary', {}).get('model_training', {}).get('success_rate', 0)
            
            if crawling_success_rate >= 0.8 and training_success_rate >= 0.8:
                self.system_status['system_health'] = 'excellent'
            elif crawling_success_rate >= 0.6 and training_success_rate >= 0.6:
                self.system_status['system_health'] = 'good'
            elif crawling_success_rate >= 0.4 and training_success_rate >= 0.4:
                self.system_status['system_health'] = 'fair'
            else:
                self.system_status['system_health'] = 'poor'
            
            logger.info(f"✅ 系統狀態更新: {self.system_status['system_health']}")
            
        except Exception as e:
            logger.error(f"更新系統狀態失敗: {e}")
    
    def _save_workflow_results(self, workflow_results: Dict[str, Any]):
        """保存工作流程結果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存完整結果
            results_file = self.results_path / f"workflow_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, ensure_ascii=False)
            
            # 保存摘要
            summary_file = self.results_path / f"workflow_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results.get('summary', {}), f, indent=2, ensure_ascii=False)
            
            # 保存系統狀態
            status_file = self.results_path / "system_status.json"
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(self.system_status, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 工作流程結果已保存: {results_file}")
            
        except Exception as e:
            logger.error(f"保存結果失敗: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return self.system_status.copy()
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """獲取可用的模型"""
        try:
            models_path = Path("trained_models")
            if not models_path.exists():
                return {}
            
            available_models = {}
            
            for symbol_dir in models_path.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    models = []
                    
                    for model_file in symbol_dir.glob("*.pkl"):
                        model_name = model_file.stem
                        models.append(model_name)
                    
                    if models:
                        available_models[symbol] = models
            
            return available_models
            
        except Exception as e:
            logger.error(f"獲取可用模型失敗: {e}")
            return {}

async def main():
    """主函數"""
    logger.info("🚀 啟動整合AI系統...")
    
    # 創建系統實例
    system = IntegratedAISystem()
    
    try:
        # 運行完整工作流程
        results = await system.run_complete_workflow()
        
        # 顯示結果摘要
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n🎯 工作流程結果摘要:")
            print("=" * 50)
            
            print(f"📊 數據爬取:")
            print(f"   總計符號: {summary['data_crawling']['total_symbols']}")
            print(f"   成功: {summary['data_crawling']['successful_symbols']}")
            print(f"   成功率: {summary['data_crawling']['success_rate']:.2%}")
            
            print(f"\n🤖 模型訓練:")
            print(f"   總計符號: {summary['model_training']['total_symbols']}")
            print(f"   成功: {summary['model_training']['successful_symbols']}")
            print(f"   成功率: {summary['model_training']['success_rate']:.2%}")
            
            print(f"\n📈 模型評估:")
            print(f"   評估符號: {summary['model_evaluation']['evaluated_symbols']}")
            print(f"   集成建議: {summary['model_evaluation']['ensemble_recommendations']}")
            
            print(f"\n🏥 系統健康: {summary['system_health']}")
            
            # 顯示最佳模型
            if summary['model_evaluation']['best_models']:
                print(f"\n🏆 最佳模型:")
                for symbol, best_model in summary['model_evaluation']['best_models'].items():
                    print(f"   {symbol}: {best_model['model']} (MSE: {best_model['mse']:.4f})")
        
        # 顯示可用模型
        available_models = system.get_available_models()
        if available_models:
            print(f"\n📚 可用模型:")
            for symbol, models in available_models.items():
                print(f"   {symbol}: {', '.join(models)}")
        
        print(f"\n✅ 整合AI系統運行完成！")
        
    except Exception as e:
        logger.error(f"系統運行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
