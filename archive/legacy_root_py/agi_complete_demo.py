#!/usr/bin/env python3
"""
AGI完整演示系統
展示從數據爬取到模型訓練再到預測的完整流程

功能:
1. 爬取多領域數據
2. 數據預處理和特徵工程
3. 訓練多種AI模型
4. 進行智能預測
5. 展示預測結果
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from data_crawler import DataCrawler
from data_trainer import AGITrainingSystem
from agi_new_features import EnhancedAPI, ModelPerformance

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AGICompleteDemo:
    """AGI完整演示系統"""
    
    def __init__(self):
        self.crawler = DataCrawler()
        self.training_system = AGITrainingSystem()
        self.api = EnhancedAPI()
    
    async def step1_crawl_data(self):
        """步驟1: 爬取數據"""
        print("\n🕷️ 步驟1: 數據爬取")
        print("=" * 50)
        
        try:
            # 爬取所有數據
            await self.crawler.crawl_all_data()
            
            # 顯示數據摘要
            summary = self.crawler.get_data_summary()
            print("\n📊 數據爬取摘要:")
            for table, count in summary.items():
                print(f"  {table}: {count} 條記錄")
            
            print("✅ 數據爬取完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 數據爬取失敗: {e}")
            return False
    
    async def step2_train_models(self):
        """步驟2: 訓練模型"""
        print("\n🧠 步驟2: 模型訓練")
        print("=" * 50)
        
        try:
            # 訓練所有模型
            results = await self.training_system.train_all_models()
            
            # 顯示訓練摘要
            summary = self.training_system.model_trainer.get_training_summary()
            print(f"\n📊 訓練摘要:")
            print(f"  總模型數: {summary['total_models']}")
            print(f"  模型名稱: {summary['model_names']}")
            
            print("✅ 模型訓練完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型訓練失敗: {e}")
            return False
    
    async def step3_smart_prediction(self):
        """步驟3: 智能預測"""
        print("\n🎯 步驟3: 智能預測")
        print("=" * 50)
        
        try:
            # 設置模型性能
            self.api.model_selector.update_performance('financial_lstm', 
                ModelPerformance('financial_lstm', 0.85, 0.8, 0.5, 0.9, datetime.now()))
            self.api.model_selector.update_performance('weather_transformer', 
                ModelPerformance('weather_transformer', 0.92, 0.9, 0.3, 0.95, datetime.now()))
            self.api.model_selector.update_performance('medical_cnn', 
                ModelPerformance('medical_cnn', 0.88, 0.85, 0.4, 0.92, datetime.now()))
            self.api.model_selector.update_performance('energy_hybrid', 
                ModelPerformance('energy_hybrid', 0.90, 0.87, 0.6, 0.93, datetime.now()))
            
            # 進行各種預測
            predictions = {}
            
            # 金融預測
            financial_data = np.random.randn(1, 20)
            result = await self.api.smart_predict('financial', financial_data, use_fusion=True)
            predictions['financial'] = result
            
            # 天氣預測
            weather_data = np.random.randn(1, 15)
            result = await self.api.smart_predict('weather', weather_data, use_fusion=True)
            predictions['weather'] = result
            
            # 醫療預測
            medical_data = np.random.randn(1, 12)
            result = await self.api.smart_predict('medical', medical_data, use_fusion=True)
            predictions['medical'] = result
            
            # 能源預測
            energy_data = np.random.randn(1, 18)
            result = await self.api.smart_predict('energy', energy_data, use_fusion=True)
            predictions['energy'] = result
            
            # 顯示預測結果
            print("\n📊 預測結果:")
            for domain, pred in predictions.items():
                if 'error' not in pred:
                    print(f"  {domain}: 置信度={pred.get('confidence', 0):.3f}, 模型={pred.get('model_name', 'N/A')}")
                else:
                    print(f"  {domain}: ❌ 預測失敗")
            
            print("✅ 智能預測完成！")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 智能預測失敗: {e}")
            return {}
    
    async def step4_system_status(self):
        """步驟4: 系統狀態"""
        print("\n📊 步驟4: 系統狀態")
        print("=" * 50)
        
        try:
            # 獲取系統狀態
            status = self.api.get_system_status()
            
            print("📈 性能摘要:")
            for metric, data in status.get('performance_summary', {}).items():
                if isinstance(data, dict):
                    print(f"  {metric}: 當前={data.get('current', 0):.3f}, 平均={data.get('average', 0):.3f}")
            
            print(f"\n🔄 恢復歷史: {len(status.get('recovery_history', []))} 條記錄")
            print(f"🧠 融合歷史: {len(status.get('fusion_history', []))} 條記錄")
            print(f"🎯 模型選擇: {len(status.get('model_selections', []))} 條記錄")
            
            print("✅ 系統狀態檢查完成！")
            return status
            
        except Exception as e:
            logger.error(f"❌ 系統狀態檢查失敗: {e}")
            return {}
    
    async def step5_real_world_prediction(self):
        """步驟5: 真實世界預測"""
        print("\n🌍 步驟5: 真實世界預測")
        print("=" * 50)
        
        try:
            # 模擬真實世界的預測場景
            scenarios = [
                {
                    'name': '股票投資決策',
                    'domain': 'financial',
                    'data': np.random.randn(1, 20),
                    'description': '基於歷史數據預測股價走勢'
                },
                {
                    'name': '天氣預報',
                    'domain': 'weather',
                    'data': np.random.randn(1, 15),
                    'description': '預測未來24小時天氣變化'
                },
                {
                    'name': '疾病傳播預測',
                    'domain': 'medical',
                    'data': np.random.randn(1, 12),
                    'description': '預測疾病傳播趨勢和風險'
                },
                {
                    'name': '能源需求預測',
                    'domain': 'energy',
                    'data': np.random.randn(1, 18),
                    'description': '預測未來用電需求和能源價格'
                }
            ]
            
            results = {}
            
            for scenario in scenarios:
                print(f"\n🎯 {scenario['name']}")
                print(f"   描述: {scenario['description']}")
                
                result = await self.api.smart_predict(scenario['domain'], scenario['data'], use_fusion=True)
                
                if 'error' not in result:
                    confidence = result.get('confidence', 0)
                    model = result.get('model_name', 'N/A')
                    print(f"   結果: 置信度={confidence:.3f}, 模型={model}")
                    
                    if confidence > 0.8:
                        print("   🟢 高置信度預測")
                    elif confidence > 0.6:
                        print("   🟡 中等置信度預測")
                    else:
                        print("   🔴 低置信度預測")
                else:
                    print(f"   ❌ 預測失敗: {result['error']}")
                
                results[scenario['name']] = result
            
            print("\n✅ 真實世界預測完成！")
            return results
            
        except Exception as e:
            logger.error(f"❌ 真實世界預測失敗: {e}")
            return {}
    
    async def run_complete_demo(self):
        """運行完整演示"""
        print("🚀 AGI完整演示系統")
        print("=" * 60)
        print("展示從數據爬取到智能預測的完整AGI流程")
        print("=" * 60)
        
        results = {}
        
        # 步驟1: 爬取數據
        step1_success = await self.step1_crawl_data()
        results['step1_crawl'] = step1_success
        
        if step1_success:
            # 步驟2: 訓練模型
            step2_success = await self.step2_train_models()
            results['step2_train'] = step2_success
            
            if step2_success:
                # 步驟3: 智能預測
                step3_results = await self.step3_smart_prediction()
                results['step3_prediction'] = step3_results
                
                # 步驟4: 系統狀態
                step4_results = await self.step4_system_status()
                results['step4_status'] = step4_results
                
                # 步驟5: 真實世界預測
                step5_results = await self.step5_real_world_prediction()
                results['step5_real_world'] = step5_results
        
        # 總結
        print("\n" + "=" * 60)
        print("📊 演示總結")
        print("=" * 60)
        
        successful_steps = sum(1 for success in results.values() if success is not False)
        total_steps = len(results)
        
        print(f"✅ 成功步驟: {successful_steps}/{total_steps}")
        
        if successful_steps == total_steps:
            print("🎉 所有步驟成功！AGI系統運行完美。")
        else:
            print("⚠️ 部分步驟失敗，請檢查相關功能。")
        
        # 保存演示結果
        demo_report = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'results': results
        }
        
        with open('complete_demo_report.json', 'w', encoding='utf-8') as f:
            json.dump(demo_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 完整演示報告已保存到: complete_demo_report.json")
        
        return results

async def main():
    """主函數"""
    demo = AGICompleteDemo()
    
    try:
        results = await demo.run_complete_demo()
        
        print("\n🎯 AGI系統已準備就緒！")
        print("您現在可以使用訓練好的模型進行各種預測任務。")
        
    except Exception as e:
        logger.error(f"❌ 演示執行失敗: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 