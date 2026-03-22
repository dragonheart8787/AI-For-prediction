#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清潔模型融合系統
完全避免日誌問題，專注於核心功能
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

class CleanModelDownloader:
    """清潔模型下載器"""
    
    def __init__(self, models_dir: str = "./pretrained_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 預訓練模型配置
        self.pretrained_models = {
            'timesfm': {
                'name': 'TimesFM',
                'description': '零訓練時間序列預測模型',
                'type': 'zero_shot',
                'architecture': 'transformer',
                'input_size': 512,
                'output_size': 1
            },
            'chronos': {
                'name': 'Chronos-Bolt',
                'description': '基於Transformer的時間序列模型',
                'type': 'zero_shot',
                'architecture': 't5',
                'input_size': 256,
                'output_size': 1
            },
            'tft': {
                'name': 'Temporal Fusion Transformer',
                'description': '時間融合Transformer模型',
                'type': 'deep_learning',
                'architecture': 'transformer',
                'input_size': 128,
                'output_size': 1
            },
            'nbeats': {
                'name': 'N-BEATS',
                'description': '神經基於擴展的自適應時間序列',
                'type': 'deep_learning',
                'architecture': 'neural_network',
                'input_size': 64,
                'output_size': 1
            },
            'lstm_pretrained': {
                'name': 'Pre-trained LSTM',
                'description': '預訓練的LSTM時間序列模型',
                'type': 'deep_learning',
                'architecture': 'lstm',
                'input_size': 32,
                'output_size': 1
            }
        }
    
    def create_local_model(self, model_key: str) -> bool:
        """創建本地模型"""
        try:
            model_info = self.pretrained_models[model_key]
            model_dir = self.models_dir / model_key
            model_dir.mkdir(exist_ok=True)
            
            print(f"為模型 {model_key} 創建本地模型")
            
            # 創建本地模型配置
            local_config = {
                'model_type': model_info['architecture'],
                'input_size': model_info['input_size'],
                'output_size': model_info['output_size'],
                'hidden_size': model_info['input_size'] * 2,
                'num_layers': 2,
                'dropout': 0.1,
                'is_local': True,
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            config_file = model_dir / 'local_config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(local_config, f, ensure_ascii=False, indent=2)
            
            # 創建模型狀態文件
            status_file = model_dir / 'status.json'
            status_data = {
                'model_key': model_key,
                'name': f"Local {model_info['name']}",
                'description': f"本地創建的 {model_info['description']}",
                'type': model_info['type'],
                'architecture': model_info['architecture'],
                'input_size': model_info['input_size'],
                'output_size': model_info['output_size'],
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'local_created',
                'is_local': True
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功創建本地模型: {model_key}")
            return True
            
        except Exception as e:
            print(f"創建本地模型 {model_key} 失敗: {e}")
            return False
    
    def create_all_local_models(self) -> Dict[str, Any]:
        """創建所有本地模型"""
        print("開始創建所有本地模型...")
        
        results = {}
        for model_key in self.pretrained_models.keys():
            print(f"創建模型: {model_key}")
            result = self.create_local_model(model_key)
            results[model_key] = result
        
        # 統計結果
        successful_creations = sum(1 for r in results.values() if r)
        total_models = len(self.pretrained_models)
        
        print(f"創建完成: {successful_creations}/{total_models} 個模型成功")
        
        return {
            'total_models': total_models,
            'successful_creations': successful_creations,
            'failed_creations': total_models - successful_creations,
            'results': results
        }
    
    def get_created_models(self) -> List[str]:
        """獲取已創建的模型列表"""
        created = []
        for model_key in self.pretrained_models.keys():
            model_dir = self.models_dir / model_key
            status_file = model_dir / 'status.json'
            if status_file.exists():
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status_data = json.load(f)
                    if status_data.get('status') == 'local_created':
                        created.append(model_key)
                except Exception as e:
                    print(f"讀取模型狀態失敗 {model_key}: {e}")
        
        return created
    
    def list_available_models(self) -> List[Dict]:
        """列出所有可用模型"""
        models = []
        for model_key, model_info in self.pretrained_models.items():
            model_data = self.get_model_info(model_key)
            if model_data:
                models.append(model_data)
        return models
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """獲取模型信息"""
        if model_key not in self.pretrained_models:
            return None
        
        model_dir = self.models_dir / model_key
        status_file = model_dir / 'status.json'
        
        if status_file.exists():
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                return {**self.pretrained_models[model_key], **status_data}
            except Exception as e:
                print(f"讀取模型狀態失敗 {model_key}: {e}")
        
        return self.pretrained_models[model_key]


class CleanModelFusion:
    """清潔模型融合系統"""
    
    def __init__(self, models_dir: str = "./pretrained_models"):
        self.models_dir = Path(models_dir)
        self.downloader = CleanModelDownloader(models_dir)
    
    def create_fusion_model(self, model_keys: List[str], fusion_type: str = 'weighted_average') -> Dict[str, Any]:
        """創建融合模型"""
        available_models = self.downloader.get_created_models()
        
        # 檢查請求的模型是否可用
        missing_models = [key for key in model_keys if key not in available_models]
        if missing_models:
            print(f"缺少模型: {missing_models}")
            return {'error': f'缺少模型: {missing_models}'}
        
        # 創建融合模型配置
        fusion_config = {
            'fusion_type': fusion_type,
            'base_models': model_keys,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'created'
        }
        
        # 保存融合模型配置
        fusion_dir = self.models_dir / 'fusion_models'
        fusion_dir.mkdir(exist_ok=True)
        
        fusion_name = f"fusion_{fusion_type}_{'_'.join(model_keys)}"
        fusion_file = fusion_dir / f"{fusion_name}.json"
        
        with open(fusion_file, 'w', encoding='utf-8') as f:
            json.dump(fusion_config, f, ensure_ascii=False, indent=2)
        
        print(f"成功創建融合模型: {fusion_name}")
        return {
            'fusion_name': fusion_name,
            'config': fusion_config,
            'file_path': str(fusion_file)
        }
    
    def list_fusion_models(self) -> List[Dict]:
        """列出所有融合模型"""
        fusion_dir = self.models_dir / 'fusion_models'
        if not fusion_dir.exists():
            return []
        
        fusion_models = []
        for fusion_file in fusion_dir.glob("*.json"):
            try:
                with open(fusion_file, 'r', encoding='utf-8') as f:
                    fusion_config = json.load(f)
                fusion_models.append({
                    'name': fusion_file.stem,
                    'config': fusion_config,
                    'file_path': str(fusion_file)
                })
            except Exception as e:
                print(f"讀取融合模型失敗 {fusion_file}: {e}")
        
        return fusion_models


class CleanFusionSystem:
    """清潔融合系統主類"""
    
    def __init__(self):
        self.downloader = CleanModelDownloader()
        self.fusion = CleanModelFusion()
    
    def show_menu(self):
        """顯示主菜單"""
        print("\n" + "="*50)
        print("清潔模型融合系統")
        print("="*50)
        print("1. 創建所有本地模型")
        print("2. 查看可用模型")
        print("3. 創建融合模型")
        print("4. 查看融合模型")
        print("5. 系統狀態")
        print("6. 退出")
        print("="*50)
    
    def run(self):
        """運行主程序"""
        while True:
            self.show_menu()
            try:
                choice = input("請選擇操作 (1-6): ").strip()
                
                if choice == '1':
                    self.create_all_models()
                elif choice == '2':
                    self.show_available_models()
                elif choice == '3':
                    self.create_fusion_model()
                elif choice == '4':
                    self.show_fusion_models()
                elif choice == '5':
                    self.show_system_status()
                elif choice == '6':
                    print("感謝使用，再見！")
                    break
                else:
                    print("無效選擇，請重新輸入")
                    
            except KeyboardInterrupt:
                print("\n\n程序被中斷")
                break
            except Exception as e:
                print(f"操作失敗: {e}")
    
    def create_all_models(self):
        """創建所有模型"""
        print("\n開始創建所有本地模型...")
        result = self.downloader.create_all_local_models()
        
        if result['successful_creations'] > 0:
            print(f"成功創建 {result['successful_creations']} 個模型")
        else:
            print("沒有模型被創建")
    
    def show_available_models(self):
        """顯示可用模型"""
        print("\n可用模型列表:")
        models = self.downloader.list_available_models()
        
        if not models:
            print("沒有可用的模型")
            return
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['type']})")
            print(f"   描述: {model['description']}")
            print(f"   架構: {model['architecture']}")
            print(f"   狀態: {model.get('status', 'unknown')}")
            print()
    
    def create_fusion_model(self):
        """創建融合模型"""
        available_models = self.downloader.get_created_models()
        
        if not available_models:
            print("沒有可用的模型來創建融合模型")
            return
        
        print(f"\n可用模型: {', '.join(available_models)}")
        
        # 選擇模型
        model_input = input("請輸入要融合的模型名稱 (用逗號分隔): ").strip()
        model_keys = [key.strip() for key in model_input.split(',')]
        
        # 選擇融合類型
        print("\n融合類型:")
        print("1. weighted_average (加權平均)")
        print("2. stacking (堆疊)")
        print("3. voting (投票)")
        print("4. neural_fusion (神經網絡融合)")
        
        fusion_choice = input("請選擇融合類型 (1-4): ").strip()
        fusion_types = {
            '1': 'weighted_average',
            '2': 'stacking',
            '3': 'voting',
            '4': 'neural_fusion'
        }
        
        fusion_type = fusion_types.get(fusion_choice, 'weighted_average')
        
        # 創建融合模型
        result = self.fusion.create_fusion_model(model_keys, fusion_type)
        
        if 'error' not in result:
            print(f"成功創建融合模型: {result['fusion_name']}")
        else:
            print(f"創建融合模型失敗: {result['error']}")
    
    def show_fusion_models(self):
        """顯示融合模型"""
        print("\n融合模型列表:")
        fusion_models = self.fusion.list_fusion_models()
        
        if not fusion_models:
            print("沒有融合模型")
            return
        
        for i, fusion_model in enumerate(fusion_models, 1):
            print(f"{i}. {fusion_model['name']}")
            print(f"   類型: {fusion_model['config']['fusion_type']}")
            print(f"   基礎模型: {', '.join(fusion_model['config']['base_models'])}")
            print(f"   創建時間: {fusion_model['config']['created_time']}")
            print()
    
    def show_system_status(self):
        """顯示系統狀態"""
        print("\n系統狀態:")
        
        # 模型狀態
        total_models = len(self.downloader.pretrained_models)
        created_models = len(self.downloader.get_created_models())
        
        print(f"總模型數: {total_models}")
        print(f"已創建模型: {created_models}")
        print(f"模型創建率: {created_models/total_models*100:.1f}%")
        
        # 融合模型狀態
        fusion_models = self.fusion.list_fusion_models()
        print(f"融合模型數: {len(fusion_models)}")
        
        # 目錄信息
        try:
            models_dir_size = sum(f.stat().st_size for f in self.downloader.models_dir.rglob('*') if f.is_file())
            print(f"模型目錄大小: {models_dir_size / 1024:.1f} KB")
        except Exception as e:
            print(f"無法計算目錄大小: {e}")


def main():
    """主函數"""
    print("啟動清潔模型融合系統...")
    
    try:
        system = CleanFusionSystem()
        system.run()
    except Exception as e:
        print(f"系統運行失敗: {e}")


if __name__ == "__main__":
    main()
