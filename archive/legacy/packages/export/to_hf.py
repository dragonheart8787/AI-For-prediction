#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統一匯出為 Hugging Face 格式
支援從各種模型格式轉換為 HF 格式
"""

import os
import torch
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)

class HFExporter:
    """Hugging Face 格式匯出器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_from_pytorch(self, model: torch.nn.Module, tokenizer=None, 
                           config: Optional[Dict[str, Any]] = None) -> str:
        """從 PyTorch 模型匯出"""
        logger.info(f"匯出 PyTorch 模型到: {self.output_dir}")
        
        # 保存模型權重
        model_path = self.output_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        
        # 創建配置
        if config is None:
            config = self._create_default_config(model)
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 保存 tokenizer
        if tokenizer is not None:
            self._save_tokenizer(tokenizer)
        
        logger.info("PyTorch 模型匯出完成")
        return str(self.output_dir)
    
    def export_from_checkpoint(self, checkpoint_path: str, 
                              model_class=None, tokenizer=None) -> str:
        """從檢查點匯出"""
        logger.info(f"從檢查點匯出: {checkpoint_path}")
        
        # 載入檢查點
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型狀態
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 保存模型權重
        model_path = self.output_dir / "pytorch_model.bin"
        torch.save(state_dict, model_path)
        
        # 創建配置
        config = self._extract_config_from_checkpoint(checkpoint)
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 保存 tokenizer
        if tokenizer is not None:
            self._save_tokenizer(tokenizer)
        
        logger.info("檢查點匯出完成")
        return str(self.output_dir)
    
    def export_from_onnx(self, onnx_path: str, config: Optional[Dict[str, Any]] = None) -> str:
        """從 ONNX 模型匯出"""
        logger.info(f"從 ONNX 匯出: {onnx_path}")
        
        # 複製 ONNX 文件
        import shutil
        onnx_dest = self.output_dir / "model.onnx"
        shutil.copy2(onnx_path, onnx_dest)
        
        # 創建配置
        if config is None:
            config = {"model_type": "onnx"}
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("ONNX 模型匯出完成")
        return str(self.output_dir)
    
    def _create_default_config(self, model: torch.nn.Module) -> Dict[str, Any]:
        """創建預設配置"""
        config = {
            "model_type": "custom",
            "architecture": model.__class__.__name__,
            "torch_dtype": "float32"
        }
        
        # 嘗試提取模型參數
        try:
            total_params = sum(p.numel() for p in model.parameters())
            config["num_parameters"] = total_params
        except:
            pass
        
        return config
    
    def _extract_config_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """從檢查點提取配置"""
        config = {
            "model_type": "custom",
            "torch_dtype": "float32"
        }
        
        # 提取訓練相關信息
        if 'epoch' in checkpoint:
            config['training_epoch'] = checkpoint['epoch']
        if 'optimizer' in checkpoint:
            config['has_optimizer'] = True
        if 'scheduler' in checkpoint:
            config['has_scheduler'] = True
        
        return config
    
    def _save_tokenizer(self, tokenizer):
        """保存 tokenizer"""
        try:
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(self.output_dir)
            else:
                logger.warning("tokenizer 不支援 save_pretrained 方法")
        except Exception as e:
            logger.error(f"保存 tokenizer 失敗: {e}")
    
    def create_model_card(self, model_name: str, description: str = ""):
        """創建模型卡片"""
        model_card = f"""---
license: apache-2.0
tags:
- custom
- pytorch
---

# {model_name}

{description}

## 使用方式

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{self.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{self.output_dir}")
```
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)

def main():
    parser = argparse.ArgumentParser(description="匯出模型為 Hugging Face 格式")
    parser.add_argument("--input", required=True, help="輸入模型路徑")
    parser.add_argument("--output", required=True, help="輸出目錄")
    parser.add_argument("--type", choices=["pytorch", "checkpoint", "onnx"], 
                       default="pytorch", help="輸入類型")
    parser.add_argument("--config", help="配置 JSON 文件路徑")
    parser.add_argument("--tokenizer", help="tokenizer 路徑")
    parser.add_argument("--model-name", default="custom_model", help="模型名稱")
    parser.add_argument("--description", default="", help="模型描述")
    
    args = parser.parse_args()
    
    # 載入配置
    config = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 載入 tokenizer
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            logger.warning(f"載入 tokenizer 失敗: {e}")
    
    # 創建匯出器
    exporter = HFExporter(args.output)
    
    # 根據類型匯出
    if args.type == "pytorch":
        # 載入 PyTorch 模型
        model = torch.load(args.input, map_location='cpu')
        if isinstance(model, dict):
            # 如果是狀態字典，需要模型類別
            logger.error("PyTorch 狀態字典需要指定模型類別")
            return
        
        exporter.export_from_pytorch(model, tokenizer, config)
    
    elif args.type == "checkpoint":
        exporter.export_from_checkpoint(args.input, tokenizer=tokenizer)
    
    elif args.type == "onnx":
        exporter.export_from_onnx(args.input, config)
    
    # 創建模型卡片
    exporter.create_model_card(args.model_name, args.description)
    
    print(f"模型已匯出到: {args.output}")

if __name__ == "__main__":
    main()
