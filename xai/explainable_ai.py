#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 解釋型 AI (XAI) 模組
整合 SHAP, LIME, Attention 可視化
"""

import shap
import lime
import lime.lime_tabular
import lime.lime_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExplainableAI:
    """解釋型 AI 分析器"""
    
    def __init__(self, model=None, feature_names=None, class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.explanations = {}
        
    def setup_shap_explainer(self, X_train: np.ndarray, model_type: str = "tree"):
        """設置 SHAP 解釋器"""
        try:
            if model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, X_train)
            elif model_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, X_train)
            elif model_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, X_train)
            else:
                raise ValueError(f"不支援的模型類型: {model_type}")
            
            logger.info(f"✅ SHAP {model_type} 解釋器設置完成")
            return True
        except Exception as e:
            logger.error(f"❌ SHAP 解釋器設置失敗: {e}")
            return False
    
    def get_shap_explanations(self, X: np.ndarray, max_display: int = 10) -> Dict:
        """獲取 SHAP 解釋"""
        if self.explainer is None:
            raise ValueError("請先設置 SHAP 解釋器")
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            # 如果是多類別，取第一個類別
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # 計算特徵重要性
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # 排序特徵重要性
            if self.feature_names is not None:
                feature_importance_dict = dict(zip(self.feature_names, feature_importance))
                sorted_features = sorted(feature_importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True)[:max_display]
            else:
                sorted_features = [(f"Feature_{i}", importance) 
                                 for i, importance in enumerate(feature_importance)]
                sorted_features = sorted(sorted_features, key=lambda x: x[1], reverse=True)[:max_display]
            
            explanations = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': sorted_features,
                'explanation_type': 'shap'
            }
            
            self.explanations['shap'] = explanations
            return explanations
            
        except Exception as e:
            logger.error(f"❌ SHAP 解釋獲取失敗: {e}")
            return {}
    
    def setup_lime_explainer(self, X_train: np.ndarray, task_type: str = "regression"):
        """設置 LIME 解釋器"""
        try:
            if task_type == "regression":
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train, 
                    feature_names=self.feature_names,
                    mode='regression',
                    discretize_continuous=True
                )
            else:  # classification
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train, 
                    feature_names=self.feature_names,
                    mode='classification',
                    discretize_continuous=True
                )
            
            logger.info(f"✅ LIME {task_type} 解釋器設置完成")
            return True
        except Exception as e:
            logger.error(f"❌ LIME 解釋器設置失敗: {e}")
            return False
    
    def get_lime_explanations(self, X: np.ndarray, num_features: int = 10) -> Dict:
        """獲取 LIME 解釋"""
        if not hasattr(self, 'lime_explainer') or self.lime_explainer is None:
            raise ValueError("請先設置 LIME 解釋器")
        
        try:
            explanations = []
            feature_importance = []
            
            for i in range(len(X)):
                exp = self.lime_explainer.explain_instance(
                    X[i], 
                    self.model.predict, 
                    num_features=num_features
                )
                
                # 獲取特徵重要性
                feature_weights = exp.as_list()
                explanations.append(feature_weights)
                
                # 累積特徵重要性
                for feature, weight in feature_weights:
                    feature_importance.append((feature, abs(weight)))
            
            # 計算平均特徵重要性
            feature_importance_dict = {}
            for feature, weight in feature_importance:
                if feature in feature_importance_dict:
                    feature_importance_dict[feature] += weight
                else:
                    feature_importance_dict[feature] = weight
            
            # 正規化並排序
            total_weight = sum(feature_importance_dict.values())
            if total_weight > 0:
                feature_importance_dict = {k: v/total_weight for k, v in feature_importance_dict.items()}
            
            sorted_features = sorted(feature_importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            explanations_dict = {
                'explanations': explanations,
                'feature_importance': feature_importance_dict,
                'top_features': sorted_features,
                'explanation_type': 'lime'
            }
            
            self.explanations['lime'] = explanations_dict
            return explanations_dict
            
        except Exception as e:
            logger.error(f"❌ LIME 解釋獲取失敗: {e}")
            return {}
    
    def visualize_shap_summary(self, X: np.ndarray, max_display: int = 10, 
                              save_path: Optional[str] = None) -> go.Figure:
        """可視化 SHAP 摘要圖"""
        if 'shap' not in self.explanations:
            self.get_shap_explanations(X, max_display)
        
        shap_values = self.explanations['shap']['shap_values']
        top_features = self.explanations['shap']['top_features']
        
        # 創建 Plotly 圖表
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('SHAP 特徵重要性', 'SHAP 值分佈'),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 特徵重要性條形圖
        features, importances = zip(*top_features)
        fig.add_trace(
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                name='特徵重要性',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # SHAP 值散點圖（取前幾個樣本）
        sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
        for i, feature in enumerate(features[:5]):  # 只顯示前5個特徵
            feature_idx = self.feature_names.index(feature) if self.feature_names else i
            fig.add_trace(
                go.Scatter(
                    x=X[sample_indices, feature_idx],
                    y=shap_values[sample_indices, feature_idx],
                    mode='markers',
                    name=f'SHAP {feature}',
                    opacity=0.6
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="SHAP 解釋分析",
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"SHAP 可視化已保存到: {save_path}")
        
        return fig
    
    def visualize_lime_explanations(self, X: np.ndarray, sample_idx: int = 0, 
                                   num_features: int = 10, 
                                   save_path: Optional[str] = None) -> go.Figure:
        """可視化 LIME 解釋"""
        if 'lime' not in self.explanations:
            self.get_lime_explanations(X, num_features)
        
        explanations = self.explanations['lime']['explanations']
        if sample_idx >= len(explanations):
            sample_idx = 0
        
        sample_explanation = explanations[sample_idx]
        
        # 創建 LIME 解釋圖
        features, weights = zip(*sample_explanation)
        
        # 按權重絕對值排序
        sorted_data = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
        features, weights = zip(*sorted_data)
        
        # 創建顏色映射
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(weights),
                y=list(features),
                orientation='h',
                marker_color=colors,
                text=[f"{w:.3f}" for w in weights],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"LIME 解釋 - 樣本 {sample_idx}",
            xaxis_title="特徵權重",
            yaxis_title="特徵",
            height=max(400, len(features) * 30),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"LIME 可視化已保存到: {save_path}")
        
        return fig
    
    def get_attention_weights(self, model, input_data, layer_name: str = None) -> Dict:
        """獲取注意力權重（適用於 Transformer 模型）"""
        if not isinstance(model, nn.Module):
            raise ValueError("模型必須是 PyTorch nn.Module")
        
        model.eval()
        attention_weights = {}
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights['attention'] = module.attention_weights.detach().cpu().numpy()
        
        # 註冊鉤子
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if layer_name is None or layer_name in name:
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
        
        try:
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_data = torch.tensor(input_data, dtype=torch.float32)
                
                if torch.cuda.is_available():
                    input_data = input_data.cuda()
                    model = model.cuda()
                
                _ = model(input_data)
            
            # 移除鉤子
            for hook in hooks:
                hook.remove()
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"❌ 注意力權重獲取失敗: {e}")
            return {}
    
    def visualize_attention_heatmap(self, attention_weights: np.ndarray, 
                                   tokens: List[str] = None,
                                   save_path: Optional[str] = None) -> go.Figure:
        """可視化注意力熱力圖"""
        if len(attention_weights.shape) == 3:  # (batch, heads, seq_len, seq_len)
            attention_weights = attention_weights[0]  # 取第一個樣本
        
        if len(attention_weights.shape) == 3:  # (heads, seq_len, seq_len)
            # 平均所有注意力頭
            attention_weights = attention_weights.mean(axis=0)
        
        # 創建熱力圖
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=tokens if tokens else [f"Token_{i}" for i in range(attention_weights.shape[1])],
            y=tokens if tokens else [f"Token_{i}" for i in range(attention_weights.shape[0])],
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="注意力權重熱力圖",
            xaxis_title="Key",
            yaxis_title="Query",
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"注意力熱力圖已保存到: {save_path}")
        
        return fig
    
    def generate_explanation_report(self, X: np.ndarray, y_pred: np.ndarray = None,
                                  save_path: str = "explanation_report.html") -> str:
        """生成綜合解釋報告"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI 模型解釋報告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .feature-importance {{ display: flex; flex-wrap: wrap; }}
                .feature-item {{ margin: 5px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 AI 模型解釋報告</h1>
                <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>樣本數量: {len(X)}</p>
            </div>
        """
        
        # SHAP 解釋
        if 'shap' in self.explanations:
            report_html += """
            <div class="section">
                <h2>📊 SHAP 特徵重要性分析</h2>
            """
            
            top_features = self.explanations['shap']['top_features']
            for i, (feature, importance) in enumerate(top_features[:10]):
                report_html += f"""
                <div class="feature-item">
                    <strong>{i+1}. {feature}</strong>: {importance:.4f}
                </div>
                """
            
            report_html += "</div>"
        
        # LIME 解釋
        if 'lime' in self.explanations:
            report_html += """
            <div class="section">
                <h2>🍋 LIME 局部解釋分析</h2>
            """
            
            top_features = self.explanations['lime']['top_features']
            for i, (feature, importance) in enumerate(top_features[:10]):
                report_html += f"""
                <div class="feature-item">
                    <strong>{i+1}. {feature}</strong>: {importance:.4f}
                </div>
                """
            
            report_html += "</div>"
        
        # 模型預測摘要
        if y_pred is not None:
            report_html += f"""
            <div class="section">
                <h2>🎯 模型預測摘要</h2>
                <p><strong>平均預測值:</strong> {np.mean(y_pred):.4f}</p>
                <p><strong>預測標準差:</strong> {np.std(y_pred):.4f}</p>
                <p><strong>預測範圍:</strong> [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]</p>
            </div>
            """
        
        report_html += """
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"解釋報告已保存到: {save_path}")
        return save_path
    
    def compare_explanations(self, methods: List[str] = ['shap', 'lime']) -> Dict:
        """比較不同解釋方法"""
        comparison = {}
        
        for method in methods:
            if method in self.explanations:
                top_features = self.explanations[method]['top_features']
                comparison[method] = {
                    'top_5_features': [f[0] for f in top_features[:5]],
                    'feature_importance': dict(top_features[:10])
                }
        
        return comparison

# 使用示例
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # 創建示例數據
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    
    # 訓練模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 創建 XAI 分析器
    xai = ExplainableAI(model=model, feature_names=feature_names)
    
    # 設置解釋器
    xai.setup_shap_explainer(X, "tree")
    xai.setup_lime_explainer(X, "regression")
    
    # 獲取解釋
    print("🔍 獲取 SHAP 解釋...")
    shap_explanations = xai.get_shap_explanations(X[:100])
    print(f"✅ SHAP 前5個重要特徵: {[f[0] for f in shap_explanations['top_features'][:5]]}")
    
    print("\n🔍 獲取 LIME 解釋...")
    lime_explanations = xai.get_lime_explanations(X[:100])
    print(f"✅ LIME 前5個重要特徵: {[f[0] for f in lime_explanations['top_features'][:5]]}")
    
    # 生成報告
    y_pred = model.predict(X[:100])
    report_path = xai.generate_explanation_report(X[:100], y_pred)
    print(f"\n📊 解釋報告已生成: {report_path}")
    
    # 比較解釋方法
    comparison = xai.compare_explanations(['shap', 'lime'])
    print(f"\n🔄 解釋方法比較: {comparison}")
