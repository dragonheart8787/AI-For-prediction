# 🎓 SuperFusionAGI 知識蒸餾系統指南

## 📋 概述

知識蒸餾（Knowledge Distillation）是一種模型壓縮技術，通過讓小模型（學生）學習大模型（老師）的知識來提升性能。在 SuperFusionAGI 系統中，我們實現了多種知識蒸餾技術，專門針對金融預測任務進行優化。

## 🚀 核心功能

### 1. 單步回歸知識蒸餾
- **用途**: 隔日/下一根K線的報酬或價格變化預測
- **老師輸出**: 連續值預測
- **學生損失**: MSE蒸餾 + 真實標籤的混合損失
- **公式**: `L = α * MSE(y_student, y_true) + (1-α) * MSE(y_student, y_teacher)`

### 2. 多地平線知識蒸餾
- **用途**: 同時預測多個時間地平線（1, 5, 10, 20天）
- **輸出**: 學生輸出向量 `[y_t+1, y_t+5, y_t+10, y_t+20]`
- **權重策略**: 指數衰減 `w_h ∝ e^(-h/τ)`
- **優勢**: 一次訓練獲得多個預測地平線

### 3. 機率式預測蒸餾
- **用途**: 不只預測均值，還預測不確定度
- **老師輸出**: 均值 + 方差 `(μ_T, σ_T²)`
- **學生輸出**: 均值 + 方差對數 `(μ_S, log σ_S²)`
- **損失**: KL散度對齊老師分佈

### 4. 分位數知識蒸餾
- **用途**: 預測分位數，對風控和倉位管理友好
- **老師輸出**: 多個分位數 `q_τ` (τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9})
- **學生損失**: Pinball損失對齊老師分位數
- **優勢**: 天然對尾部風險友好

### 5. 序列到序列蒸餾
- **用途**: LSTM/Transformer學生的序列預測
- **輸出蒸餾**: 每個時間步的預測對齊
- **表徵蒸餾**: 隱藏狀態對齊（Hint Loss）
- **優勢**: 保持序列建模能力

### 6. 多資產一體化學生
- **用途**: 跨資產類別（股票、ETF、加密貨幣）的統一模型
- **資產嵌入**: 資產類型嵌入向量
- **條件化預測**: 根據資產類型調整預測
- **優勢**: 提高泛化能力

### 7. 持續學習與不遺忘
- **用途**: 新數據微調時保持舊知識
- **Replay機制**: 新數據 + 舊樣本 + 舊軟標籤
- **正則化**: 對舊老師預測的約束
- **優勢**: 避免災難性遺忘

## 🏗️ 系統架構

```
knowledge_distillation/
├── __init__.py
├── probabilistic_student.py      # 機率式學生模型
├── teacher_ensemble.py           # 老師集成模型
├── sequence_kd.py               # 序列知識蒸餾
├── kd_trainer.py                # 統一訓練器
└── demo_knowledge_distillation.py # 演示腳本
```

## 📊 模型類型

### 機率式學生模型
```python
class ProbabilisticStudent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        # 共享特徵提取器
        self.backbone = nn.Sequential(...)
        # 均值頭
        self.mu_head = nn.Linear(hidden_dim//2, output_dim)
        # 方差頭（輸出log(σ²)）
        self.logvar_head = nn.Linear(hidden_dim//2, output_dim)
```

### 多地平線學生模型
```python
class MultiHorizonProbabilisticStudent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, horizons=[1,5,10,20]):
        # 每個地平線的預測頭
        self.mu_heads = nn.ModuleList([...])
        self.logvar_heads = nn.ModuleList([...])
        # 地平線權重（指數衰減）
        self.horizon_weights = self._compute_horizon_weights()
```

### 分位數學生模型
```python
class QuantileStudent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, quantiles=[0.1,0.25,0.5,0.75,0.9]):
        # 每個分位數的預測頭
        self.quantile_heads = nn.ModuleList([...])
```

## 🎯 老師集成策略

### 默認老師模型
- RandomForestRegressor
- GradientBoostingRegressor
- Ridge回歸
- Lasso回歸
- XGBoost
- LightGBM

### 集成方法
1. **均值集成**: 簡單平均所有老師預測
2. **加權集成**: 基於性能的權重平均
3. **Stacking**: 元模型學習最佳組合

### 軟標籤生成
```python
# 時間序列交叉驗證生成OOF預測
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # 訓練老師模型
    teacher.fit(X[train_idx], y[train_idx])
    # 生成OOF預測
    oof_predictions[val_idx] = teacher.predict(X[val_idx])
```

## 🔧 訓練配置

### 基本配置
```python
config = {
    'student_model_type': 'probabilistic',  # 學生模型類型
    'hidden_dim': 128,                      # 隱藏層維度
    'epochs': 100,                          # 訓練輪數
    'batch_size': 32,                       # 批次大小
    'learning_rate': 0.001,                 # 學習率
    'alpha': 0.5,                           # 硬標籤權重
    'cv_folds': 5,                          # 交叉驗證折數
    'ensemble_method': 'weighted',          # 集成方法
    'use_advanced_teachers': False          # 是否使用進階老師
}
```

### 進階配置
```python
config.update({
    'optimizer': 'adamw',                   # 優化器
    'weight_decay': 1e-5,                   # 權重衰減
    'use_scheduler': True,                  # 學習率調度
    'scheduler_type': 'step',               # 調度器類型
    'early_stopping_patience': 10,          # 早停耐心
    'temperature': 3.0,                     # 溫度參數
    'horizons': [1, 5, 10, 20],            # 預測地平線
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9] # 分位數水平
})
```

## 📈 使用示例

### 1. 基本使用
```python
from knowledge_distillation.kd_trainer import KnowledgeDistillationTrainer, create_default_config

# 創建配置
config = create_default_config()
config['student_model_type'] = 'probabilistic'
config['epochs'] = 100

# 創建訓練器
trainer = KnowledgeDistillationTrainer(config)

# 準備數據
data = trainer.prepare_data(X, y)

# 訓練老師集成
teacher_ensemble = trainer.train_teacher_ensemble(data)

# 生成老師預測
teacher_predictions = trainer.generate_teacher_predictions(data)

# 訓練學生模型
student_model = trainer.train_student_model(data, teacher_predictions)

# 評估模型
metrics = trainer.evaluate_model(data, teacher_predictions)
```

### 2. 多地平線預測
```python
config = create_default_config()
config.update({
    'student_model_type': 'multi_horizon',
    'horizons': [1, 5, 10, 20],
    'alpha': 0.5
})

trainer = KnowledgeDistillationTrainer(config)
# ... 訓練過程相同
```

### 3. 分位數預測
```python
config = create_default_config()
config.update({
    'student_model_type': 'quantile',
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
    'alpha': 0.5
})

trainer = KnowledgeDistillationTrainer(config)
# ... 訓練過程相同
```

### 4. 序列預測
```python
config = create_default_config()
config.update({
    'student_model_type': 'lstm',
    'hidden_dim': 128,
    'num_layers': 2,
    'alpha': 0.5
})

trainer = KnowledgeDistillationTrainer(config)
# ... 訓練過程相同
```

## 🎯 最佳實踐

### 1. 數據準備
- 使用時間序列交叉驗證避免洩漏
- 標準化特徵數據
- 確保老師和學生使用相同的數據分割

### 2. 老師模型選擇
- 使用多樣化的老師模型
- 確保老師模型有足夠的性能
- 考慮計算成本和性能的平衡

### 3. 超參數調優
- α參數：通常在0.3-0.7之間
- 學習率：從0.001開始調整
- 批次大小：根據數據量調整
- 早停：避免過擬合

### 4. 評估指標
- MSE/MAE：預測精度
- R²：解釋方差
- 分位數損失：分位數預測質量
- 校準圖：不確定度校準

### 5. 生產部署
- 模型版本管理
- 性能監控
- 增量學習
- 回退機制

## 🔍 故障排除

### 常見問題

1. **老師預測質量差**
   - 增加老師模型數量
   - 調整老師模型超參數
   - 使用更複雜的集成方法

2. **學生模型性能差**
   - 調整α參數
   - 增加學生模型容量
   - 調整學習率

3. **訓練不穩定**
   - 降低學習率
   - 增加批次大小
   - 使用梯度裁剪

4. **過擬合**
   - 增加正則化
   - 使用早停
   - 減少模型容量

### 調試技巧

1. **可視化訓練過程**
   ```python
   # 繪製訓練損失
   plt.plot(trainer.train_history['train_loss'])
   plt.plot(trainer.train_history['val_loss'])
   plt.legend(['Train', 'Validation'])
   plt.show()
   ```

2. **檢查老師預測分佈**
   ```python
   # 繪製老師預測分佈
   plt.hist(teacher_predictions['y_test_teacher'], bins=50, alpha=0.7)
   plt.hist(y_test, bins=50, alpha=0.7)
   plt.legend(['Teacher', 'True'])
   plt.show()
   ```

3. **分析預測誤差**
   ```python
   # 計算預測誤差
   errors = y_pred - y_true
   plt.scatter(y_true, errors)
   plt.xlabel('True Values')
   plt.ylabel('Prediction Errors')
   plt.show()
   ```

## 📊 性能基準

### 預期性能
- **MSE比率**: 0.8-1.2（學生/老師）
- **R²比率**: 0.9-1.1（學生/老師）
- **訓練時間**: 比老師模型快10-100倍
- **推理時間**: 比老師模型快5-50倍

### 優化目標
- 保持老師模型90%以上的性能
- 模型大小減少50-90%
- 推理速度提升5-50倍
- 內存使用減少50-90%

## 🚀 未來發展

### 計劃功能
1. **自適應知識蒸餾**: 動態調整蒸餾策略
2. **多任務蒸餾**: 同時學習多個任務
3. **聯邦蒸餾**: 分散式知識蒸餾
4. **神經架構搜索**: 自動設計學生架構

### 研究方向
1. **理論分析**: 蒸餾效果的理論保證
2. **新損失函數**: 更有效的蒸餾損失
3. **老師選擇**: 智能老師模型選擇
4. **持續學習**: 更好的不遺忘機制

## 📞 支援

如果您在使用知識蒸餾系統時遇到問題：

1. 查看日誌文件了解詳細錯誤信息
2. 檢查配置參數是否正確
3. 確認數據格式和大小
4. 參考演示腳本了解正確用法
5. 查看故障排除部分

---

**🎉 開始您的知識蒸餾之旅！** 🚀

知識蒸餾是提升模型效率和性能的強大工具，在 SuperFusionAGI 系統中為您提供專業級的金融預測能力。
