# 圖表佔位符

此目錄包含論文所需的圖表檔案。由於實際圖表需要運行測試生成，這裡提供佔位符說明：

## 需要的圖表檔案

1. **memory_usage.png** - 記憶體使用趨勢圖
   - X軸：時間
   - Y軸：記憶體使用量 (MB)
   - 顯示不同批次大小下的記憶體使用模式

2. **throughput_scaling.png** - 吞吐量擴展性圖
   - X軸：批次大小
   - Y軸：吞吐量 (samples/s)
   - 比較原始模型與 ONNX 模型的吞吐量

3. **accuracy_comparison.png** - 精度比較圖
   - 柱狀圖顯示不同模型的 MAE 值
   - 比較原始模型與 ONNX 模型的精度差異

4. **performance_benchmark.png** - 性能基準測試圖
   - 線圖顯示不同模型在不同批次大小下的推理時間
   - 包含加速比標註

## 生成圖表的腳本

可以運行以下腳本來生成圖表：

```bash
# 生成性能基準測試圖表
python benchmark_onnx_cpu_batch.py --generate-plots

# 生成記憶體使用分析圖表
python test_performance_report.py --generate-plots
```

## 圖表規格

- 解析度：300 DPI
- 格式：PNG
- 尺寸：適合 A4 紙張列印
- 字體：Times New Roman，12pt
- 顏色：黑白列印友善
