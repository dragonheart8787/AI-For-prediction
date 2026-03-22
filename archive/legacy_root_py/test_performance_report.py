#!/usr/bin/env python3
"""
性能測試與報告生成腳本
測試不同模型、批量大小和配置的性能表現
"""

import json
import time
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

def run_backtest(model: str, batch_size: int) -> Dict[str, Any]:
    """執行回測並返回結果"""
    cmd = [sys.executable, "demo_backtest_all.py", "--model", model, "--batch", str(batch_size)]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            # 嘗試從最後一行解析 JSON（處理警告訊息）
            stdout_lines = result.stdout.strip().split('\n')
            json_line = None
            for line in reversed(stdout_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if json_line:
                try:
                    output = json.loads(json_line)
                    output["execution_time"] = end_time - start_time
                    output["success"] = True
                    output["error"] = None
                except json.JSONDecodeError:
                    output = {
                        "success": False,
                        "error": f"JSON解析失敗: {json_line}",
                        "execution_time": end_time - start_time
                    }
            else:
                output = {
                    "success": False,
                    "error": "找不到有效的JSON輸出",
                    "execution_time": end_time - start_time
                }
        else:
            output = {
                "success": False,
                "error": result.stderr,
                "execution_time": end_time - start_time
            }
    except subprocess.TimeoutExpired:
        output = {
            "success": False,
            "error": "執行超時（300秒）",
            "execution_time": 300
        }
    except Exception as e:
        output = {
            "success": False,
            "error": str(e),
            "execution_time": 0
        }
    
    return output

def generate_performance_report(test_results: List[Dict[str, Any]]) -> str:
    """生成性能測試報告"""
    report = []
    report.append("# SuperFusionAGI 性能測試報告")
    report.append(f"測試時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 成功測試統計
    successful_tests = [r for r in test_results if r.get("success", False)]
    report.append(f"## 測試概況")
    report.append(f"- 總測試數：{len(test_results)}")
    report.append(f"- 成功數：{len(successful_tests)}")
    report.append(f"- 失敗數：{len(test_results) - len(successful_tests)}")
    report.append("")
    
    if successful_tests:
        report.append("## 成功測試詳情")
        report.append("")
        
        # 創建表格
        table_data = []
        for result in successful_tests:
            table_data.append({
                "模型": result.get("model", "N/A"),
                "批量": result.get("batch", "N/A"),
                "資料行數": result.get("rows", "N/A"),
                "ONNX": "是" if result.get("onnx", False) else "否",
                "執行時間(秒)": f"{result.get('execution_time', 0):.2f}",
                "吞吐量(行/秒)": f"{result.get('rows', 0) / result.get('execution_time', 1):.0f}" if result.get('execution_time', 0) > 0 else "N/A"
            })
        
        df = pd.DataFrame(table_data)
        # 使用簡單表格格式，避免依賴 tabulate
        report.append("| " + " | ".join(df.columns) + " |")
        report.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            report.append("| " + " | ".join(str(v) for v in row) + " |")
        report.append("")
        
        # 性能分析
        report.append("## 性能分析")
        report.append("")
        
        # 最快執行時間
        fastest = min(successful_tests, key=lambda x: x.get('execution_time', float('inf')))
        report.append(f"**最快執行**：{fastest.get('model')} (批量: {fastest.get('batch')}) - {fastest.get('execution_time', 0):.2f}秒")
        
        # 最高吞吐量
        highest_throughput = max(successful_tests, key=lambda x: x.get('rows', 0) / max(x.get('execution_time', 1), 0.001))
        throughput = highest_throughput.get('rows', 0) / max(highest_throughput.get('execution_time', 1), 0.001)
        report.append(f"**最高吞吐**：{highest_throughput.get('model')} (批量: {highest_throughput.get('batch')}) - {throughput:.0f} 行/秒")
        
        # ONNX vs 非ONNX
        onnx_tests = [r for r in successful_tests if r.get('onnx', False)]
        non_onnx_tests = [r for r in successful_tests if not r.get('onnx', False)]
        
        if onnx_tests and non_onnx_tests:
            avg_onnx_time = sum(r.get('execution_time', 0) for r in onnx_tests) / len(onnx_tests)
            avg_non_onnx_time = sum(r.get('execution_time', 0) for r in non_onnx_tests) / len(non_onnx_tests)
            speedup = avg_non_onnx_time / avg_onnx_time if avg_onnx_time > 0 else 1
            report.append(f"**ONNX加速比**：平均 {speedup:.2f}x")
        
        report.append("")
        
        # 批量大小影響
        report.append("## 批量大小影響分析")
        report.append("")
        
        # 按模型分組分析
        models = set(r.get('model') for r in successful_tests)
        for model in models:
            model_tests = [r for r in successful_tests if r.get('model') == model]
            if len(model_tests) > 1:
                report.append(f"### {model}")
                model_tests.sort(key=lambda x: x.get('batch', 0))
                for test in model_tests:
                    batch = test.get('batch', 0)
                    time_taken = test.get('execution_time', 0)
                    throughput = test.get('rows', 0) / max(time_taken, 0.001)
                    report.append(f"- 批量 {batch}: {time_taken:.2f}秒 ({throughput:.0f} 行/秒)")
                report.append("")
    
    # 失敗測試
    failed_tests = [r for r in test_results if not r.get("success", False)]
    if failed_tests:
        report.append("## 失敗測試")
        report.append("")
        for i, result in enumerate(failed_tests, 1):
            report.append(f"### 失敗測試 {i}")
            report.append(f"- 錯誤：{result.get('error', 'Unknown error')}")
            report.append("")
    
    return "\n".join(report)

def main():
    """主測試流程"""
    print("🚀 開始 SuperFusionAGI 性能測試...")
    
    # 測試配置
    test_configs = [
        # 自動模型選擇測試
        ("auto", 1024),
        ("auto", 2048),
        ("auto", 4096),
        ("auto", 8192),
        
        # 特定模型測試
        ("xgboost", 1024),
        ("xgboost", 4096),
        ("xgboost", 8192),
        
        ("lightgbm", 1024),
        ("lightgbm", 4096),
        
        ("linear", 1024),
        ("linear", 4096),
    ]
    
    results = []
    total_tests = len(test_configs)
    
    for i, (model, batch_size) in enumerate(test_configs, 1):
        print(f"📊 測試 {i}/{total_tests}: {model} (批量: {batch_size})")
        
        result = run_backtest(model, batch_size)
        result["test_config"] = {"model": model, "batch_size": batch_size}
        results.append(result)
        
        if result.get("success"):
            print(f"   ✅ 成功 - {result.get('execution_time', 0):.2f}秒")
        else:
            print(f"   ❌ 失敗 - {result.get('error', 'Unknown error')}")
    
    print("\n📝 生成測試報告...")
    report = generate_performance_report(results)
    
    # 保存報告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 報告已保存至：{report_file}")
    
    # 同時保存原始數據
    data_file = f"test_data_{timestamp}.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 原始數據已保存至：{data_file}")
    print("\n🎯 測試完成！請查看報告文件了解詳細結果。")

if __name__ == "__main__":
    main()
