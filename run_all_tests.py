#!/usr/bin/env python3
"""
一鍵測試執行腳本
安裝 pytest（若無），執行完整測試套件，生成報告
"""
import subprocess
import sys
import os
import json
from datetime import datetime


def install_pytest():
    """安裝 pytest 如果未安裝"""
    try:
        import pytest
        print("pytest 已安裝")
        return True
    except ImportError:
        print("安裝 pytest...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
            print("pytest 安裝成功")
            return True
        except subprocess.CalledProcessError:
            print("pytest 安裝失敗")
            return False


def create_reports_dir():
    """建立報告目錄"""
    os.makedirs("reports", exist_ok=True)
    print("建立 reports 目錄")


def run_tests():
    """執行測試套件"""
    print("開始執行測試套件...")
    
    # 測試命令
    test_commands = [
        # 基本測試
        ["pytest", "tests/test_unified_predict.py", "-v", "--tb=short"],
        ["pytest", "tests/test_data_connectors.py", "-v", "--tb=short"],
        ["pytest", "tests/test_onnx_integration.py", "-v", "--tb=short"],
        ["pytest", "tests/test_performance.py", "-v", "--tb=short", "-x"],  # 遇到失敗就停止
        ["pytest", "tests/test_cli_integration.py", "-v", "--tb=short", "-x"],
        
        # 完整測試套件（簡潔輸出）
        ["pytest", "tests/", "-q", "--tb=line"],
        
        # 生成 JUnit XML 報告
        ["pytest", "tests/", "-q", "--junitxml=reports/junit_results.xml", "--tb=no"],
    ]
    
    results = []
    
    for i, cmd in enumerate(test_commands):
        print(f"\n執行測試 {i+1}/{len(test_commands)}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 分鐘超時
            )
            
            test_result = {
                "command": " ".join(cmd),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            results.append(test_result)
            
            if result.returncode == 0:
                print("測試通過")
            else:
                print("測試失敗")
                if result.stderr:
                    print(f"錯誤: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print("測試超時")
            results.append({
                "command": " ".join(cmd),
                "return_code": -1,
                "stdout": "",
                "stderr": "Test timeout",
                "success": False
            })
        except Exception as e:
            print(f"測試執行錯誤: {e}")
            results.append({
                "command": " ".join(cmd),
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            })
    
    return results


def generate_test_report(results):
    """生成測試報告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 統計結果
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    # 生成文字報告
    report_content = f"""
# SuperFusionAGI 測試報告

**測試時間**: {timestamp}
**總測試數**: {total_tests}
**通過**: {passed_tests}
**失敗**: {failed_tests}
**成功率**: {passed_tests/total_tests*100:.1f}%

## 測試結果詳情

"""
    
    for i, result in enumerate(results, 1):
        status = "✅ 通過" if result["success"] else "❌ 失敗"
        report_content += f"""
### 測試 {i}: {status}
**命令**: `{result["command"]}`
**返回碼**: {result["return_code"]}

"""
        
        if result["stdout"]:
            report_content += f"**輸出**:\n```\n{result['stdout'][:500]}{'...' if len(result['stdout']) > 500 else ''}\n```\n\n"
            
        if result["stderr"]:
            report_content += f"**錯誤**:\n```\n{result['stderr'][:500]}{'...' if len(result['stderr']) > 500 else ''}\n```\n\n"
    
    # 寫入報告檔案
    with open("reports/test_results.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # 生成 JSON 報告
    json_report = {
        "timestamp": timestamp,
        "summary": {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests/total_tests*100
        },
        "results": results
    }
    
    with open("reports/test_results.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n測試報告已生成:")
    print(f"   - reports/test_results.txt")
    print(f"   - reports/test_results.json")
    print(f"   - reports/junit_results.xml")
    
    return json_report


def main():
    """主函數"""
    print("SuperFusionAGI 一鍵測試腳本")
    print("=" * 50)
    
    # 1. 安裝 pytest
    if not install_pytest():
        print("無法安裝 pytest，測試終止")
        return 1
    
    # 2. 建立報告目錄
    create_reports_dir()
    
    # 3. 執行測試
    results = run_tests()
    
    # 4. 生成報告
    json_report = generate_test_report(results)
    
    # 5. 輸出總結
    print("\n" + "=" * 50)
    print("測試總結:")
    print(f"   總測試數: {json_report['summary']['total']}")
    print(f"   通過: {json_report['summary']['passed']}")
    print(f"   失敗: {json_report['summary']['failed']}")
    print(f"   成功率: {json_report['summary']['success_rate']:.1f}%")
    
    if json_report['summary']['failed'] == 0:
        print("\n所有測試通過！")
        return 0
    else:
        print(f"\n有 {json_report['summary']['failed']} 個測試失敗")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)