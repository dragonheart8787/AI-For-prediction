#!/usr/bin/env python3
"""
啟動超級融合AGI內網Web服務器
"""

import socket
import subprocess
import sys
import os

def get_local_ip():
    """獲取本機內網IP地址"""
    try:
        # 連接到一個外部地址來獲取本機IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def check_dependencies():
    """檢查依賴是否安裝"""
    try:
        import flask
        import flask_socketio
        print("✅ 依賴檢查通過")
        return True
    except ImportError as e:
        print(f"❌ 缺少依賴: {e}")
        print("請先安裝依賴: pip install -r requirements_web.txt")
        return False

def main():
    print("🚀 超級融合AGI內網Web服務器啟動器")
    print("=" * 60)
    
    # 檢查依賴
    if not check_dependencies():
        return
    
    # 獲取本機IP
    local_ip = get_local_ip()
    
    print(f"🌐 本機內網IP地址: {local_ip}")
    print(f"🔗 內網訪問地址: http://{local_ip}:5002")
    print(f"🏠 本機訪問地址: http://localhost:5002")
    print("=" * 60)
    print("💡 其他設備可以通過內網IP地址訪問此服務器")
    print("💡 按 Ctrl+C 停止服務器")
    print("=" * 60)
    
    # 啟動Web服務器
    try:
        subprocess.run([sys.executable, "web_server.py"])
    except KeyboardInterrupt:
        print("\n🛑 服務器已停止")
    except Exception as e:
        print(f"❌ 啟動服務器失敗: {e}")

if __name__ == "__main__":
    main()
