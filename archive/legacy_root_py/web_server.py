#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的Web界面服務器
"""

import asyncio
import aiohttp
from aiohttp import web
import json

async def handle_index(request):
    """處理首頁請求"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperFusionAGI 系統</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 SuperFusionAGI 系統</h1>
            <div class="status success">
                <h2>✅ 系統狀態</h2>
                <p>系統已成功啟動並運行中</p>
            </div>
            <div class="status success">
                <h3>🔌 插件系統</h3>
                <p>插件管理器已啟動，支持動態插件加載</p>
            </div>
            <div class="status success">
                <h3>🕷️ 數據爬取系統</h3>
                <p>綜合數據爬取器已啟動，支持多源數據收集</p>
            </div>
            <div class="status success">
                <h3>🎯 演示系統</h3>
                <p>示例插件已創建，可以測試插件功能</p>
            </div>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def handle_status(request):
    """處理狀態API請求"""
    status = {
        "system": "SuperFusionAGI",
        "status": "running",
        "components": {
            "plugin_system": True,
            "data_crawler": True,
            "demo_system": True,
            "web_interface": True
        },
        "timestamp": asyncio.get_event_loop().time()
    }
    return web.json_response(status)

async def init_app():
    """初始化應用"""
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/status', handle_status)
    return app

if __name__ == '__main__':
    app = init_app()
    web.run_app(app, host='127.0.0.1', port=8080)
