#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例爬蟲插件
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class SamplePlugin:
    """示例爬蟲插件"""
    
    def __init__(self):
        self.name = "sample_plugin"
        self.version = "1.0.0"
        self.description = "示例爬蟲插件"
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("🚀 開始示例爬取...")
        
        try:
            # 模擬數據
            sample_data = [
                {
                    'id': 1,
                    'name': '示例數據1',
                    'value': random.randint(100, 1000),
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'id': 2,
                    'name': '示例數據2',
                    'value': random.randint(100, 1000),
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return {
                'success': True,
                'data_type': 'sample_data',
                'data': sample_data,
                'metadata': {
                    'total_records': len(sample_data),
                    'crawled_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 示例爬取失敗: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_type': 'sample_data'
            }
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['sample_data']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {}

# 測試代碼
if __name__ == "__main__":
    async def test_plugin():
        plugin = SamplePlugin()
        result = await plugin.crawl({'test': True})
        print(f"測試結果: {result}")
    
    asyncio.run(test_plugin())
