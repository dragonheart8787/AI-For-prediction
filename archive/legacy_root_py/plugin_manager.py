#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件管理器
自動發現、加載和管理新的爬取插件
"""

import os
import sys
import importlib
import inspect
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from abc import ABC, abstractmethod
import threading
import time
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class PluginDiscovery:
    """插件發現器"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        self.discovered_plugins: Dict[str, Dict] = {}
        self.last_discovery = None
        self.discovery_interval = 300  # 5分鐘檢查一次
        
    def discover_plugins(self) -> Dict[str, Dict]:
        """發現可用的插件"""
        current_time = time.time()
        
        # 檢查是否需要重新發現
        if (self.last_discovery and 
            current_time - self.last_discovery < self.discovery_interval):
            return self.discovered_plugins
        
        logger.info("🔍 開始發現新插件...")
        self.discovered_plugins.clear()
        
        # 掃描插件目錄
        if self.plugins_dir.exists():
            for plugin_file in self.plugins_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                    
                try:
                    plugin_info = self._analyze_plugin_file(plugin_file)
                    if plugin_info:
                        self.discovered_plugins[plugin_file.stem] = plugin_info
                        logger.info(f"✅ 發現插件: {plugin_file.stem}")
                except Exception as e:
                    logger.warning(f"⚠️ 分析插件文件失敗 {plugin_file}: {e}")
        
        # 掃描當前目錄中的插件類
        current_dir_plugins = self._discover_current_dir_plugins()
        self.discovered_plugins.update(current_dir_plugins)
        
        self.last_discovery = current_time
        logger.info(f"🔍 插件發現完成，共發現 {len(self.discovered_plugins)} 個插件")
        
        return self.discovered_plugins
    
    def _analyze_plugin_file(self, plugin_file: Path) -> Optional[Dict]:
        """分析插件文件"""
        try:
            # 讀取文件內容
            with open(plugin_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查是否包含插件類
            if "class" in content and "DataCrawlerPlugin" in content:
                return {
                    'file_path': str(plugin_file),
                    'type': 'file',
                    'last_modified': plugin_file.stat().st_mtime,
                    'size': plugin_file.stat().st_size,
                    'hash': hashlib.md5(content.encode()).hexdigest()[:8]
                }
        except Exception as e:
            logger.error(f"分析插件文件失敗 {plugin_file}: {e}")
        
        return None
    
    def _discover_current_dir_plugins(self) -> Dict[str, Dict]:
        """發現當前目錄中的插件類"""
        plugins = {}
        
        # 獲取當前模塊
        current_module = sys.modules[__name__]
        
        # 掃描當前模塊中的類
        for name, obj in inspect.getmembers(current_module):
            if (inspect.isclass(obj) and 
                issubclass(obj, DataCrawlerPlugin) and 
                obj != DataCrawlerPlugin):
                plugins[name] = {
                    'type': 'class',
                    'class_name': name,
                    'module': current_module.__name__,
                    'discovered_at': time.time()
                }
        
        return plugins
    
    def get_plugin_changes(self) -> Dict[str, List[str]]:
        """獲取插件變化"""
        changes = {'added': [], 'removed': [], 'modified': []}
        
        # 這裡可以實現更複雜的變化檢測邏輯
        # 比如檢查文件修改時間、內容哈希等
        
        return changes

class PluginLoader:
    """插件加載器"""
    
    def __init__(self):
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_instances: Dict[str, Any] = {}
        self.load_errors: Dict[str, str] = {}
        
    def load_plugin(self, plugin_name: str, plugin_info: Dict) -> Optional[Any]:
        """加載插件"""
        try:
            if plugin_info['type'] == 'file':
                return self._load_file_plugin(plugin_name, plugin_info)
            elif plugin_info['type'] == 'class':
                return self._load_class_plugin(plugin_name, plugin_info)
            else:
                logger.warning(f"未知的插件類型: {plugin_info['type']}")
                return None
                
        except Exception as e:
            error_msg = f"加載插件失敗: {e}"
            self.load_errors[plugin_name] = error_msg
            logger.error(f"❌ 加載插件 {plugin_name} 失敗: {e}")
            return None
    
    def _load_file_plugin(self, plugin_name: str, plugin_info: Dict) -> Optional[Any]:
        """從文件加載插件"""
        try:
            # 動態導入模塊
            spec = importlib.util.spec_from_file_location(
                plugin_name, 
                plugin_info['file_path']
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找插件類
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, DataCrawlerPlugin) and 
                    obj != DataCrawlerPlugin):
                    plugin_class = obj
                    break
            
            if plugin_class:
                instance = plugin_class()
                self.plugin_instances[plugin_name] = instance
                logger.info(f"✅ 成功加載文件插件: {plugin_name}")
                return instance
            else:
                logger.warning(f"⚠️ 在文件 {plugin_info['file_path']} 中未找到插件類")
                return None
                
        except Exception as e:
            logger.error(f"❌ 加載文件插件失敗 {plugin_name}: {e}")
            return None
    
    def _load_class_plugin(self, plugin_name: str, plugin_info: Dict) -> Optional[Any]:
        """從類加載插件"""
        try:
            # 獲取模塊
            module = sys.modules[plugin_info['module']]
            
            # 獲取類
            plugin_class = getattr(module, plugin_info['class_name'])
            
            if plugin_class:
                instance = plugin_class()
                self.plugin_instances[plugin_name] = instance
                logger.info(f"✅ 成功加載類插件: {plugin_name}")
                return instance
            else:
                logger.warning(f"⚠️ 未找到插件類: {plugin_info['class_name']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 加載類插件失敗 {plugin_name}: {e}")
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸載插件"""
        try:
            if plugin_name in self.plugin_instances:
                del self.plugin_instances[plugin_name]
                logger.info(f"✅ 成功卸載插件: {plugin_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ 卸載插件失敗 {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str, plugin_info: Dict) -> Optional[Any]:
        """重新加載插件"""
        logger.info(f"🔄 重新加載插件: {plugin_name}")
        
        # 先卸載
        self.unload_plugin(plugin_name)
        
        # 重新加載
        return self.load_plugin(plugin_name, plugin_info)

class PluginRegistry:
    """插件註冊表"""
    
    def __init__(self):
        self.registered_plugins: Dict[str, Dict] = {}
        self.plugin_metadata: Dict[str, Dict] = {}
        self.registry_file = "plugin_registry.json"
        self._load_registry()
        
    def register_plugin(self, plugin_name: str, plugin_instance: Any, 
                       metadata: Optional[Dict] = None) -> bool:
        """註冊插件"""
        try:
            # 驗證插件
            if not hasattr(plugin_instance, 'crawl'):
                logger.error(f"❌ 插件 {plugin_name} 缺少必要方法: crawl")
                return False
            
            if not hasattr(plugin_instance, 'get_supported_types'):
                logger.error(f"❌ 插件 {plugin_name} 缺少必要方法: get_supported_types")
                return False
            
            # 註冊插件
            self.registered_plugins[plugin_name] = {
                'instance': plugin_instance,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # 保存元數據
            if metadata:
                self.plugin_metadata[plugin_name] = metadata
            
            # 保存註冊表
            self._save_registry()
            
            logger.info(f"✅ 成功註冊插件: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 註冊插件失敗 {plugin_name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """註銷插件"""
        try:
            if plugin_name in self.registered_plugins:
                del self.registered_plugins[plugin_name]
                
                if plugin_name in self.plugin_metadata:
                    del self.plugin_metadata[plugin_name]
                
                self._save_registry()
                logger.info(f"✅ 成功註銷插件: {plugin_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ 註銷插件失敗 {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """獲取插件實例"""
        if plugin_name in self.registered_plugins:
            return self.registered_plugins[plugin_name]['instance']
        return None
    
    def get_all_plugins(self) -> Dict[str, Any]:
        """獲取所有註冊的插件"""
        return {name: info['instance'] for name, info in self.registered_plugins.items()}
    
    def get_plugin_status(self, plugin_name: str) -> Optional[str]:
        """獲取插件狀態"""
        if plugin_name in self.registered_plugins:
            return self.registered_plugins[plugin_name]['status']
        return None
    
    def _load_registry(self):
        """加載註冊表"""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.registered_plugins = data.get('plugins', {})
                    self.plugin_metadata = data.get('metadata', {})
                logger.info("✅ 成功加載插件註冊表")
        except Exception as e:
            logger.warning(f"⚠️ 加載插件註冊表失敗: {e}")
    
    def _save_registry(self):
        """保存註冊表"""
        try:
            # 準備保存數據（不包含實例對象）
            save_data = {
                'plugins': {},
                'metadata': self.plugin_metadata
            }
            
            for name, info in self.registered_plugins.items():
                save_data['plugins'][name] = {
                    'registered_at': info['registered_at'],
                    'status': info['status']
                }
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"❌ 保存插件註冊表失敗: {e}")

class PluginManager:
    """插件管理器主類"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.discovery = PluginDiscovery(plugins_dir)
        self.loader = PluginLoader()
        self.registry = PluginRegistry()
        self.auto_discovery = True
        self.discovery_thread = None
        self.running = False
        
    def start_auto_discovery(self):
        """啟動自動發現"""
        if self.auto_discovery and not self.running:
            self.running = True
            self.discovery_thread = threading.Thread(
                target=self._discovery_loop, 
                daemon=True
            )
            self.discovery_thread.start()
            logger.info("🚀 啟動自動插件發現")
    
    def stop_auto_discovery(self):
        """停止自動發現"""
        self.running = False
        if self.discovery_thread:
            self.discovery_thread.join(timeout=5)
        logger.info("🛑 停止自動插件發現")
    
    def _discovery_loop(self):
        """發現循環"""
        while self.running:
            try:
                # 發現新插件
                discovered = self.discovery.discover_plugins()
                
                # 檢查新插件
                for plugin_name, plugin_info in discovered.items():
                    if plugin_name not in self.registry.registered_plugins:
                        # 嘗試加載新插件
                        plugin_instance = self.loader.load_plugin(plugin_name, plugin_info)
                        if plugin_instance:
                            # 註冊新插件
                            self.registry.register_plugin(plugin_name, plugin_instance)
                
                # 檢查已註冊插件的變化
                changes = self.discovery.get_plugin_changes()
                for plugin_name in changes.get('modified', []):
                    if plugin_name in self.registry.registered_plugins:
                        # 重新加載插件
                        plugin_instance = self.loader.reload_plugin(plugin_name, plugin_info)
                        if plugin_instance:
                            # 更新註冊
                            self.registry.unregister_plugin(plugin_name)
                            self.registry.register_plugin(plugin_name, plugin_instance)
                
                # 等待下次檢查
                time.sleep(self.discovery.discovery_interval)
                
            except Exception as e:
                logger.error(f"❌ 自動發現循環錯誤: {e}")
                time.sleep(60)  # 錯誤時等待1分鐘
    
    def add_plugin_manually(self, plugin_name: str, plugin_instance: Any, 
                           metadata: Optional[Dict] = None) -> bool:
        """手動添加插件"""
        try:
            # 先加載到加載器
            self.loader.plugin_instances[plugin_name] = plugin_instance
            
            # 註冊到註冊表
            success = self.registry.register_plugin(plugin_name, plugin_instance, metadata)
            
            if success:
                logger.info(f"✅ 手動添加插件成功: {plugin_name}")
            return success
            
        except Exception as e:
            logger.error(f"❌ 手動添加插件失敗 {plugin_name}: {e}")
            return False
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """移除插件"""
        try:
            # 從加載器卸載
            self.loader.unload_plugin(plugin_name)
            
            # 從註冊表註銷
            success = self.registry.unregister_plugin(plugin_name)
            
            if success:
                logger.info(f"✅ 移除插件成功: {plugin_name}")
            return success
            
        except Exception as e:
            logger.error(f"❌ 移除插件失敗 {plugin_name}: {e}")
            return False
    
    def get_plugin_info(self) -> Dict[str, Dict]:
        """獲取插件信息"""
        info = {}
        
        for name, plugin_info in self.registry.registered_plugins.items():
            plugin_instance = plugin_info['instance']
            
            info[name] = {
                'status': plugin_info['status'],
                'registered_at': plugin_info['registered_at'],
                'supported_types': plugin_instance.get_supported_types(),
                'requirements': plugin_instance.get_requirements(),
                'type': type(plugin_instance).__name__
            }
            
            # 添加元數據
            if name in self.registry.plugin_metadata:
                info[name]['metadata'] = self.registry.plugin_metadata[name]
        
        return info
    
    def test_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """測試插件"""
        try:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin:
                return {'success': False, 'error': f'插件 {plugin_name} 未找到'}
            
            # 創建測試配置
            test_config = {'test': True, 'symbols': ['TEST']}
            
            # 執行測試爬取
            if asyncio.iscoroutinefunction(plugin.crawl):
                # 異步插件
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(plugin.crawl(test_config))
                finally:
                    loop.close()
            else:
                # 同步插件
                result = plugin.crawl(test_config)
            
            return {
                'success': True,
                'result': result,
                'tested_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False, 
                'error': str(e),
                'tested_at': datetime.now().isoformat()
            }
    
    def create_plugin_template(self, plugin_name: str, plugin_type: str = "custom") -> str:
        """創建插件模板"""
        template = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{plugin_name} 插件
{plugin_type} 類型數據爬取
"""

import asyncio
import logging
from typing import Dict, List, Any
from comprehensive_data_crawler import DataCrawlerPlugin

logger = logging.getLogger(__name__)

class {plugin_name.capitalize()}Plugin(DataCrawlerPlugin):
    """{plugin_name} 數據爬取插件"""
    
    def __init__(self):
        super().__init__()
        self.name = "{plugin_name}"
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info(f"🚀 開始爬取 {{self.name}} 數據...")
        
        try:
            # 在這裡實現您的爬取邏輯
            # 例如：爬取API、解析網頁、處理數據等
            
            # 示例數據結構
            sample_data = [
                {{
                    'id': 1,
                    'value': 'sample_value',
                    'timestamp': '2024-01-15T10:00:00Z'
                }}
            ]
            
            return {{
                'success': True,
                'data_type': '{plugin_name}',
                'data': sample_data,
                'metadata': {{
                    'total_records': len(sample_data),
                    'crawled_at': '2024-01-15T10:00:00Z'
                }}
            }}
            
        except Exception as e:
            logger.error(f"❌ 爬取 {{self.name}} 數據失敗: {{e}}")
            return {{
                'success': False,
                'error': str(e),
                'data_type': '{plugin_name}'
            }}
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['{plugin_name}']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {{
            'requests': '>=2.25.0',
            'aiohttp': '>=3.8.0'
        }}

# 如果直接運行此文件，進行測試
if __name__ == "__main__":
    async def test_plugin():
        plugin = {plugin_name.capitalize()}Plugin()
        result = await plugin.crawl({{'test': True}})
        print(f"測試結果: {{result}}")
    
    asyncio.run(test_plugin())
'''
        return template
    
    def save_plugin_template(self, plugin_name: str, plugin_type: str = "custom") -> str:
        """保存插件模板到文件"""
        template = self.create_plugin_template(plugin_name, plugin_type)
        
        # 確保插件目錄存在
        plugins_dir = Path(self.discovery.plugins_dir)
        plugins_dir.mkdir(exist_ok=True)
        
        # 保存模板文件
        template_file = plugins_dir / f"{plugin_name}_plugin.py"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template)
        
        logger.info(f"✅ 插件模板已保存到: {template_file}")
        return str(template_file)

# 為了兼容性，保留原有的基類
class DataCrawlerPlugin(ABC):
    """數據爬取插件基類"""
    
    @abstractmethod
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        pass

if __name__ == "__main__":
    # 測試插件管理器
    manager = PluginManager()
    
    print("🔧 插件管理器測試")
    print("=" * 40)
    
    # 創建示例插件模板
    template_file = manager.save_plugin_template("example", "test")
    print(f"✅ 創建了插件模板: {template_file}")
    
    # 啟動自動發現
    manager.start_auto_discovery()
    
    # 等待一下讓自動發現運行
    time.sleep(2)
    
    # 顯示插件信息
    plugin_info = manager.get_plugin_info()
    print(f"\n📋 當前註冊的插件: {len(plugin_info)}")
    for name, info in plugin_info.items():
        print(f"  - {name}: {info['status']}")
    
    # 停止自動發現
    manager.stop_auto_discovery()
    
    print("\n🎉 插件管理器測試完成！")
