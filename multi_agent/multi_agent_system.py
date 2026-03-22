#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤝 多代理系統 (Multi-Agent System)
不同模型代理負責不同領域，透過決策協調器整合結果
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """代理類型"""
    FINANCIAL = "financial"
    WEATHER = "weather"
    MEDICAL = "medical"
    ENERGY = "energy"
    SOCIAL = "social"
    GENERAL = "general"

class AgentStatus(Enum):
    """代理狀態"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class AgentMessage:
    """代理間消息"""
    sender: str
    receiver: str
    message_type: str
    data: Any
    timestamp: datetime
    priority: int = 1  # 1=低, 2=中, 3=高

@dataclass
class PredictionResult:
    """預測結果"""
    agent_id: str
    prediction: Any
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime

class BaseAgent:
    """基礎代理類"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 model: Any = None, confidence_threshold: float = 0.7):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.status = AgentStatus.IDLE
        self.message_queue = queue.Queue()
        self.prediction_history = []
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
        self.last_update = datetime.now()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """處理消息"""
        try:
            self.status = AgentStatus.PROCESSING
            start_time = time.time()
            
            # 根據消息類型處理
            if message.message_type == "prediction_request":
                result = await self.make_prediction(message.data)
                response = AgentMessage(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type="prediction_response",
                    data=result,
                    timestamp=datetime.now(),
                    priority=message.priority
                )
                
                # 更新性能指標
                processing_time = time.time() - start_time
                self._update_metrics(result.confidence, processing_time)
                
                self.status = AgentStatus.IDLE
                return response
            
            elif message.message_type == "status_request":
                response = AgentMessage(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type="status_response",
                    data={
                        'status': self.status.value,
                        'performance': self.performance_metrics,
                        'last_update': self.last_update.isoformat()
                    },
                    timestamp=datetime.now(),
                    priority=message.priority
                )
                return response
            
            else:
                logger.warning(f"未知消息類型: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"代理 {self.agent_id} 處理消息失敗: {e}")
            self.status = AgentStatus.ERROR
            return None
    
    async def make_prediction(self, data: Any) -> PredictionResult:
        """進行預測"""
        try:
            # 如果沒有模型，使用默認預測邏輯
            if self.model is None:
                # 根據代理類型生成不同的預測
                if self.agent_type == AgentType.FINANCIAL:
                    prediction = np.random.uniform(0.5, 1.0, 1)
                elif self.agent_type == AgentType.WEATHER:
                    prediction = np.random.uniform(0.3, 0.8, 1)
                elif self.agent_type == AgentType.MEDICAL:
                    prediction = np.random.uniform(0.7, 1.0, 1)
                elif self.agent_type == AgentType.ENERGY:
                    prediction = np.random.uniform(0.4, 0.9, 1)
                else:
                    prediction = np.random.uniform(0.0, 1.0, 1)
            else:
                # 使用實際模型進行預測
                if hasattr(self.model, 'predict'):
                    prediction = self.model.predict(data)
                elif hasattr(self.model, 'forward'):
                    with torch.no_grad():
                        if isinstance(data, np.ndarray):
                            data = torch.tensor(data, dtype=torch.float32)
                        prediction = self.model(data).numpy()
                else:
                    prediction = np.random.randn(1)
            
            # 計算置信度（這裡是示例）
            confidence = min(0.9, max(0.1, np.random.random()))
            
            # 生成推理過程
            reasoning = self._generate_reasoning(data, prediction, confidence)
            
            result = PredictionResult(
                agent_id=self.agent_id,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'agent_type': self.agent_type.value,
                    'model_type': type(self.model).__name__ if self.model else 'default',
                    'input_shape': data.shape if hasattr(data, 'shape') else 'unknown'
                },
                timestamp=datetime.now()
            )
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"代理 {self.agent_id} 預測失敗: {e}")
            raise
    
    def _generate_reasoning(self, data: Any, prediction: Any, confidence: float) -> str:
        """生成推理過程"""
        reasoning_templates = {
            AgentType.FINANCIAL: f"基於金融數據分析，預測值為 {prediction}，置信度 {confidence:.2f}",
            AgentType.WEATHER: f"基於氣象數據分析，預測值為 {prediction}，置信度 {confidence:.2f}",
            AgentType.MEDICAL: f"基於醫療數據分析，預測值為 {prediction}，置信度 {confidence:.2f}",
            AgentType.ENERGY: f"基於能源數據分析，預測值為 {prediction}，置信度 {confidence:.2f}",
            AgentType.SOCIAL: f"基於社交數據分析，預測值為 {prediction}，置信度 {confidence:.2f}",
            AgentType.GENERAL: f"基於通用數據分析，預測值為 {prediction}，置信度 {confidence:.2f}"
        }
        
        return reasoning_templates.get(self.agent_type, f"預測值為 {prediction}，置信度 {confidence:.2f}")
    
    def _update_metrics(self, confidence: float, processing_time: float):
        """更新性能指標"""
        self.performance_metrics['total_predictions'] += 1
        self.performance_metrics['successful_predictions'] += 1
        
        # 更新平均置信度
        total = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['average_confidence']
        self.performance_metrics['average_confidence'] = (current_avg * (total - 1) + confidence) / total
        
        # 更新平均處理時間
        current_avg_time = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (current_avg_time * (total - 1) + processing_time) / total
        
        self.last_update = datetime.now()

class DecisionCoordinator:
    """決策協調器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_router = {}
        self.consensus_strategies = {
            'weighted_average': self._weighted_average_consensus,
            'majority_vote': self._majority_vote_consensus,
            'confidence_weighted': self._confidence_weighted_consensus,
            'expert_opinion': self._expert_opinion_consensus
        }
        self.coordination_history = []
    
    def register_agent(self, agent: BaseAgent):
        """註冊代理"""
        self.agents[agent.agent_id] = agent
        logger.info(f"代理 {agent.agent_id} ({agent.agent_type.value}) 已註冊")
    
    def unregister_agent(self, agent_id: str):
        """註銷代理"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"代理 {agent_id} 已註銷")
    
    async def coordinate_prediction(self, data: Any, 
                                  target_agents: List[str] = None,
                                  consensus_strategy: str = "confidence_weighted") -> Dict:
        """協調預測"""
        if target_agents is None:
            target_agents = list(self.agents.keys())
        
        # 發送預測請求給目標代理
        prediction_requests = []
        for agent_id in target_agents:
            if agent_id in self.agents:
                message = AgentMessage(
                    sender="coordinator",
                    receiver=agent_id,
                    message_type="prediction_request",
                    data=data,
                    timestamp=datetime.now(),
                    priority=2
                )
                prediction_requests.append(message)
        
        # 並行處理預測請求
        tasks = []
        for message in prediction_requests:
            agent = self.agents[message.receiver]
            task = asyncio.create_task(agent.process_message(message))
            tasks.append(task)
        
        # 等待所有預測完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集成功的預測結果
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, AgentMessage) and result.message_type == "prediction_response":
                predictions.append(result.data)
            elif isinstance(result, Exception):
                logger.error(f"代理 {target_agents[i]} 預測失敗: {result}")
        
        if not predictions:
            raise ValueError("沒有代理成功完成預測")
        
        # 應用共識策略
        consensus_result = self.consensus_strategies[consensus_strategy](predictions)
        
        # 記錄協調歷史
        coordination_record = {
            'timestamp': datetime.now(),
            'target_agents': target_agents,
            'consensus_strategy': consensus_strategy,
            'individual_predictions': [p.__dict__ for p in predictions],
            'consensus_result': consensus_result
        }
        self.coordination_history.append(coordination_record)
        
        return consensus_result
    
    def _weighted_average_consensus(self, predictions: List[PredictionResult]) -> Dict:
        """加權平均共識"""
        weights = [p.confidence for p in predictions]
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(predictions)
            total_weight = len(predictions)
        
        weights = [w / total_weight for w in weights]
        
        # 計算加權平均預測
        weighted_prediction = sum(p.prediction * w for p, w in zip(predictions, weights))
        
        # 計算平均置信度
        avg_confidence = sum(p.confidence * w for p, w in zip(predictions, weights))
        
        # 合併推理過程
        reasoning = "加權平均共識: " + " | ".join([p.reasoning for p in predictions])
        
        return {
            'prediction': weighted_prediction,
            'confidence': avg_confidence,
            'reasoning': reasoning,
            'strategy': 'weighted_average',
            'agent_contributions': [
                {
                    'agent_id': p.agent_id,
                    'weight': w,
                    'prediction': p.prediction,
                    'confidence': p.confidence
                }
                for p, w in zip(predictions, weights)
            ]
        }
    
    def _majority_vote_consensus(self, predictions: List[PredictionResult]) -> Dict:
        """多數投票共識"""
        # 將預測值四捨五入到最近的整數進行投票
        votes = {}
        for p in predictions:
            vote_key = round(float(p.prediction))
            if vote_key not in votes:
                votes[vote_key] = []
            votes[vote_key].append(p)
        
        # 找到得票最多的預測
        majority_vote = max(votes.keys(), key=lambda k: len(votes[k]))
        majority_agents = votes[majority_vote]
        
        # 計算平均置信度
        avg_confidence = sum(p.confidence for p in majority_agents) / len(majority_agents)
        
        reasoning = f"多數投票共識: {len(majority_agents)}/{len(predictions)} 代理支持預測值 {majority_vote}"
        
        return {
            'prediction': majority_vote,
            'confidence': avg_confidence,
            'reasoning': reasoning,
            'strategy': 'majority_vote',
            'agent_contributions': [
                {
                    'agent_id': p.agent_id,
                    'prediction': p.prediction,
                    'confidence': p.confidence,
                    'vote': round(float(p.prediction))
                }
                for p in majority_agents
            ]
        }
    
    def _confidence_weighted_consensus(self, predictions: List[PredictionResult]) -> Dict:
        """置信度加權共識"""
        # 只考慮置信度超過閾值的預測
        high_confidence_predictions = [p for p in predictions if p.confidence > 0.5]
        
        if not high_confidence_predictions:
            high_confidence_predictions = predictions
        
        # 使用置信度作為權重
        weights = [p.confidence for p in high_confidence_predictions]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 計算加權平均
        weighted_prediction = sum(p.prediction * w for p, w in zip(high_confidence_predictions, weights))
        
        # 計算平均置信度
        avg_confidence = sum(p.confidence * w for p, w in zip(high_confidence_predictions, weights))
        
        reasoning = f"置信度加權共識: 基於 {len(high_confidence_predictions)} 個高置信度預測"
        
        return {
            'prediction': weighted_prediction,
            'confidence': avg_confidence,
            'reasoning': reasoning,
            'strategy': 'confidence_weighted',
            'agent_contributions': [
                {
                    'agent_id': p.agent_id,
                    'weight': w,
                    'prediction': p.prediction,
                    'confidence': p.confidence
                }
                for p, w in zip(high_confidence_predictions, weights)
            ]
        }
    
    def _expert_opinion_consensus(self, predictions: List[PredictionResult]) -> Dict:
        """專家意見共識"""
        # 選擇置信度最高的預測作為專家意見
        expert_prediction = max(predictions, key=lambda p: p.confidence)
        
        # 計算其他預測與專家預測的一致性
        consistency_scores = []
        for p in predictions:
            if p.agent_id != expert_prediction.agent_id:
                # 計算預測差異
                diff = abs(float(p.prediction) - float(expert_prediction.prediction))
                consistency = max(0, 1 - diff)  # 差異越小，一致性越高
                consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # 調整專家預測的置信度
        adjusted_confidence = expert_prediction.confidence * (0.7 + 0.3 * avg_consistency)
        
        reasoning = f"專家意見共識: 基於 {expert_prediction.agent_id} 的預測，一致性 {avg_consistency:.2f}"
        
        return {
            'prediction': expert_prediction.prediction,
            'confidence': adjusted_confidence,
            'reasoning': reasoning,
            'strategy': 'expert_opinion',
            'expert_agent': expert_prediction.agent_id,
            'consistency_score': avg_consistency,
            'agent_contributions': [
                {
                    'agent_id': p.agent_id,
                    'prediction': p.prediction,
                    'confidence': p.confidence,
                    'consistency': consistency_scores[i] if i < len(consistency_scores) else 0
                }
                for i, p in enumerate(predictions)
            ]
        }
    
    def get_system_status(self) -> Dict:
        """獲取系統狀態"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                'status': agent.status.value,
                'agent_type': agent.agent_type.value,
                'performance': agent.performance_metrics,
                'last_update': agent.last_update.isoformat()
            }
        
        return {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.IDLE]),
            'agent_statuses': agent_statuses,
            'coordination_history_count': len(self.coordination_history)
        }
    
    def get_coordination_analytics(self) -> Dict:
        """獲取協調分析"""
        if not self.coordination_history:
            return {'message': '沒有協調歷史'}
        
        # 統計不同策略的使用情況
        strategy_usage = {}
        for record in self.coordination_history:
            strategy = record['consensus_strategy']
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # 計算平均置信度
        avg_confidence = np.mean([
            record['consensus_result']['confidence'] 
            for record in self.coordination_history
        ])
        
        return {
            'total_coordinations': len(self.coordination_history),
            'strategy_usage': strategy_usage,
            'average_confidence': avg_confidence,
            'recent_coordinations': self.coordination_history[-5:]  # 最近5次協調
        }

class MultiAgentSystem:
    """多代理系統主類"""
    
    def __init__(self):
        self.coordinator = DecisionCoordinator()
        self.running = False
        self.message_loop_task = None
    
    async def start(self):
        """啟動系統"""
        self.running = True
        self.message_loop_task = asyncio.create_task(self._message_loop())
        logger.info("多代理系統已啟動")
    
    async def stop(self):
        """停止系統"""
        self.running = False
        if self.message_loop_task:
            self.message_loop_task.cancel()
        logger.info("多代理系統已停止")
    
    async def _message_loop(self):
        """消息循環"""
        while self.running:
            try:
                await asyncio.sleep(0.1)  # 避免過度佔用CPU
            except asyncio.CancelledError:
                break
    
    def add_agent(self, agent: BaseAgent):
        """添加代理"""
        self.coordinator.register_agent(agent)
    
    def remove_agent(self, agent_id: str):
        """移除代理"""
        self.coordinator.unregister_agent(agent_id)
    
    async def predict(self, data: Any, target_agents: List[str] = None, 
                     consensus_strategy: str = "confidence_weighted") -> Dict:
        """進行預測"""
        return await self.coordinator.coordinate_prediction(
            data, target_agents, consensus_strategy
        )
    
    def get_status(self) -> Dict:
        """獲取系統狀態"""
        return self.coordinator.get_system_status()
    
    def get_analytics(self) -> Dict:
        """獲取分析數據"""
        return self.coordinator.get_coordination_analytics()

# 使用示例
if __name__ == "__main__":
    async def main():
        # 創建多代理系統
        mas = MultiAgentSystem()
        await mas.start()
        
        # 創建示例代理
        financial_agent = BaseAgent("financial_001", AgentType.FINANCIAL)
        weather_agent = BaseAgent("weather_001", AgentType.WEATHER)
        medical_agent = BaseAgent("medical_001", AgentType.MEDICAL)
        
        # 添加代理到系統
        mas.add_agent(financial_agent)
        mas.add_agent(weather_agent)
        mas.add_agent(medical_agent)
        
        # 進行預測
        test_data = np.random.randn(10)
        
        print("🤝 多代理系統預測測試")
        print("=" * 50)
        
        # 測試不同的共識策略
        strategies = ["weighted_average", "majority_vote", "confidence_weighted", "expert_opinion"]
        
        for strategy in strategies:
            print(f"\n📊 使用 {strategy} 策略:")
            result = await mas.predict(test_data, consensus_strategy=strategy)
            print(f"預測結果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"推理: {result['reasoning']}")
        
        # 獲取系統狀態
        print(f"\n📈 系統狀態:")
        status = mas.get_status()
        print(f"總代理數: {status['total_agents']}")
        print(f"活躍代理數: {status['active_agents']}")
        
        # 獲取分析數據
        analytics = mas.get_analytics()
        print(f"\n📊 協調分析:")
        print(f"總協調次數: {analytics['total_coordinations']}")
        print(f"策略使用情況: {analytics['strategy_usage']}")
        
        await mas.stop()
    
    # 運行示例
    asyncio.run(main())
