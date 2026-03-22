"""
持久化AGI預測系統API接口
提供RESTful API來使用AGI預測功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import numpy as np
import json
import uvicorn
from datetime import datetime

from agi_persistent import PersistentConfig, PersistentAGISystem

# 創建FastAPI應用
app = FastAPI(
    title="持久化AGI預測系統API",
    description="提供AGI預測、訓練、雲端儲存等功能的RESTful API",
    version="1.0.0"
)

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局AGI系統實例
agi_system = None

# Pydantic模型
class PredictionRequest(BaseModel):
    model_name: str
    input_data: List[List[float]]
    description: Optional[str] = None

class PredictionResponse(BaseModel):
    model_name: str
    prediction: List[float]
    confidence: float
    timestamp: str
    success: bool
    message: str

class TrainingRequest(BaseModel):
    model_names: List[str] = ["financial_lstm", "weather_transformer"]
    training_epochs: Optional[int] = 100
    batch_size: Optional[int] = 32

class TrainingResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    training_time: Optional[str] = None

class SystemStatusResponse(BaseModel):
    system_running: bool
    total_models: int
    total_predictions: int
    continuous_learning_enabled: bool
    storage_path: str
    cloud_enabled: bool
    last_update: str
    model_performance: Dict[str, float]

class CloudOperationRequest(BaseModel):
    model_name: str
    operation: str  # "upload" or "download"

class CloudOperationResponse(BaseModel):
    success: bool
    message: str
    operation: str
    model_name: str

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化AGI系統"""
    global agi_system
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    print("🚀 AGI系統已初始化")

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時停止AGI系統"""
    global agi_system
    if agi_system:
        agi_system.stop_continuous_operation()
    print("🛑 AGI系統已停止")

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "持久化AGI預測系統API",
        "version": "1.0.0",
        "status": "運行中",
        "endpoints": {
            "預測": "/predict",
            "訓練": "/train",
            "狀態": "/status",
            "雲端操作": "/cloud",
            "持續運行": "/continuous"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """進行預測"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        # 檢查模型是否存在
        status = agi_system.get_system_status()
        if status.get('total_models', 0) == 0:
            # 如果沒有模型，先進行訓練
            await agi_system.train_all_models()
        
        # 轉換輸入資料
        input_data = np.array(request.input_data)
        
        # 進行預測
        result = await agi_system.make_prediction(request.model_name, input_data)
        
        if result:
            return PredictionResponse(
                model_name=result['model_name'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                timestamp=result['timestamp'],
                success=True,
                message="預測成功"
            )
        else:
            return PredictionResponse(
                model_name=request.model_name,
                prediction=[],
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                message="預測失敗"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """訓練模型"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        start_time = datetime.now()
        
        # 更新配置
        agi_system.config.training_epochs = request.training_epochs
        agi_system.config.training_batch_size = request.batch_size
        
        # 訓練模型
        results = await agi_system.train_all_models()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        if results:
            return TrainingResponse(
                success=True,
                message="模型訓練完成",
                results=results,
                training_time=f"{training_time:.2f}秒"
            )
        else:
            return TrainingResponse(
                success=False,
                message="模型訓練失敗",
                results=None,
                training_time=f"{training_time:.2f}秒"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"訓練失敗: {str(e)}")

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """獲取系統狀態"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        status = agi_system.get_system_status()
        
        return SystemStatusResponse(
            system_running=status.get('system_running', False),
            total_models=status.get('total_models', 0),
            total_predictions=status.get('total_predictions', 0),
            continuous_learning_enabled=status.get('continuous_learning_enabled', False),
            storage_path=status.get('storage_path', 'N/A'),
            cloud_enabled=status.get('cloud_enabled', False),
            last_update=status.get('last_update', datetime.now().isoformat()),
            model_performance=status.get('model_performance', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"狀態獲取失敗: {str(e)}")

@app.post("/cloud", response_model=CloudOperationResponse)
async def cloud_operation(request: CloudOperationRequest):
    """雲端操作"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        if request.operation == "upload":
            success = await agi_system.upload_to_cloud(request.model_name)
            message = "上傳成功" if success else "上傳失敗"
        elif request.operation == "download":
            success = await agi_system.download_from_cloud(request.model_name)
            message = "下載成功" if success else "下載失敗"
        else:
            raise HTTPException(status_code=400, detail="不支援的操作")
        
        return CloudOperationResponse(
            success=success,
            message=message,
            operation=request.operation,
            model_name=request.model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"雲端操作失敗: {str(e)}")

@app.post("/continuous/start")
async def start_continuous_operation():
    """啟動持續運行"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        # 在背景任務中啟動持續運行
        background_tasks = BackgroundTasks()
        background_tasks.add_task(agi_system.start_continuous_operation)
        
        return {
            "success": True,
            "message": "持續運行已啟動",
            "note": "系統將在背景持續運行，包括自動重新訓練和性能監控"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"持續運行啟動失敗: {str(e)}")

@app.post("/continuous/stop")
async def stop_continuous_operation():
    """停止持續運行"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        agi_system.stop_continuous_operation()
        
        return {
            "success": True,
            "message": "持續運行已停止"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"持續運行停止失敗: {str(e)}")

@app.get("/models")
async def list_models():
    """列出所有模型"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        # 從資料庫獲取模型列表
        cursor = agi_system.storage.cursor
        cursor.execute('''
            SELECT name, type, version, accuracy, created_at, updated_at
            FROM models
            ORDER BY updated_at DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            models.append({
                "name": row[0],
                "type": row[1],
                "version": row[2],
                "accuracy": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            })
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型列表獲取失敗: {str(e)}")

@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """獲取最近的預測記錄"""
    try:
        if not agi_system:
            raise HTTPException(status_code=500, detail="AGI系統未初始化")
        
        # 從資料庫獲取最近的預測
        cursor = agi_system.storage.cursor
        cursor.execute('''
            SELECT model_name, prediction_result, confidence, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                "model_name": row[0],
                "prediction": json.loads(row[1]),
                "confidence": row[2],
                "timestamp": row[3]
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "total_count": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測記錄獲取失敗: {str(e)}")

@app.get("/health")
async def health_check():
    """健康檢查"""
    try:
        if not agi_system:
            return {
                "status": "unhealthy",
                "message": "AGI系統未初始化",
                "timestamp": datetime.now().isoformat()
            }
        
        status = agi_system.get_system_status()
        
        return {
            "status": "healthy",
            "message": "系統運行正常",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "total_models": status.get('total_models', 0),
                "total_predictions": status.get('total_predictions', 0),
                "system_running": status.get('system_running', False)
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"系統異常: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # 運行API服務器
    uvicorn.run(
        "agi_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 