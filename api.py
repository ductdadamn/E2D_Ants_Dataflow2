from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Khởi tạo App
app = FastAPI(title="NASA Autoscaling API", version="1.0")

# Load Model (Nếu đã chạy train_model.py)
try:
    model_requests = joblib.load("models/model_requests.pkl")
    model_status = "Model Loaded ✅"
except:
    model_status = "Model Not Found ⚠️ (Running in Mock Mode)"
    model_requests = None

class InputData(BaseModel):
    lag_requests_1: float
    lag_requests_288: float
    hour: int
    dayofweek: int
    ratio_5xx: float

@app.get("/")
def health_check():
    return {"status": "ok", "model": model_status}

@app.post("/predict")
def predict_traffic(data: InputData):
    """
    API nhận feature đầu vào và trả về dự báo số lượng Request
    """
    if model_requests:
        # Tạo DataFrame từ input
        features = pd.DataFrame([data.dict()])
        # Predict Log
        pred_log = model_requests.predict(features)[0]
        # Bung Log ra số thực
        pred_value = np.expm1(pred_log)
    else:
        # Mock logic nếu chưa có model
        pred_value = data.lag_requests_1 * 1.05 # Giả sử tăng 5%
    
    # Logic Autoscaling đơn giản
    capacity = 100
    servers_needed = max(1, int(np.ceil(pred_value / capacity)))
    
    return {
        "predicted_requests": round(pred_value, 2),
        "servers_recommended": servers_needed,
        "action": "SCALE_OUT" if servers_needed > 1 else "MAINTAIN"
    }