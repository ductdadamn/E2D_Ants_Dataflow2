import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

# Import hàm xử lý log từ file của đồng đội
try:
    from auto_scaling import load_log, aggregate_per_minute, add_status_features
except ImportError:
    print("❌ Lỗi: Cần file 'auto_scaling.py' ở cùng thư mục để parse log.")
    exit()

# ==========================================
# 1. DATA PROCESSING (Gộp 2 model)
# ==========================================
def process_log_to_5m(log_path):
    """Đọc log -> 1 phút -> 5 phút (Request + Bytes + Status)"""
    if not Path(log_path).exists():
        print(f"⚠️ Không tìm thấy: {log_path}")
        return None
        
    df_raw = load_log(log_path)
    df_1m = aggregate_per_minute(df_raw)
    df_1m = add_status_features(df_raw, df_1m)
    
    # Gom 5 phút: Sum hết các chỉ số
    agg_rules = {
        'requests': 'sum', 'bytes': 'sum',
        'status_200': 'sum', 'status_500': 'sum'
    }
    # Tự động thêm các cột status khác nếu có
    for c in df_1m.columns:
        if 'status_' in c and c not in agg_rules:
            agg_rules[c] = 'sum'
            
    df_5m = df_1m.resample('5min').agg(agg_rules).fillna(0)
    return df_5m

def feature_engineering(df):
    df = df.copy()
    
    # --- Features cho REQUEST ---
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['ratio_5xx'] = df['status_500'] / (df['requests'] + 1e-9)
    df['is_crash'] = (df['ratio_5xx'] > 0.2).astype(int)
    
    # Lag Requests
    df['lag_requests_1'] = df['requests'].shift(1)
    df['lag_requests_288'] = df['requests'].shift(288)

    # --- Features cho BYTES ---
    # Lag Bytes
    df['lag_bytes_1'] = df['bytes'].shift(1)
    df['lag_bytes_288'] = df['bytes'].shift(288)
    
    return df

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def main():
    # PATH
    raw_dir = Path("F:/projects/DataflowSS2/")
    
    # A. XỬ LÝ DỮ LIỆU
    print("🚀 Đang xử lý log Train & Test...")
    train_df = process_log_to_5m(raw_dir / "train.txt")
    test_df = process_log_to_5m(raw_dir / "test.txt")
    
    if train_df is None or test_df is None: return

    # Nối lại để tạo lag (tránh đứt gãy giữa tháng 7 và 8)
    full = pd.concat([train_df, test_df]).sort_index()
    full = feature_engineering(full)
    
    # Tách Train/Test (Mốc 22/08)
    split_date = "1995-08-22 23:59:59-04:00"
    train_final = full[full.index <= split_date].dropna()
    test_final = full[full.index > split_date].copy()
    test_final = test_final.fillna(method='bfill') # Fix lag đầu chuỗi

    # B. TRAIN MODEL 1: REQUESTS (requests)
    print("🧠 Training Model 1: Requests...")
    feats_req = ['hour', 'dayofweek', 'lag_requests_1', 'lag_requests_288', 'ratio_5xx', 'is_crash']
    model_req = HistGradientBoostingRegressor(random_state=42)
    model_req.fit(train_final[feats_req], np.log1p(train_final['requests']))
    
    # Dự báo Request
    pred_requests_log = model_req.predict(test_final[feats_req])
    test_final['pred_requests'] = np.expm1(pred_requests_log)

    # C. TRAIN MODEL 2: BYTES (Băng thông)
    print("🧠 Training Model 2: Bytes...")
    # Mẹo: Dùng 'requests' thực tế để train, nhưng dùng 'pred_requests' để predict
    # Điều này giúp model Bytes hưởng lợi từ độ chính xác của model Requests
    
    feats_bytes = ['hour', 'dayofweek', 'lag_bytes_1', 'lag_bytes_288', 'ratio_5xx']
    
    # Thêm feature 'requests' vào training (requests thực tế)
    X_train_bytes = train_final[feats_bytes].copy()
    X_train_bytes['current_requests'] = train_final['requests'] 
    
    model_bytes = HistGradientBoostingRegressor(random_state=42)
    model_bytes.fit(X_train_bytes, np.log1p(train_final['bytes']))
    
    # Dự báo Bytes (Dùng 'pred_requests' làm đầu vào thay vì requests thực tế - để tránh leak)
    X_test_bytes = test_final[feats_bytes].copy()
    X_test_bytes['current_requests'] = test_final['pred_requests'] # Quan trọng!
    
    pred_bytes_log = model_bytes.predict(X_test_bytes)
    test_final['pred_bytes'] = np.expm1(pred_bytes_log)

    # D. XUẤT KẾT QUẢ
    out_cols = ['requests', 'pred_requests', 'bytes', 'pred_bytes']
    test_final[out_cols].to_csv("submission_final.csv")
    print("🎉 Xong! File kết quả: submission_final.csv")
    
    # Vẽ đồ thị kép
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot Request
    ax1.plot(test_final.index, test_final['requests'], color='black', alpha=0.3, label='Real requests')
    ax1.plot(test_final.index, test_final['pred_requests'], color='blue', label='Predicted requests')
    ax1.set_title("Dự báo Requests (Model 1)")
    ax1.legend()
    
    # Plot Bytes
    ax2.plot(test_final.index, test_final['bytes'], color='gray', alpha=0.3, label='Real Bytes')
    ax2.plot(test_final.index, test_final['pred_bytes'], color='green', label='Predicted Bytes')
    ax2.set_title("Dự báo Bytes (Model 2)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("forecast_full.png")
    print("📊 Đã lưu biểu đồ: forecast_full.png")

if __name__ == "__main__":
    main()