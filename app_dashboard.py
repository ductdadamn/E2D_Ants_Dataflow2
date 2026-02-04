import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="NASA Autoscaling Monitor", layout="wide", page_icon="🚀")

# CSS để giao diện đẹp hơn
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center;}
    .stMetric {text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- LOAD DỮ LIỆU ---
@st.cache_data
def load_results():
    # Load file kết quả dự báo mà bạn đã chạy ra từ pipeline trước (submission_final.csv)
    # Nếu chưa có file thật, ta tạo dummy data để demo logic
    try:
        df = pd.read_csv("submission_final.csv", parse_dates=[0], index_col=0)
    except FileNotFoundError:
        # Dummy data generator (Dùng để test giao diện nếu chưa chạy model xong)
        dates = pd.date_range("1995-08-23", periods=1000, freq="5min")
        df = pd.DataFrame(index=dates)
        df['requests'] = np.random.poisson(150, 1000) + np.sin(np.arange(1000)/50)*50
        df['pred_requests'] = df['requests'] * np.random.normal(1, 0.1, 1000)
        df['bytes'] = df['requests'] * 15000
        df['pred_bytes'] = df['bytes'] * 0.95
    return df

df = load_results()

# --- SIDEBAR (THANH ĐIỀU KHIỂN) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=100)
st.sidebar.title("⚙️ Cấu hình Scaling")

capacity_per_server = st.sidebar.number_input("Sức chịu tải (Reqs/Server)", value=100, step=10)
safety_margin = st.sidebar.slider("Hệ số an toàn (Safety Buffer)", 0.0, 50.0, 10.0, help="Cộng thêm % dự phòng để tránh sập")
cooldown = st.sidebar.selectbox("Thời gian Cooldown (Phút)", [0, 5, 15, 30, 60], index=2)

st.sidebar.divider()
st.sidebar.info("Hệ thống sử dụng Hybrid Model (LightGBM + LSTM Logic) để dự báo tải trước 5 phút.")

# --- MAIN DASHBOARD ---
st.title("🚀 NASA Access Logs - Intelligent Autoscaling")

# 1. Tính toán Scaling Logic (Real-time Simulation)
df['safe_demand'] = df['pred_requests'] * (1 + safety_margin/100)
df['servers_needed'] = np.ceil(df['safe_demand'] / capacity_per_server)
df['servers_needed'] = df['servers_needed'].apply(lambda x: max(1, int(x)))

# Logic Cooldown (Giả lập)
servers_final = []
current_s = 1
last_change = -999
cooldown_steps = cooldown // 5

for i in range(len(df)):
    needed = df['servers_needed'].iloc[i]
    if needed > current_s: # Scale UP (Luôn ưu tiên)
        current_s = needed
        last_change = i
    elif needed < current_s: # Scale DOWN (Check cooldown)
        if i - last_change >= cooldown_steps:
            current_s = needed
            last_change = i
    servers_final.append(current_s)

df['servers_online'] = servers_final
df['system_capacity'] = df['servers_online'] * capacity_per_server

# 2. Metrics Tổng quan (Dòng trên cùng)
col1, col2, col3, col4 = st.columns(4)
last_idx = -1 # Lấy thời điểm mới nhất
with col1:
    st.metric("Lưu lượng Hiện tại", f"{int(df['requests'].iloc[last_idx])} reqs", delta=f"{int(df['requests'].iloc[last_idx] - df['requests'].iloc[last_idx-1])}")
with col2:
    st.metric("Server Đang chạy", f"{int(df['servers_online'].iloc[last_idx])}", delta_color="off")
with col3:
    load_percent = (df['requests'].iloc[last_idx] / df['system_capacity'].iloc[last_idx]) * 100
    st.metric("Tải hệ thống (%)", f"{load_percent:.1f}%", delta=None)
with col4:
    cost = df['servers_online'].sum() * 0.5 # Giả sử $0.5/server/5min
    st.metric("Ước tính Chi phí", f"${cost:,.0f}")

# 3. Biểu đồ Chính (Request & Scaling)
st.subheader("📈 Giám sát Tải & Scaling")
tab1, tab2 = st.tabs(["Requests (CPU Scaling)", "Bytes (Bandwidth Scaling)"])

with tab1:
    fig = go.Figure()
    # Nhu cầu thực
    fig.add_trace(go.Scatter(x=df.index, y=df['requests'], name='Thực tế', line=dict(color='gray', width=1), opacity=0.6))
    # Dự báo AI
    fig.add_trace(go.Scatter(x=df.index, y=df['pred_requests'], name='AI Dự báo', line=dict(color='#3366CC', width=2)))
    # Khả năng phục vụ
    fig.add_trace(go.Scatter(x=df.index, y=df['system_capacity'], name='Năng lực Server', 
                             line=dict(color='#2ecc71', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.1)'))
    
    # Highlight Overload
    overload = df[df['requests'] > df['system_capacity']]
    if not overload.empty:
        fig.add_trace(go.Scatter(x=overload.index, y=overload['requests'], mode='markers', name='QUÁ TẢI (Crash)', marker=dict(color='red', size=8)))

    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("Dự báo băng thông mạng để tối ưu hóa đường truyền.")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['bytes'], name='Bytes Thực tế', line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['pred_bytes'], name='Dự báo Bytes', line=dict(color='purple', dash='dot')))
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# 4. Phân tích Chi tiết
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("📊 Thống kê Hiệu quả")
    overload_count = len(overload)
    total_reqs = len(df)
    uptime = 100 - (overload_count/total_reqs * 100)
    st.write(f"**Uptime (SLA):** {uptime:.2f}%")
    st.write(f"**Số lần Flapping (Bật/Tắt):** {np.sum(np.abs(np.diff(df['servers_online']))) } lần")
    st.progress(uptime/100)
    
    if uptime < 99.9:
        st.error("⚠️ Cần tăng hệ số an toàn!")
    else:
        st.success("✅ Hệ thống hoạt động ổn định.")

with c2:
    st.subheader("📋 Log Hoạt động (Dữ liệu 5 phút cuối)")
    st.dataframe(df[['requests', 'pred_requests', 'servers_online', 'system_capacity']].tail(5), use_container_width=True)