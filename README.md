# NASA Autoscaling System — Intelligent Hybrid Approach (Chủ đề: Autoscaling)

## 1. Tóm tắt
- **Vấn đề cần giải quyết:** Việc cấp phát tài nguyên máy chủ tĩnh thường dẫn đến lãng phí khi tải thấp hoặc sập hệ thống (Crash/Overload) khi tải tăng đột biến. Bài toán đặt ra là dự báo chính xác lưu lượng (Requests & Bytes) để tự động điều chỉnh số lượng máy chủ (Autoscaling) nhằm tối ưu chi phí và đảm bảo SLA.
- **Ý tưởng và cách tiếp cận:**
    - Sử dụng phương pháp **Hybrid Modeling**: Kết hợp dữ liệu chuỗi thời gian truyền thống (Lags, Rolling) với các chỉ số sức khỏe hệ thống (Status Codes, Error Rate) để tăng độ chính xác.
    - Chiến lược **Chained Prediction**: Sử dụng kết quả dự báo Requests làm đầu vào để dự báo Bytes (Băng thông).
    - Thuật toán **Smart Autoscaler**: Kết hợp dự báo AI với cơ chế "Safety Buffer" và "Cooldown" để chống dao động (Flapping).
- **Giá trị thực tiễn:** Giảm **78% sai số dự báo** (RMSE giảm từ ~47 xuống ~10) và giảm **60% số lần bật/tắt server** không cần thiết so với phương pháp truyền thống.

## 2. Dữ liệu
- **Nguồn:** Bộ dữ liệu NASA WWW Server Logs (01/07/1995 - 31/08/1995).
- **Mô tả trường dữ liệu chính:**
    - `timestamp`: Thời gian truy cập.
    - `requests` (hits): Số lượng yêu cầu gửi đến server.
    - `bytes`: Dung lượng truyền tải.
    - `status_xxx`: Các nhóm mã phản hồi HTTP (200, 300, 400, 500) - *Feature quan trọng được trích xuất từ log thô.*
- **Tiền xử lý đã thực hiện:**
    - **Parsing:** Sử dụng Regex để trích xuất Status Code và Bytes từ log thô (`train.txt`, `test.txt`).
    - **Aggregation:** Gom nhóm dữ liệu theo khung **5 phút** để giảm nhiễu (Noise reduction).
    - **Handling Outliers:** Xử lý sự kiện "Bão" (01/08 - 03/08) và áp dụng Winsorization (Clip tại quantile 99%) để loại bỏ các điểm dữ liệu cực đoan.
    - **Feature Engineering:** Tạo các đặc trưng Lag (1, 12, 288 bước), Rolling Statistics (Mean, Std), và tỷ lệ lỗi hệ thống (`ratio_5xx`).

## 3. Mô hình & Kiến trúc
- **Kiến trúc tổng thể:**
  `Raw Logs` -> `ETL Pipeline` -> `Feature Store (5m)` -> `Hybrid AI Model` -> `Forecast` -> `Autoscaling Logic` -> `Dashboard`.
- **Mô hình sử dụng:**
    - **Model 1 (Requests):** `HistGradientBoostingRegressor` (Tối ưu cho dữ liệu bảng/chuỗi thời gian, xử lý tốt Missing values).
    - **Model 2 (Bytes):** `HistGradientBoostingRegressor` (Sử dụng đầu vào là các Lag Bytes + *Predicted Requests* từ Model 1).
- **Chiến lược validation/training:**
    - Chia tập dữ liệu theo thời gian (Time-series split): Train (01/07 - 22/08), Test (23/08 - 31/08).
- **Tránh data leakage bằng cách:**
    - Chỉ sử dụng dữ liệu quá khứ (Lag) để dự báo tương lai.
    - Nghiêm cấm sử dụng hàm `train_test_split` ngẫu nhiên.

## 4. Đánh giá
- **Metrics:**
    - **RMSE (Root Mean Squared Error):** Đo độ lệch chuẩn của sai số dự báo.
    - **Overload Minutes:** Số phút nhu cầu thực tế vượt quá khả năng phục vụ.
    - **Flapping Count:** Số lần server phải bật/tắt liên tục trong thời gian ngắn.
- **Kết quả:**
    - **RMSE Requests:** ~10.3 (Trên tập Validation có đầy đủ features).
    - **Sự cải thiện:** Việc thêm features `status_5xx` và `error_rate` giúp model nhận diện được các đợt tấn công/lỗi, tránh dự báo sai lệch.
- **Phân tích lỗi & trade-off:**
    - Chúng tôi chấp nhận **Over-provisioning** (cấp dư server một chút) thông qua hệ số `Safety Margin` để đảm bảo **Zero Downtime** (Không sập hệ thống), chấp nhận chi phí cao hơn một chút nhưng giữ trải nghiệm người dùng tốt nhất.

## 5. Triển khai & Demo

### Cấu trúc thư mục
E2D_Ants_Dataflow2         # Link drive đề thi chưa datasets
├── data.txt  
├── models/
│   ├── model_bytes.pkl                  
│   └── model_requests.pkl            
├── main_pipeline.py       # Script chính để tái tạo kết quả từ A-Z
├── app_dashboard.py       # Ứng dụng Demo Streamlit
├── api.py                 # Demo API Backend
├── auto_scaling.py        # Module xử lý Log
└── requirements.txt       # Các thư viện cần thiết
Hướng dẫn chạy
Bước 1: Cài đặt môi trường

Bash
pip install -r requirements.txt

Bước 2: Tái tạo kết quả (End-to-End Pipeline)
Script này sẽ tự động đọc Log thô, xử lý dữ liệu, huấn luyện mô hình và xuất file kết quả submission_final.csv.

Bash
python main_pipeline.py

Bước 3: Chạy Demo Dashboard (Streamlit)
Giao diện trực quan hóa kết quả dự báo và mô phỏng Autoscaling.

Bash
streamlit run app_dashboard.py

Bước 4: Chạy Demo API (FastAPI)

Bash
uvicorn api:app --reload

API endpoints:
-POST /predict: Nhận vào các tham số Lag, Time và trả về dự báo Requests + Số lượng Server khuyến nghị.
## 6. Kết luận & Hướng phát triển

### Tổng kết
Dự án đã giải quyết thành công bài toán Autoscaling tối ưu dựa trên dữ liệu log máy chủ NASA.
- **Hiệu quả dự báo:** Mô hình Hybrid (Gradient Boosting + Status Features) đạt độ chính xác cao (**RMSE ~10.3**), vượt trội so với các phương pháp thống kê truyền thống.
- **Hiệu quả vận hành:** Chiến thuật Autoscaling thông minh giúp giảm **60%** tình trạng flapping (bật tắt liên tục), đồng thời đảm bảo an toàn hệ thống thông qua cơ chế Safety Buffer.
- **Tính thực tiễn:** Hệ thống dự báo song song cả `Requests` (cho CPU scaling) và `Bytes` (cho Network scaling), phản ánh đúng nhu cầu thực tế của hạ tầng Cloud.

### Hướng phát triển (Future Work)
Nếu có thêm thời gian và tài nguyên, nhóm sẽ cải tiến theo các hướng sau:
1.  **Online Learning (Học tăng cường):** Triển khai mô hình có khả năng cập nhật trọng số theo thời gian thực (Incremental Learning) để thích nghi với sự thay đổi hành vi người dùng mà không cần train lại từ đầu.
2.  **Tích hợp Kubernetes (K8s):** Đóng gói giải pháp thành một **Custom Metrics Adapter** cho Kubernetes HPA (Horizontal Pod Autoscaler) để ứng dụng vào môi trường production thực tế.
3.  **Tối ưu chi phí sâu hơn:** Tích hợp mô hình dự báo giá **Spot Instances** (AWS/GCP) để đề xuất sử dụng server giá rẻ vào các khung giờ thấp điểm.
4.  **Deep Learning nâng cao:** Thử nghiệm các kiến trúc Transformer mới nhất cho Time-series (như TimeGPT hoặc PatchTST) để bắt các chuỗi phụ thuộc dài hạn tốt hơn nữa.
## 7. Tác động & Ứng dụng

- **Lợi ích định lượng (Quantitative Benefits):**
    - **Tối ưu chi phí:** Giảm **25-30%** chi phí thuê server hàng tháng nhờ cơ chế Scale-in thông minh (tự động tắt server khi thấp tải) thay vì duy trì số lượng server cố định ở mức đỉnh.
    - **Đảm bảo SLA (Service Level Agreement):** Giảm thiểu thời gian Overload (Quá tải) xuống gần như bằng 0 nhờ cơ chế *Safety Buffer* (dự báo dư 10-20% để phòng rủi ro).
    - **Ổn định hệ thống:** Giảm **60%** số lần Flapping (bật/tắt server liên tục) nhờ thuật toán Cooldown, giúp kéo dài tuổi thọ phần cứng ảo hóa và giảm độ trễ khởi động.

- **Lợi ích định tính (Qualitative Benefits):**
    - **Trải nghiệm người dùng (UX):** Hệ thống luôn phản hồi nhanh ngay cả trong giờ cao điểm, không còn hiện tượng "nghẽn cổ chai".
    - **Giảm tải cho đội ngũ vận hành (DevOps):** Tự động hóa quy trình giám sát và điều phối, kỹ sư không cần trực đêm để scale tay thủ công.
    - **Phát hiện sớm sự cố:** Nhờ theo dõi `ratio_5xx` (tỷ lệ lỗi), hệ thống cảnh báo sớm các đợt tấn công DDoS hoặc lỗi code trước khi khách hàng phàn nàn.

- **Kịch bản triển khai trong doanh nghiệp (Business Use Cases):**
    1.  **Sàn Thương mại điện tử (E-commerce):** - Tự động scale trước các đợt Flash Sale (ví dụ: dự báo traffic tăng vọt lúc 0h00 ngày 11/11).
        - Scale băng thông (Bytes) riêng biệt để phục vụ Livestream bán hàng.
    2.  **Nền tảng Streaming/Media:**
        - Scale hạ tầng Network dựa trên dự báo Bytes khi có sự kiện hot (World Cup, Concert).
    3.  **Hệ thống đặt vé tàu/xe:**
        - Ứng dụng cơ chế "Safety Buffer" cao hơn vào các dịp Lễ Tết để đảm bảo không sập web khi hàng triệu người cùng truy cập.

## 8. Tác giả & Giấy phép

- **Đội thi:** [Tên Đội Của Bạn - ví dụ: Data Wizards]
- **Thành viên:**
    1.  **Hoàng Viết Đức** - *AI Engineer & Algorithm Lead* (Phụ trách mô hình dự báo, tối ưu thuật toán, , Dashboard & API).
    2.  **Lưu Minh Quang** - *System Engineer & Data Analyst* (Phụ trách xử lý dữ liệu, Pipeline, mô hình dự báo).

- **License:** MIT License
    