# MỤC LỤC ĐỒ ÁN TỐT NGHIỆP

**Đề tài:** Xây dựng Hệ thống Phát hiện Gian lận Thẻ Tín dụng ứng dụng Học máy và Triển khai trên Hạ tầng Microservices (MLOps)

---

## LỜI CẢM ƠN
## LỜI CAM ĐOAN

## DANH MỤC CÁC TỪ VIẾT TẮT

| Từ viết tắt | Ý nghĩa |
|---|---|
| ML | Machine Learning — Học máy |
| MLOps | Machine Learning Operations |
| SMOTE | Synthetic Minority Oversampling Technique |
| EDA | Exploratory Data Analysis |
| XAI | Explainable Artificial Intelligence |
| LLM | Large Language Model |
| CI/CD | Continuous Integration / Continuous Deployment |
| K8s | Kubernetes |
| NFS | Network File System |
| ELK | Elasticsearch, Filebeat, Kibana |
| API | Application Programming Interface |
| PCA | Principal Component Analysis |
| PVC | Persistent Volume Claim |
| TPE | Tree-structured Parzen Estimator |

## DANH MỤC HÌNH ẢNH

| STT | Tên hình | Vị trí |
|-----|----------|--------|
| Hình 1.1 | Sơ đồ kiến trúc tổng thể hệ thống | Mục 1.2.1 |
| Hình 1.2 | Quy trình vòng đời MLOps | Mục 1.2.2 |
| Hình 2.1 | Phân phối lớp dữ liệu (Class Distribution: 99.83% vs 0.17%) | Mục 2.3.1 |
| Hình 2.2 | Phân phối Amount theo Fraud/Legit | Mục 2.3.2 |
| Hình 2.3 | Correlation Heatmap giữa các đặc trưng PCA | Mục 2.3.2 |
| Hình 2.4 | Minh họa cơ chế hoạt động SMOTE | Mục 2.4.1 |
| Hình 2.5 | Confusion Matrix 4 model — Baseline (lưới 1×4) | Mục 2.4.2 |
| Hình 2.6 | Confusion Matrix 4 model — SMOTE (lưới 1×4) | Mục 2.4.2 |
| Hình 2.7 | Biểu đồ so sánh F1-Score: Baseline vs Under-sampling vs SMOTE | Mục 2.4.2 |
| Hình 2.8 | Feature Importance — Random Forest (Top 10 features) | Mục 2.5.4 |
| Hình 2.9 | Optuna Optimization History (50 trials) | Mục 2.5.4 |
| Hình 3.1 | Giao diện Webapp — Màn hình nhập giao dịch | Mục 3.3.1 |
| Hình 3.2 | Giao diện Webapp — Kết quả phân tích (Verdict + Risk Level) | Mục 3.3.1 |
| Hình 3.3 | Giao diện AI Chatbot — Giải thích lý do giao dịch bị từ chối | Mục 3.4.2 |
| Hình 3.4 | Sơ đồ luồng xử lý Dual-Provider (Gemini → Groq failover) | Mục 3.4.1 |
| Hình 4.1 | Giao diện MLflow Dashboard — Danh sách Experiment Runs | Mục 4.1.4 |
| Hình 4.2 | MLflow Parallel Coordinates — So sánh tham số tuning | Mục 4.1.4 |
| Hình 4.3 | MLflow Scatter Plot — F1-Score vs Hyperparameters | Mục 4.1.4 |
| Hình 4.4 | Data Drift Report — Tổng quan (Dataset Drift: YES/NO) | Mục 4.2.4 |
| Hình 4.5 | Data Drift Report — Phân phối Amount (Reference vs Current) | Mục 4.2.4 |
| Hình 4.6 | Data Drift Report — Phân phối V14 bị đảo ngược | Mục 4.2.4 |
| Hình 5.1 | Giao diện Proxmox VE — Danh sách VM (Master, Workers, pfSense, NFS) | Mục 5.2.1 |
| Hình 5.2 | Sơ đồ mạng: pfSense kết nối K8s Cluster và NFS Server | Mục 5.2.2 |
| Hình 5.3 | Giao diện Lens IDE — Quản lý Pod và Deployment | Mục 5.3.2 |
| Hình 5.4 | Giao diện Jenkins — Pipeline Build/Deploy thành công | Mục 5.4.2 |
| Hình 5.5 | Dashboard Grafana — Node Exporter (CPU, RAM, Network) | Mục 5.5.1 |
| Hình 5.6 | Dashboard Grafana — PostgreSQL Monitoring | Mục 5.5.1 |
| Hình 5.7 | Cảnh báo Telegram — NodeHighCPU Alert | Mục 5.5.2 |
| Hình 5.8 | Dashboard Kibana — Truy vấn Log tập trung | Mục 5.5.3 |
| Hình 6.1 | Kết quả kiểm thử HA: Pod tự khởi động lại sau khi bị kill | Mục 6.3.1 |
| Hình 6.2 | Kết quả giả lập tải cao CPU — Alert Telegram kích hoạt | Mục 6.3.3 |

## DANH MỤC BẢNG BIỂU

| STT | Tên bảng | Vị trí |
|-----|----------|--------|
| Bảng 2.1 | Thống kê mô tả tập dữ liệu Credit Card Fraud | Mục 2.3.1 |
| Bảng 2.2 | So sánh kích thước dữ liệu: Baseline vs Under-sampling vs SMOTE | Mục 2.4.1 |
| Bảng 2.3 | Kết quả 4 model × 3 chiến lược (12 kết quả) | Mục 2.4.2 |
| Bảng 2.4 | Bảng xếp hạng Best Model sau Tuning | Mục 2.5.4 |
| Bảng 5.1 | Danh sách VM trên Proxmox (IP, CPU, RAM) | Mục 5.2.1 |
| Bảng 5.2 | Các PVC sử dụng NFS (Namespace, Size, Mục đích) | Mục 5.3.3 |
| Bảng 5.3 | Danh sách Alert Rules (Tên, Điều kiện, Severity) | Mục 5.5.2 |

## MỤC LỤC

---

## MỞ ĐẦU
### 1. Lý do chọn đề tài
### 2. Mục tiêu nghiên cứu
### 3. Phạm vi nghiên cứu
### 4. Phương pháp nghiên cứu
### 5. Cấu trúc đồ án

---

## CHƯƠNG 1. TỔNG QUAN VỀ BÀI TOÁN VÀ KIẾN TRÚC HỆ THỐNG

### 1.1. Tổng quan về Gian lận Thẻ Tín dụng
#### 1.1.1. Định nghĩa và phân loại gian lận thẻ tín dụng
#### 1.1.2. Tác động kinh tế của gian lận trên toàn cầu
#### 1.1.3. Những thách thức cốt lõi trong phát hiện gian lận

### 1.2. Đề xuất Kiến trúc Tổng thể của Hệ thống
#### 1.2.1. Sơ đồ kiến trúc tổng quát
#### 1.2.2. Quy trình vòng đời MLOps

---

## CHƯƠNG 2. ỨNG DỤNG HỌC MÁY TRONG PHÁT HIỆN GIAN LẬN

### 2.1. Cơ sở lý thuyết các thuật toán Học máy
#### 2.1.1. Logistic Regression
#### 2.1.2. K-Nearest Neighbors (KNN)
#### 2.1.3. Support Vector Machine (SVM)
#### 2.1.4. Random Forest

### 2.2. Các chỉ số đánh giá mô hình phân loại
#### 2.2.1. Confusion Matrix, Precision, Recall, F1-Score
#### 2.2.2. AUC-ROC và Precision-Recall Curve

### 2.3. Phân tích Khám phá Dữ liệu (EDA) và Tiền xử lý
#### 2.3.1. Mô tả tập dữ liệu Credit Card Fraud Detection
#### 2.3.2. Trực quan hóa bằng Python (Matplotlib, Seaborn)
#### 2.3.3. Chuẩn hóa dữ liệu — RobustScaler

### 2.4. Giải pháp cho Dữ liệu Mất cân bằng (Imbalanced Data)
#### 2.4.1. Cơ sở lý thuyết về Under-sampling và SMOTE
#### 2.4.2. So sánh hiệu năng Baseline vs Under-sampling vs SMOTE

### 2.5. Tinh chỉnh Siêu tham số (Hyperparameter Tuning)
#### 2.5.1. Giới thiệu Optuna
#### 2.5.2. Stratified K-Fold Cross-validation
#### 2.5.3. Kỹ thuật Sub-sampling để tăng tốc huấn luyện
#### 2.5.4. Kết quả Tuning và Lựa chọn mô hình tốt nhất

---

## CHƯƠNG 3. XÂY DỰNG API VÀ TÍCH HỢP TRÍ TUỆ NHÂN TẠO GIẢI THÍCH ĐƯỢC (XAI)

### 3.1. Cơ sở lý thuyết về XAI và LLMs
#### 3.1.1. Vai trò của XAI trong tài chính
#### 3.1.2. Mô hình Ngôn ngữ Lớn (LLMs) và Prompt Engineering

### 3.2. Thiết kế Kiến trúc Backend
#### 3.2.1. Giới thiệu FastAPI
#### 3.2.2. Giới thiệu PostgreSQL

### 3.3. Xây dựng các Endpoint cốt lõi
#### 3.3.1. Endpoint `/analyze/` — Phân tích Giao dịch
#### 3.3.2. Endpoint `/history/` — Lịch sử Giao dịch

### 3.4. Tích hợp Hệ thống Trợ lý Ảo (Dual-Provider AI Chatbot)
#### 3.4.1. Kiến trúc Dual-Provider: Google Gemini và Groq Llama3
#### 3.4.2. Endpoint `/chat/` — Chatbot giải thích kết quả dự đoán

---

## CHƯƠNG 4. MLOPS — QUẢN LÝ THÍ NGHIỆM VÀ GIÁM SÁT MÔ HÌNH

### 4.1. MLflow — Experiment Tracking
#### 4.1.1. Giới thiệu MLflow
#### 4.1.2. Tích hợp MLflow vào Optuna Tuning
#### 4.1.3. Tích hợp MLflow vào Final Training
#### 4.1.4. Giao diện Dashboard MLflow

### 4.2. Evidently AI — Data Drift Monitoring
#### 4.2.1. Giới thiệu Evidently AI
#### 4.2.2. Thiết kế hệ thống giám sát drift
#### 4.2.3. Giả lập kịch bản Data Drift cho kiểm thử
#### 4.2.4. Giao diện Dashboard báo cáo Drift

---

## CHƯƠNG 5. TRIỂN KHAI VÀ QUẢN TRỊ VẬN HÀNH TRÊN HẠ TẦNG MICROSERVICES (DEVOPS)

### 5.1. Công nghệ Containerization và Orchestration
#### 5.1.1. Giới thiệu Docker
#### 5.1.2. Giới thiệu Kubernetes
#### 5.1.3. Giới thiệu Helm — Kubernetes Package Manager
*(Tại sao chọn Helm thay vì raw YAML manifests: quản lý ứng dụng phức tạp bằng Chart + Values, rollback phiên bản dễ dàng, tái sử dụng cấu hình, cộng đồng chart phong phú (Prometheus, Jenkins, ELK đều cài bằng Helm))*

### 5.2. Thiết lập Hạ tầng Máy chủ và Mạng (On-Premise)
#### 5.2.1. Giới thiệu Proxmox VE
#### 5.2.2. Giới thiệu pfSense & OpenVPN

### 5.3. Triển khai Cụm Kubernetes (K8s Cluster)
#### 5.3.1. Giới thiệu Kubespray
#### 5.3.2. Giới thiệu Lens IDE
#### 5.3.3. Cấp phát lưu trữ bền vững — NFS Server riêng biệt

### 5.4. Đóng gói Ứng dụng và CI/CD Pipeline
#### 5.4.1. Viết và Tối ưu Dockerfile
#### 5.4.2. Giới thiệu Jenkins
*(Tại sao chọn Jenkins thay vì GitHub Actions/GitLab CI: self-hosted trên K8s (không phụ thuộc cloud), plugin ecosystem khổng lồ, Jenkinsfile as code, tích hợp Docker + kubectl native)*

### 5.5. Hệ thống Giám sát và Cảnh báo
#### 5.5.1. Giới thiệu Prometheus và Grafana
#### 5.5.2. Cấu hình AlertManager → Telegram
#### 5.5.3. Giới thiệu Elastic Stack (ELK)

### 5.6. Áp dụng Infrastructure as Code (IaC) và Định hướng GitOps
#### 5.6.1. Khái niệm Infrastructure as Code (IaC)
*(Cách tiếp cận quản lý hạ tầng thông qua các tệp code (YAML, Groovy, Ansible) thay vì click chuột thủ công)*
#### 5.6.2. Các cấp độ IaC đã triển khai trong dự án
#### 5.6.3. Định hướng phát triển GitOps và Tự động hóa ảo hóa

---

## CHƯƠNG 6. KIỂM THỬ VÀ ĐÁNH GIÁ KẾT QUẢ

### 6.1. Đánh giá Mô hình Học máy
#### 6.1.1. Hiệu suất trên dữ liệu chưa biết (Unseen Test Data)
#### 6.1.2. So sánh Baseline vs Under-sampling vs SMOTE

### 6.2. Đánh giá Hệ thống XAI Chatbot
#### 6.2.1. Kiểm tra tính hợp lý của câu trả lời từ LLM

### 6.3. Đánh giá Hạ tầng và Vận hành
#### 6.3.1. Kiểm tra tính Sẵn sàng cao (High Availability) của K8s
#### 6.3.2. Kiểm tra độ trễ API và luồng CI/CD
#### 6.3.3. Kiểm thử kích hoạt Alert Telegram

### 6.4. Đánh giá MLOps
#### 6.4.1. Kiểm tra MLflow Dashboard
#### 6.4.2. Kiểm tra Data Drift Report

---

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
### 1. Kết luận
### 2. Những khó khăn và hạn chế
### 3. Hướng phát triển tương lai

---

## TÀI LIỆU THAM KHẢO

## PHỤ LỤC
- Phụ lục 1: Hướng dẫn cài đặt và cấu hình môi trường phát triển
- Phụ lục 2: Một số đoạn mã nguồn cốt lõi (Source Code Snippets)
