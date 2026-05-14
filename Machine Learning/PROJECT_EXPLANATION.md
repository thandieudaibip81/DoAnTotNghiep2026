# Credit Card Fraud Detection — Machine Learning Pipeline

## 1. Tổng quan dự án

Dự án xây dựng hệ thống **phát hiện gian lận thẻ tín dụng** (Credit Card Fraud Detection) hoàn chỉnh, bao gồm:

1. **Pipeline Machine Learning:** 4 mô hình ML, so sánh 3 chiến lược xử lý dữ liệu mất cân bằng, tối ưu siêu tham số bằng Optuna.
4. **Ứng dụng Web (Webapp):** Giao diện Neo-Brutalism UI cho phép dự đoán gian lận thời gian thực.
5. **Phân tích AI:** Tích hợp 2 nhà cung cấp AI (Google Gemini + Groq) để phân tích chi tiết giao dịch.
6. **Cơ sở dữ liệu:** PostgreSQL lưu lịch sử giao dịch với hỗ trợ JSONB.
7. **CI/CD & Kubernetes:** Tự động hóa build/deploy với Jenkins, đóng gói Docker, chạy trên K8s cluster.

**Bài toán cốt lõi:** Dataset gồm 284.807 giao dịch, trong đó chỉ **492 giao dịch gian lận (0,173%)**. Đây là bài toán **phân loại nhị phân với dữ liệu cực kỳ mất cân bằng** — nếu model luôn dự đoán "Legit" thì accuracy đã đạt 99,83%, nhưng bỏ sót toàn bộ fraud.

---

## 2. Cấu trúc thư mục

```
Machine Learning/
├── run_pipeline.py          # Entry point CLI — chạy toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện Python
├── PROJECT_EXPLANATION.md   # Tài liệu giải thích dự án
│
├── src/                     # Source code chính (package Python)
│   ├── __init__.py          # Đánh dấu thư mục là Python package
│   ├── config.py            # Cấu hình toàn cục (đường dẫn, hằng số, siêu tham số)
│   ├── preprocessing.py     # Load dữ liệu, scaling, train/test split, sampling
│   ├── models.py            # Model factory — tạo các classifier
│   ├── trainer.py           # Huấn luyện model, lưu file .pkl
│   ├── tuner.py             # Tối ưu siêu tham số bằng Optuna
│   ├── evaluator.py         # Đánh giá model, vẽ biểu đồ, xuất CSV
│   └── db_utils.py          # Tích hợp PostgreSQL
│
├── webapp/                  # Ứng dụng web — giao diện dự đoán gian lận
│   ├── app.py               # FastAPI backend — API dự đoán, AI phân tích, DB
│   ├── .env                 # Biến môi trường (API keys, PostgreSQL)
│   ├── .env.example         # Mẫu biến môi trường
│   └── static/              # Frontend (HTML/CSS/JS)
│
├── notebooks/               # Jupyter Notebooks — phân tích & trình bày kết quả
│   ├── 01_eda.ipynb         # Khám phá dữ liệu
│   ├── 02_model_training.ipynb  # So sánh Baseline vs Under-sampling vs SMOTE
│   └── 03_hyperparameter_tuning.ipynb  # Kết quả tuning siêu tham số
│
├── data/                    # Dữ liệu đầu vào
│   └── creditcard.csv       # Dataset gốc (284.807 giao dịch)
│
├── models/                  # Model đã huấn luyện (file .pkl)
│   ├── robust_scaler.pkl              # Scaler dùng khi inference
│   ├── random_forest_smote.pkl        # ★ Production model — RF + SMOTE + Tuned
│   ├── logistic_regression_smote.pkl  # ★ Production model — LR + SMOTE + Tuned
│   ├── knn_smote.pkl                  # ★ Production model — KNN + SMOTE + Tuned
│   └── svm_smote.pkl                  # ★ Production model — SVM + SMOTE + Tuned
│
└── reports/                 # Output: biểu đồ, CSV, JSON, HTML
    ├── best_params_*.json             # Siêu tham số tốt nhất từ Optuna
    ├── model_comparison_smote.csv     # Bảng so sánh kết quả cuối cùng
    ├── cm_*.png / pr_*.png            # Confusion Matrix & Precision-Recall curves
    └── eda_*.png                      # Biểu đồ phân tích dữ liệu

Ops/                         # Thư mục DevOps & Infrastructure
├── docker/                  # Dockerfile cho Jenkins agent
├── helm/                    # Các Helm charts để deploy lên Kubernetes
│   ├── fraud-guard/         # Helm chart của Webapp + PostgreSQL
│   ├── elasticsearch-values.yaml
│   ├── kibana-values.yaml
│   ├── filebeat-values.yaml
│   ├── prometheus-stack-values.yaml
│   └── postgres-exporter-values.yaml
└── k8s/                     # Cấu hình Kubernetes tĩnh (NFS, Jenkins)
```

---

## 3. Chi tiết từng file trong `src/`

### 3.1. `config.py` — Cấu hình toàn cục

**Vai trò:** Tập trung mọi hằng số, đường dẫn, và cấu hình để các module khác import từ đây thay vì hard-code.

**Nội dung chính:**

| Hằng số           | Giá trị                                                  | Ý nghĩa                                                        |
| ----------------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| `RANDOM_STATE`    | `42`                                                     | Seed cho mọi thao tác ngẫu nhiên → đảm bảo **tái lập kết quả** |
| `TEST_SIZE`       | `0.2`                                                    | 20% dữ liệu dành cho test, 80% cho train                       |
| `TARGET_COL`      | `"Class"`                                                | Tên cột nhãn (0 = Legit, 1 = Fraud)                            |
| `SCALE_COLS`      | `["Amount", "Time"]`                                     | Chỉ 2 cột này cần scaling (V1–V28 đã PCA-scaled sẵn)           |
| `MODEL_NAMES`     | `["random_forest", "logistic_regression", "knn", "svm"]` | Danh sách 4 model                                              |
| `TUNER_N_TRIALS`  | `50`                                                     | Số trial cho mỗi model khi tuning Optuna                       |
| `TUNER_CV_FOLDS`  | `5`                                                      | Số fold cho Stratified K-Fold CV khi tuning                    |
| `SAMPLE_FRACTION` | `0.15`                                                   | Tỷ lệ subsample cho KNN/SVM khi tuning (tăng tốc)              |
| `SAMPLING_*`      | `"none"`, `"undersample"`, `"smote"`                     | Ba chiến lược sampling hợp lệ                                  |

---

### 3.2. `preprocessing.py` — Tiền xử lý dữ liệu

**Vai trò:** Xử lý dữ liệu từ CSV thô thành dạng sẵn sàng cho training.

**Các hàm chính:**

#### `load_data(path=None) → DataFrame`

- Đọc file `creditcard.csv`
- Kiểm tra cột `Class` tồn tại, không có giá trị null
- In thống kê: tổng số dòng, số fraud, tỷ lệ fraud

#### `scale_features(df, fit=True) → DataFrame`

- **Công nghệ:** `RobustScaler` (sklearn)
- **Tại sao RobustScaler?** Sử dụng **median** và **IQR** thay vì mean/std → **ít bị ảnh hưởng bởi outlier** (giao dịch fraud thường có Amount rất bất thường)
- Chỉ scale 2 cột `Amount` và `Time` (V1–V28 đã scaled bởi PCA từ nguồn dữ liệu)
- Khi `fit=True`: fit scaler mới → lưu vào `models/robust_scaler.pkl`
- Khi `fit=False`: load scaler đã lưu (dùng khi inference trong webapp)

#### `split_data(df) → (X_train, X_test, y_train, y_test)`

- **Công nghệ:** `train_test_split` với `stratify=y`
- **Stratified split** đảm bảo tỷ lệ fraud/legit được giữ nguyên trong cả train và test set
- Tỷ lệ: 80% train / 20% test
- `random_state=42` để kết quả tái lập

#### `apply_undersampling(X_train, y_train) → (X_res, y_res)`

- **Công nghệ:** `RandomUnderSampler` (imbalanced-learn)
- **Cách hoạt động:** Giảm số lượng class đa số (Legit) xuống bằng class thiểu số (Fraud)
- Kết quả: ~394 Legit + ~394 Fraud = ~788 mẫu (từ ~227.000 mẫu ban đầu)
- **Ưu điểm:** Nhanh, đơn giản
- **Nhược điểm:** Mất rất nhiều dữ liệu Legit → model có thể không học đủ pattern

#### `apply_smote(X_train, y_train) → (X_res, y_res)`

- **Công nghệ:** `SMOTE` — Synthetic Minority Over-sampling Technique (imbalanced-learn)
- **Cách hoạt động:** Tạo mẫu fraud **tổng hợp** (synthetic) bằng cách nội suy giữa các mẫu fraud thật và k-nearest neighbors của chúng
- Kết quả: ~227.000 Legit + ~227.000 Fraud (synthetic) = ~454.000 mẫu
- **Ưu điểm:** Không mất dữ liệu Legit, tạo thêm dữ liệu Fraud đa dạng
- **Nhược điểm:** Tốn thời gian hơn, mẫu synthetic có thể gây noise

#### `subsample_for_tuning(X, y, fraction=0.15) → (X_sub, y_sub)`

- Lấy 15% dữ liệu (stratified) cho tuning KNN/SVM vì 2 model này rất chậm trên toàn bộ dataset

---

### 3.3. `models.py` — Model Factory

**Vai trò:** Cung cấp **Factory Pattern** — caller chỉ cần gọi `get_model("logistic_regression")` mà không cần import trực tiếp các class sklearn.

**Bốn mô hình và tham số mặc định:**

#### Random Forest (`RandomForestClassifier`)

```python
{
    "n_estimators": 100,       # Số cây quyết định
    "max_depth": None,         # Độ sâu tối đa (không giới hạn)
    "min_samples_split": 2,    # Số mẫu tối thiểu để split node
    "class_weight": "balanced", # ← Tự điều chỉnh trọng số theo tỷ lệ class
    "random_state": 42,
    "n_jobs": -1,              # Dùng tất cả CPU cores
}
```

#### Logistic Regression (`LogisticRegression`)

```python
{
    "C": 1.0,                  # Inverse regularization strength
    "solver": "lbfgs",         # Thuật toán tối ưu hóa
    "max_iter": 1000,          # Số vòng lặp tối đa
    "class_weight": "balanced", # ← Tự điều chỉnh trọng số theo tỷ lệ class
    "random_state": 42,
}
```

#### K-Nearest Neighbors (`KNeighborsClassifier`)

```python
{
    "n_neighbors": 5,          # Số láng giềng
    "weights": "uniform",      # Tất cả láng giềng có trọng số bằng nhau
    "metric": "minkowski",     # Hàm khoảng cách
    "n_jobs": -1,
}
```

#### Support Vector Machine (`SVC`)

```python
{
    "C": 1.0,                  # Regularization parameter
    "kernel": "rbf",           # Radial Basis Function kernel
    "gamma": "scale",          # Hệ số kernel
    "class_weight": "balanced", # ← Tự điều chỉnh trọng số theo tỷ lệ class
    "probability": True,       # Bật predict_proba → cần cho ROC/PR curves
    "random_state": 42,
}
```

**: `class_weight="balanced"`**

Khi bật setting này, sklearn tự tính trọng số cho mỗi class theo công thức:

```
weight_class = n_samples / (n_classes × n_samples_class)
```

Với dataset này:

- Legit (class 0): weight ≈ 0.5
- Fraud (class 1): weight ≈ 289.4

→ Model **phạt nặng gấp ~289 lần** khi miss một giao dịch fraud so với khi miss một giao dịch legit. Điều này đẩy **Recall lên cao** (phát hiện được nhiều fraud hơn) nhưng **Precision giảm** (cảnh báo nhầm nhiều hơn).

> **Lưu ý:** KNN không hỗ trợ `class_weight` — model này dựa vào voting từ k láng giềng gần nhất.

**Các hàm chính:**

| Hàm                                  | Chức năng                                                       |
| ------------------------------------ | --------------------------------------------------------------- |
| `get_model(name, params=None)`       | Tạo 1 classifier từ tên + params (merge với defaults)           |
| `get_all_models(custom_params=None)` | Tạo cả 4 classifier cùng lúc                                    |
| `get_model_display_name(name)`       | Trả về tên hiển thị (VD: `"random_forest"` → `"Random Forest"`) |

---

### 3.4. `evaluator.py` — Đánh giá & Trực quan hóa

**Vai trò:** Tính toán metrics, vẽ biểu đồ, và xuất CSV để Power BI import.

#### `evaluate_model(model, X_test, y_test, model_name, sampling) → dict`

Tính 6 metrics cho mỗi model:

| Metric        | Ý nghĩa                                             | Quan trọng vì...                                                |
| ------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| **Accuracy**  | % dự đoán đúng tổng thể                             | Dễ hiểu nhưng **gây hiểu nhầm** với imbalanced data             |
| **Precision** | Trong số dự đoán Fraud, bao nhiêu thật sự là Fraud  | Quan trọng để đánh giá **tỷ lệ cảnh báo nhầm**                  |
| **Recall**    | Trong số Fraud thật, model phát hiện được bao nhiêu | **Metric quan trọng nhất** — bỏ sót fraud = thiệt hại tài chính |
| **F1-Score**  | Trung bình điều hòa của Precision & Recall          | Cân bằng giữa Precision và Recall                               |
| **AUC-ROC**   | Diện tích dưới đường cong ROC                       | Khả năng phân biệt Fraud/Legit ở mọi ngưỡng                     |
| **AUC-PR**    | Diện tích dưới đường Precision-Recall               | **Đáng tin hơn AUC-ROC** cho imbalanced data                    |

**Các hàm phụ:**

| Hàm                                     | Output                                                               |
| --------------------------------------- | -------------------------------------------------------------------- |
| `plot_confusion_matrix(...)`            | `reports/cm_{model}_{sampling}.png`                                  |
| `plot_precision_recall_curve(...)`      | `reports/pr_{model}_{sampling}.png`                                  |
| `export_metrics_csv(df, sampling)`      | `reports/model_comparison_{sampling}.csv`                            |
| `export_feature_importance(model, ...)` | `reports/feature_importance_{model}_{sampling}.csv` + biểu đồ top-20 |

---

### 3.5. `tuner.py` — Tối ưu siêu tham số

**Vai trò:** Tìm bộ siêu tham số tốt nhất cho mỗi model bằng Optuna.

**Công nghệ:** [Optuna](https://optuna.org/) — framework tối ưu hóa tham số tự động sử dụng thuật toán **TPE (Tree-structured Parzen Estimator)**, hiệu quả hơn Grid Search truyền thống vì nó **học từ các trial trước** để chọn tham số tiếp theo thông minh hơn.

**Mục tiêu tối ưu:** Tối đa hóa **F1-Score** qua Stratified 5-Fold Cross Validation.

> **Tại sao F1 mà không phải Recall?** F1 cân bằng giữa Precision và Recall — nếu chỉ tối ưu Recall, model có thể đạt Recall = 100% bằng cách cảnh báo tất cả giao dịch là Fraud (nhưng Precision = 0.17%).

**Các hàm chính:**

| Hàm                                                         | Chức năng                           |
| ----------------------------------------------------------- | ----------------------------------- |
| `tune_model(model_name, X_train, y_train, n_trials=50)`     | Tuning 1 model → trả về best params |
| `tune_all(X_train, y_train, model_names=None, n_trials=50)` | Tuning tất cả model                 |

---

### 3.6. `trainer.py` — Huấn luyện & Lưu Model

**Vai trò:** Huấn luyện model với tham số tốt nhất, lưu file `.pkl`, .

**Các hàm chính:**

| Hàm                                                  | Chức năng                                       |
| ---------------------------------------------------- | ----------------------------------------------- |
| `train_model(name, X_train, y_train, params)`        | Fit 1 model, nhận params từ tuner hoặc defaults |
| `train_all(sampling, model_names, use_tuned_params)` | Pipeline end-to-end cho tất cả model            |
| `save_model(model, name, sampling)`                  | Lưu model → `models/{name}_{sampling}.pkl`      |
| `load_model(name, sampling)`                         | Load model đã lưu                               |

---

### 3.7. `db_utils.py` — Tích hợp PostgreSQL

**Vai trò:** Đẩy kết quả vào PostgreSQL để Power BI kết nối trực tiếp.

---

## 4. Webapp — Ứng dụng Web dự đoán gian lận

### 4.1. Kiến trúc

```
Browser (Frontend)           FastAPI (Backend)            PostgreSQL
  ┌──────────────┐           ┌──────────────┐            ┌──────────┐
  │  index.html  │  ──API──▶ │   app.py     │  ──SQL──▶  │fraud_guard│
  │  style.css   │           │              │            │          │
  │  script.js   │  ◀─JSON── │  Models .pkl │  ◀─SQL──  │fraud_    │
  └──────────────┘           │  AI (Gemini/ │            │history   │
                             │      Groq)   │            └──────────┘
                             └──────────────┘
```

### 4.2. Backend — `webapp/app.py`

**Framework:** FastAPI + Uvicorn (ASGI server)

**API Endpoints:**

| Endpoint              | Method | Chức năng                                                    |
| --------------------- | ------ | ------------------------------------------------------------ |
| `/`                   | GET    | Serve trang `index.html`                                     |
| `/models/`            | GET    | Trả danh sách 4 model khả dụng cho dropdown                 |
| `/predict/`           | POST   | Nhận giao dịch, dự đoán fraud/legit, lưu DB                 |
| `/ai-analysis/`       | POST   | Gửi dữ liệu giao dịch cho AI phân tích (Gemini hoặc Groq)  |
| `/history/`           | GET    | Trả lịch sử giao dịch đã dự đoán                            |

### 4.3. Frontend — Giao diện Neo-Brutalism

**Thiết kế:** Neo-Brutalism UI — phong cách hiện đại với viền đậm, bóng đổ mạnh, màu sắc tối thiểu.

**Tính năng giao diện:**

1. **Form nhập giao dịch:** Nhập 30 features (Time, V1-V28, Amount)
2. **Chọn model:** Dropdown chọn 1 trong 4 model đã tuned
3. **Nút "Dự đoán":** Gọi API `/predict/`, hiển thị kết quả
4. **Nút "Phân tích bằng AI":** Gọi API `/ai-analysis/`, hiển thị phân tích chi tiết

### 4.4. Tích hợp AI — Dual Provider

Webapp hỗ trợ **2 nhà cung cấp AI** để phân tích chi tiết giao dịch:

| Provider        | Model              | Ưu điểm                              |
| --------------- | ------------------ | ------------------------------------- |
| **Google Gemini** | `gemini-1.5-flash` | Phân tích sâu, chất lượng cao        |
| **Groq**        | `llama-3.3-70b-versatile` | Tốc độ nhanh, không giới hạn request |

### 4.5. Cơ sở dữ liệu — PostgreSQL

**Database:** `fraud_guard` (PostgreSQL, encoding UTF-8)

---

## 5. File `run_pipeline.py` — CLI Entry Point

**Vai trò:** Điểm vào duy nhất để chạy toàn bộ pipeline qua command line.

**Các bước (steps) hỗ trợ:**

| Step         | Lệnh                                                      | Mô tả                                           |
| ------------ | --------------------------------------------------------- | ----------------------------------------------- |
| **baseline** | `python run_pipeline.py --step baseline`                  | Train 4 model trên dữ liệu thô (không sampling) |
| **tune**     |      | Tuning siêu tham số bằng Optuna    |
| **train**    |     | Train với params đã tune |
| **evaluate** | `python run_pipeline.py --step evaluate --sampling smote` | Evaluate model đã train trước đó                |
| **all**      | `python run_pipeline.py --step all --sampling smote`      | Chạy cả 3: baseline → tune → train              |

---

## 6. Notebooks

### 6.1. `01_eda.ipynb` — Khám phá dữ liệu (EDA)

- Phân bố class (`Class = 0` vs `Class = 1`) → chứng minh imbalanced
- Phân bố Amount và Time
- Ma trận tương quan (Correlation Heatmap)

### 6.2. `02_model_training.ipynb` — So sánh Baseline vs Sampling

Notebook chính, so sánh **4 model × 3 chiến lược sampling** = 12 thí nghiệm:

| Section                    | Nội dung                                                     |
| -------------------------- | ------------------------------------------------------------ |
| **Phần 1: Baseline**       | 4 model trên dữ liệu **thô** |
| **Phần 2: Under-sampling** | 4 model trên dữ liệu đã **giảm mẫu Legit**                   |
| **Phần 3: SMOTE**          | 4 model trên dữ liệu đã **tạo thêm mẫu Fraud tổng hợp**      |

### 6.3. `03_hyperparameter_tuning.ipynb` — Kết quả Tuning

- Load `best_params_*.json` cho mỗi model
- Biểu đồ so sánh: Recall, F1, AUC-ROC, AUC-PR

---

## 7. Chiến lược triển khai (Deployment Strategy)

**Tại sao chọn SMOTE + Tuned Params cho Production?**

1. **Dữ liệu mất cân bằng 99.83% vs 0.17%**: Baseline khiến model "lười" → predict tất cả là Legit.
2. **Recall là vua**: Bỏ sót 1 giao dịch gian lận nguy hiểm hơn nhiều so với báo nhầm. SMOTE giúp model phát hiện được các pattern Fraud mà dữ liệu thô bị bỏ qua.

---

## 8. Công nghệ & Thư viện sử dụng

### 8.1. ML Pipeline
- **scikit-learn**: Framework ML chính.
- **imbalanced-learn**: Xử lý imbalanced data (`SMOTE`, `RandomUnderSampler`).
- **Optuna**: Tối ưu siêu tham số.

### 8.2. Webapp & Backend
- **FastAPI**: Web framework hiện đại, hiệu năng cao.
- **google-generativeai / groq**: Tích hợp LLMs (Gemini, Llama) để phân tích giao dịch.
- **psycopg2-binary**: Kết nối cơ sở dữ liệu PostgreSQL.
- **python-dotenv**: Quản lý biến môi trường bảo mật.

---

## 9. Luồng dữ liệu tổng thể (Data Flow)

### 9.1. Pipeline Training & MLOps

```
creditcard.csv ──▶ Preprocessing ──▶ Sampling (SMOTE) ──▶ Tuning (Optuna)
                                                              │
       ┌──────────────────────────────┴──────────────────────────────┐
       ▼                                                             ▼
   MLflow Tracking                                           Evidently AI
 (Metrics & Artifacts)                                   (Data Drift Report)
```

### 9.2. Webapp Inference (Dự đoán trực tuyến)

```
User nhập giao dịch (browser)
       │
       ▼
  POST /predict/           ← FastAPI
       │
       ├── Scale Amount, Time    ← robust_scaler.pkl
       ├── Predict               ← {model}_smote.pkl
       ├── Lưu lịch sử          ← PostgreSQL (fraud_history)
       │
       ▼
  JSON response → Frontend hiển thị kết quả
       │
       ▼ (tùy chọn)
  POST /ai-analysis/       ← Gemini hoặc Groq
       │
       ▼
  AI phân tích chi tiết → Frontend hiển thị nhận xét
```

---

## 10. Cách chạy dự án

### 10.1. Cài đặt

```bash
cd "Machine Learning"
pip install -r requirements.txt
```

### 10.2. Chạy ML Pipeline

```bash
# Chạy toàn bộ: baseline → tune → train (SMOTE)
python run_pipeline.py --step all --sampling smote

# Chỉ chạy baseline
python run_pipeline.py --step baseline

# Chỉ tuning
python run_pipeline.py --step tune --sampling smote

# Chỉ train (dùng params đã tune)
python run_pipeline.py --step train --sampling smote
```

### 10.3. Chạy Webapp

```bash
cd "Machine Learning/webapp"

# Cấu hình biến môi trường
cp .env.example .env
# Chỉnh sửa .env: thêm API keys, PostgreSQL credentials

# Khởi động server
/usr/bin/python3 -m uvicorn app:app --reload --port 8000
```

Mở trình duyệt: **http://localhost:8000**

### 10.4. Xem dữ liệu PostgreSQL

```bash
# Kết nối psql
psql -U postgres -d fraud_guard

# Xem lịch sử
SELECT * FROM fraud_history;

# Hoặc dùng pgAdmin 4 (GUI)
# → Add Server → Host: localhost, Port: 5432, User: postgres, DB: fraud_guard
```

### 10.5. Chạy Notebooks

```bash
jupyter notebook notebooks/
```

Mở lần lượt: `01_eda.ipynb` → `02_model_training.ipynb` → `03_hyperparameter_tuning.ipynb`

---

## 11. Biến môi trường (`.env`)

| Biến              | Mô tả                         | Mặc định         |
| ----------------- | ------------------------------ | ----------------- |
| `GEMINI_API_KEY`  | API key Google Gemini          | (bắt buộc)        |
| `GROQ_API_KEY`    | API key Groq                   | (bắt buộc)        |
| `PG_DBNAME`       | Tên database PostgreSQL        | `fraud_guard`     |
| `PG_USER`         | Username PostgreSQL            | `postgres`        |
| `PG_PASSWORD`     | Mật khẩu PostgreSQL            | (trống)           |
| `PG_HOST`         | Host PostgreSQL                | `localhost`       |
| `PG_PORT`         | Port PostgreSQL                | `5432`            |

> **Bảo mật:** File `.env` chứa thông tin nhạy cảm → nằm trong `.gitignore`, KHÔNG push lên Git. File `.env.example` là mẫu tham khảo.

---



