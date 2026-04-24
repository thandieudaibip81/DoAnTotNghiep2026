# Đồ Án Tốt Nghiệp 2026 — Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng

**Sinh viên:** Cao Quang Tiến — 223630714 — KHMT-K63

---

## 1. Tổng quan dự án

Dự án xây dựng hệ thống **phát hiện gian lận thẻ tín dụng** (Credit Card Fraud Detection) hoàn chỉnh, bao gồm:

1. **Pipeline Machine Learning:** 4 mô hình ML, so sánh 3 chiến lược xử lý dữ liệu mất cân bằng, tối ưu siêu tham số bằng Optuna
2. **Ứng dụng Web (Webapp):** Giao diện Neo-Brutalism UI cho phép nhập giao dịch, chọn model, dự đoán gian lận
3. **Phân tích AI:** Tích hợp 2 nhà cung cấp AI (Google Gemini + Groq) để phân tích chi tiết từng giao dịch
4. **Cơ sở dữ liệu:** PostgreSQL lưu lịch sử giao dịch với hỗ trợ JSONB

**Bài toán cốt lõi:** Dataset gồm 284.807 giao dịch, trong đó chỉ **492 giao dịch gian lận (0,173%)**. Đây là bài toán **phân loại nhị phân với dữ liệu cực kỳ mất cân bằng** — nếu model luôn dự đoán "Legit" thì accuracy đã đạt 99,83%, nhưng bỏ sót toàn bộ fraud.

---

## 2. Cấu trúc thư mục

```
Machine Learning/
├── run_pipeline.py          # Entry point CLI — chạy toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện Python
├── PROJECT_EXPLANATION.md   # Tài liệu giải thích dự án chi tiết
│
├── src/                     # Source code chính (package Python)
│   ├── __init__.py          # Đánh dấu thư mục là Python package
│   ├── config.py            # Cấu hình toàn cục (đường dẫn, hằng số, siêu tham số)
│   ├── preprocessing.py     # Load dữ liệu, scaling, train/test split, sampling
│   ├── models.py            # Model factory — tạo 4 classifier với tham số mặc định
│   ├── trainer.py           # Huấn luyện model, lưu/tải file .pkl
│   ├── tuner.py             # Tối ưu siêu tham số bằng Optuna
│   ├── evaluator.py         # Đánh giá model, vẽ biểu đồ, xuất CSV
│   └── db_utils.py          # Tích hợp PostgreSQL cho Power BI (tùy chọn)
│
├── webapp/                  # Ứng dụng web — giao diện dự đoán gian lận
│   ├── app.py               # FastAPI backend — API dự đoán, AI phân tích, DB
│   ├── .env                 # Biến môi trường (API keys, PostgreSQL) — KHÔNG push Git
│   ├── .env.example         # Mẫu biến môi trường (push Git)
│   └── static/              # Frontend
│       ├── index.html       # Giao diện Neo-Brutalism UI
│       ├── style.css        # CSS với dark mode, animations
│       └── script.js        # JavaScript logic (form, API calls, AI)
│
├── notebooks/               # Jupyter Notebooks — phân tích & trình bày kết quả
│   ├── 01_eda_and_powerbi.ipynb       # Khám phá dữ liệu
│   ├── 02_baseline_vs_sampling.ipynb  # So sánh Baseline vs Under-sampling vs SMOTE
│   └── 03_tuning_results.ipynb        # Kết quả tuning siêu tham số
│
├── data/                    # Dữ liệu đầu vào
│   └── creditcard.csv       # Dataset gốc (284.807 giao dịch)
│
├── models/                  # Model đã huấn luyện (file .pkl)
│   ├── robust_scaler.pkl              # Scaler đã fit (dùng khi inference)
│   ├── random_forest_smote.pkl        # ★ Production model — RF + SMOTE + Tuned
│   ├── logistic_regression_smote.pkl  # ★ Production model — LR + SMOTE + Tuned
│   ├── knn_smote.pkl                  # ★ Production model — KNN + SMOTE + Tuned
│   ├── svm_smote.pkl                  # ★ Production model — SVM + SMOTE + Tuned
│   └── logistic_regression_baseline.pkl  # Model báo cáo (so sánh)
│
└── reports/                 # Output: biểu đồ, CSV, JSON
    ├── best_params_*.json             # Siêu tham số tốt nhất từ Optuna
    ├── all_models_*.csv               # Bảng so sánh 4 model
    ├── cm_*.png                       # Confusion matrix từng model
    ├── pr_*.png                       # Precision-Recall curves
    ├── feat_imp_*.png / .csv          # Feature importance (Random Forest)
    └── eda_*.png / .csv               # Biểu đồ & thống kê EDA
```

---

## 3. Chi tiết source code (`src/`)

### 3.1. `config.py` — Cấu hình toàn cục

| Hằng số | Giá trị | Ý nghĩa |
| ------- | ------- | -------- |
| `RANDOM_STATE` | `42` | Seed → tái lập kết quả |
| `TEST_SIZE` | `0.2` | 80% train / 20% test |
| `TARGET_COL` | `"Class"` | 0 = Legit, 1 = Fraud |
| `SCALE_COLS` | `["Amount", "Time"]` | V1–V28 đã PCA-scaled sẵn |
| `MODEL_NAMES` | RF, LR, KNN, SVM | 4 model |
| `TUNER_N_TRIALS` | `50` | Số trial Optuna |
| `TUNER_CV_FOLDS` | `5` | Stratified K-Fold |

### 3.2. `preprocessing.py` — Tiền xử lý

| Hàm | Chức năng |
| --- | --------- |
| `load_data()` | Đọc CSV, kiểm tra, thống kê |
| `scale_features()` | RobustScaler (median + IQR, robust với outlier) |
| `split_data()` | Stratified 80/20 split |
| `apply_undersampling()` | Giảm Legit = Fraud (~788 mẫu) |
| `apply_smote()` | Tạo thêm Fraud tổng hợp (~454.000 mẫu) |

### 3.3. `models.py` — Model Factory

| Model | Class | Đặc điểm |
| ----- | ----- | --------- |
| Random Forest | `RandomForestClassifier` | Ensemble, `class_weight="balanced"` |
| Logistic Regression | `LogisticRegression` | `class_weight="balanced"` |
| KNN | `KNeighborsClassifier` | K=5, Minkowski |
| SVM | `SVC` | RBF kernel, `probability=True` |

### 3.4. `evaluator.py` — Đánh giá

6 metrics: Accuracy, Precision, **Recall** (quan trọng nhất), F1-Score, AUC-ROC, AUC-PR.
Output: confusion matrix, PR curves, feature importance, CSV so sánh.

### 3.5. `tuner.py` — Optuna Tuning

**Kết quả tuning thực tế:**

| Model | Best F1 (CV) | Best Params |
| ----- | ------------ | ----------- |
| **Random Forest** | **0.9999** | `n_estimators=200, max_depth=30` |
| **Logistic Reg.** | 0.9524 | `C=0.98, l1_ratio=0.16` |
| **KNN** | 0.9964 | `K=3, distance, manhattan` |
| **SVM** | 0.9988 | `RBF, C=28.48, balanced` |

### 3.6. `trainer.py` — Huấn luyện

```
Load data → Scale → Split → Sampling → Load best params → Fit → Evaluate → Save .pkl
```

---

## 4. Webapp — Ứng dụng Web

### Kiến trúc

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

### API Endpoints

| Endpoint | Method | Chức năng |
| -------- | ------ | --------- |
| `/` | GET | Trang chủ |
| `/models/` | GET | Danh sách 4 model |
| `/predict/` | POST | Dự đoán fraud/legit, lưu DB |
| `/ai-analysis/` | POST | AI phân tích (Gemini/Groq) |
| `/history/` | GET | Lịch sử giao dịch |

### 4 Model Production (SMOTE + Tuned)

| Model | Tên hiển thị | F1 |
| ----- | ------------ | -- |
| `random_forest_smote` | Random Forest | 0.9999 |
| `logistic_regression_smote` | Logistic Regression | 0.9524 |
| `knn_smote` | KNN | 0.9964 |
| `svm_smote` | SVM | 0.9988 |

### AI Dual Provider

| Provider | Model | Ưu điểm |
| -------- | ----- | -------- |
| Google Gemini | `gemini-1.5-flash` | Chất lượng cao |
| Groq | `llama-3.3-70b-versatile` | Nhanh, không giới hạn |

### PostgreSQL — `fraud_guard`

| Cột | Kiểu | Mô tả |
| --- | ---- | ----- |
| `id` | SERIAL PK | ID tự tăng |
| `timestamp` | TIMESTAMP | Thời điểm |
| `model_used` | VARCHAR | Model đã dùng |
| `amount` | FLOAT | Số tiền |
| `v_features` | JSONB | V1-V28 |
| `verdict` | VARCHAR | An toàn / Nghi ngờ / Gian lận |
| `probability` | FLOAT | Xác suất fraud |

Fallback: Tự động chuyển SQLite nếu PostgreSQL không khả dụng.

---

## 5. Chiến lược triển khai

### Tại sao SMOTE + Tuned Params?

| Chiến lược | Deploy? | Lý do |
| ---------- | ------- | ----- |
| Baseline | ❌ | Recall tệ, bỏ sót fraud |
| Under-sampling | ⚠️ | Mất ~99% data |
| **SMOTE** | ✅ | **Recall + F1 tốt nhất** |

- **Webapp:** 4 model SMOTE + Tuned → tên đơn giản (RF, LR, KNN, SVM)
- **Notebooks:** So sánh 3 strategy × 4 model = 12 thí nghiệm (báo cáo)

---

## 6. Notebooks

| Notebook | Nội dung |
| -------- | -------- |
| `01_eda_and_powerbi` | EDA: phân bố class, Amount, Time, correlation |
| `02_baseline_vs_sampling` | 4 model × 3 sampling = 12 thí nghiệm |
| `03_tuning_results` | Optuna tuning, F1/Recall/AUC, feature importance |

---

## 7. Công nghệ

### ML Pipeline
scikit-learn, imbalanced-learn (SMOTE), Optuna, pandas, numpy, matplotlib, seaborn, joblib

### Webapp
FastAPI, Uvicorn, google-generativeai (Gemini), groq, psycopg2-binary, python-dotenv

### Kỹ thuật ML quan trọng
- **`class_weight="balanced"`:** Phạt ~289x khi bỏ sót fraud
- **`RobustScaler`:** Scale bằng median/IQR, robust với outlier
- **Stratified K-Fold CV:** Giữ tỷ lệ fraud/legit khi tuning
- **SMOTE:** Tạo mẫu fraud tổng hợp, giữ toàn bộ data Legit
- **TPE (Optuna):** Bayesian optimization, thông minh hơn Grid Search

---

## 8. Hướng dẫn chạy

### Cài đặt
```bash
cd "Machine Learning"
pip install -r requirements.txt
```

### ML Pipeline
```bash
python run_pipeline.py --step all --sampling smote
```

### Webapp
```bash
cd "Machine Learning/webapp"
cp .env.example .env   # Chỉnh sửa API keys
/usr/bin/python3 -m uvicorn app:app --reload --port 8000
```
Mở: **http://localhost:8000**

### PostgreSQL
```bash
psql -U postgres -d fraud_guard
SELECT * FROM fraud_history;
```

### Notebooks
```bash
jupyter notebook "Machine Learning/notebooks/"
```

---

## 9. Biến môi trường (`.env`)

| Biến | Mô tả | Mặc định |
| ---- | ----- | -------- |
| `GEMINI_API_KEY` | API key Google Gemini | (bắt buộc) |
| `GROQ_API_KEY` | API key Groq | (bắt buộc) |
| `PG_DBNAME` | Database name | `fraud_guard` |
| `PG_USER` | PostgreSQL user | `postgres` |
| `PG_PASSWORD` | PostgreSQL password | (trống) |
| `PG_HOST` | PostgreSQL host | `localhost` |
| `PG_PORT` | PostgreSQL port | `5432` |

> **Bảo mật:** `.env` chứa API keys → nằm trong `.gitignore`, KHÔNG push Git.
