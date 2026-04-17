# Credit Card Fraud Detection — Machine Learning Pipeline

## 1. Tổng quan dự án

Dự án xây dựng pipeline **phát hiện gian lận thẻ tín dụng** (Credit Card Fraud Detection) sử dụng bốn mô hình Machine Learning, so sánh hiệu quả trên ba chiến lược xử lý dữ liệu mất cân bằng (imbalanced data), và tối ưu siêu tham số bằng Optuna.

**Bài toán cốt lõi:** Dataset gồm 284.807 giao dịch, trong đó chỉ **492 giao dịch gian lận (0,173%)**. Đây là bài toán **phân loại nhị phân với dữ liệu cực kỳ mất cân bằng** — nếu model luôn dự đoán "Legit" thì accuracy đã đạt 99,83%, nhưng bỏ sót toàn bộ fraud.

---

## 2. Cấu trúc thư mục

```
Machine Learning/
├── run_pipeline.py          # Entry point CLI — chạy toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện Python
├── PROJECT_EXPLANATION.md
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
├── notebooks/               # Jupyter Notebooks — phân tích & trình bày kết quả
│   ├── 01_eda_and_powerbi.ipynb       # Khám phá dữ liệu
│   ├── 02_baseline_vs_sampling.ipynb  # So sánh Baseline vs Under-sampling vs SMOTE
│   └── 03_tuning_results.ipynb        # Kết quả tuning siêu tham số
│
├── data/                    # Dữ liệu đầu vào
│   └── creditcard.csv       # Dataset gốc
│
├── models/                  # Model đã huấn luyện (file .pkl)
│   ├── robust_scaler.pkl    # Scaler đã fit (dùng lại khi inference)
│   ├── random_forest_smote.pkl
│   ├── logistic_regression_smote.pkl
│   ├── knn_smote.pkl
│   └── svm_smote.pkl
│
└── reports/                 # Output: biểu đồ, CSV, JSON
    ├── best_params_*.json             # Siêu tham số tốt nhất từ Optuna
    ├── all_models_baseline.csv        # Bảng so sánh 4 model (baseline)
    ├── all_models_under_sampling.csv  # Bảng so sánh 4 model (under-sampling)
    ├── all_models_smote.csv           # Bảng so sánh 4 model (SMOTE)
    ├── cm_*.png                       # Confusion matrix từng model
    ├── pr_*.png                       # Precision-Recall curves
    ├── feat_imp_*.png / .csv          # Feature importance (Random Forest)
    └── eda_*.png / .csv               # Biểu đồ & thống kê EDA
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
- Khi `fit=False`: load scaler đã lưu (dùng khi inference/evaluate)

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

**Không gian tìm kiếm (search space) cho từng model:**

#### Random Forest

| Tham số             | Khoảng                                   | Ý nghĩa                              |
| ------------------- | ---------------------------------------- | ------------------------------------ |
| `n_estimators`      | 50 – 300, bước 50                        | Số cây                               |
| `max_depth`         | 5 – 30                                   | Giới hạn tối đa để chống overfitting |
| `min_samples_split` | 2 – 10                                   | Số mẫu tối thiểu để split            |
| `class_weight`      | `"balanced"` hoặc `"balanced_subsample"` | Cách tính trọng số                   |

#### Logistic Regression

| Tham số        | Khoảng                   | Ý nghĩa                                        |
| -------------- | ------------------------ | ---------------------------------------------- |
| `C`            | 0.01 – 100 (log scale)   | Regularization: C nhỏ = regularize mạnh        |
| `l1_ratio`     | 0.0 – 1.0                | Tỷ lệ L1/L2 penalty (0 = pure L2, 1 = pure L1) |
| `solver`       | `"saga"` (cố định)       | Solver duy nhất hỗ trợ ElasticNet              |
| `class_weight` | `"balanced"` hoặc `None` | Có/không dùng trọng số tự động                 |

#### KNN

| Tham số       | Khoảng                                        | Ý nghĩa                                 |
| ------------- | --------------------------------------------- | --------------------------------------- |
| `n_neighbors` | 3 – 25, bước 2                                | Số láng giềng (chỉ số lẻ để tránh tie)  |
| `weights`     | `"uniform"` / `"distance"`                    | Trọng số đều hay nghịch đảo khoảng cách |
| `metric`      | `"minkowski"` / `"euclidean"` / `"manhattan"` | Hàm khoảng cách                         |

#### SVM

| Tham số             | Khoảng                   | Ý nghĩa        |
| ------------------- | ------------------------ | -------------- |
| `C`                 | 0.1 – 50 (log scale)     | Regularization |
| `kernel`            | `"rbf"` / `"poly"`       | Loại kernel    |
| `class_weight`      | `"balanced"` hoặc `None` | Trọng số class |
| `gamma` (nếu rbf)   | `"scale"` / `"auto"`     | Hệ số kernel   |
| `degree` (nếu poly) | 2 – 4                    | Bậc đa thức    |

**Tối ưu tốc độ:** KNN và SVM rất chậm trên toàn bộ dataset (~227.000 mẫu), nên tuner tự động **subsample 15%** (giữ tỷ lệ stratified) để tuning nhanh hơn.

**Output:** File JSON trong `reports/best_params_{model}.json`, ví dụ:

```json
{
  "model": "random_forest",
  "best_f1": 0.8523,
  "params": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 3,
    "class_weight": "balanced"
  }
}
```

**Các hàm chính:**

| Hàm                                                         | Chức năng                           |
| ----------------------------------------------------------- | ----------------------------------- |
| `tune_model(model_name, X_train, y_train, n_trials=50)`     | Tuning 1 model → trả về best params |
| `tune_all(X_train, y_train, model_names=None, n_trials=50)` | Tuning tất cả model                 |

---

### 3.6. `trainer.py` — Huấn luyện & Lưu Model

**Vai trò:** Huấn luyện model với tham số tốt nhất, lưu file `.pkl`, và (tùy chọn) log lên MLflow.

**Luồng xử lý:**

```
Load data → Scale → Split → Sampling → Load best params → Fit → Evaluate → Save .pkl → (MLflow)
```

**Các hàm chính:**

| Hàm                                                  | Chức năng                                       |
| ---------------------------------------------------- | ----------------------------------------------- |
| `train_model(name, X_train, y_train, params)`        | Fit 1 model, nhận params từ tuner hoặc defaults |
| `train_all(sampling, model_names, use_tuned_params)` | Pipeline end-to-end cho tất cả model            |
| `save_model(model, name, sampling)`                  | Lưu model → `models/{name}_{sampling}.pkl`      |
| `load_model(name, sampling)`                         | Load model đã lưu                               |
| `_load_best_params(model_name)`                      | Đọc `reports/best_params_{model}.json`          |

**Tích hợp MLflow (tùy chọn):**
Nếu biến môi trường `MLFLOW_TRACKING_URI` được set, trainer tự động log params, metrics, và model artifacts lên MLflow server.

---

### 3.7. `db_utils.py` — Tích hợp PostgreSQL (Tùy chọn)

**Vai trò:** Đẩy kết quả vào PostgreSQL để Power BI kết nối trực tiếp.

> **Module này là tùy chọn** — pipeline hoạt động hoàn toàn bình thường không cần database. Power BI cũng có thể import trực tiếp từ CSV trong `reports/`.

**Ba bảng được tạo:**

| Bảng                 | Mục đích                                                          |
| -------------------- | ----------------------------------------------------------------- |
| `model_metrics`      | Lưu accuracy, precision, recall, f1, AUC cho mỗi model + sampling |
| `feature_importance` | Lưu feature importance scores (Random Forest)                     |
| `predictions_log`    | Ghi lại predictions cho monitoring trong production               |

**Kết nối qua biến môi trường:** `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

---

## 4. File `run_pipeline.py` — CLI Entry Point

**Vai trò:** Điểm vào duy nhất để chạy toàn bộ pipeline qua command line.

**Các bước (steps) hỗ trợ:**

| Step         | Lệnh                                                      | Mô tả                                           |
| ------------ | --------------------------------------------------------- | ----------------------------------------------- |
| **baseline** | `python run_pipeline.py --step baseline`                  | Train 4 model trên dữ liệu thô (không sampling) |
| **tune**     | `python run_pipeline.py --step tune --sampling smote`     | Tuning siêu tham số bằng Optuna                 |
| **train**    | `python run_pipeline.py --step train --sampling smote`    | Train với params đã tune                        |
| **evaluate** | `python run_pipeline.py --step evaluate --sampling smote` | Evaluate model đã train trước đó                |
| **all**      | `python run_pipeline.py --step all --sampling smote`      | Chạy cả 3: baseline → tune → train              |

**Shortcut tên model:** `rf`, `lr`, `knn`, `svm`

```bash
# Chỉ tune Random Forest và Logistic Regression
python run_pipeline.py --step tune --models rf,lr --sampling smote
```

---

## 5. Notebooks

### 5.1. `01_eda_and_powerbi.ipynb` — Khám phá dữ liệu (EDA)

- Thống kê mô tả (mean, std, min, max cho mỗi feature)
- Phân bố class (`Class = 0` vs `Class = 1`) → chứng minh imbalanced
- Phân bố Amount và Time
- Ma trận tương quan (Correlation Heatmap)
- Xuất CSV đã clean cho Power BI

### 5.2. `02_baseline_vs_sampling.ipynb` — So sánh Baseline vs Sampling

Notebook chính, so sánh **4 model × 3 chiến lược sampling** = 12 thí nghiệm:

| Section                    | Nội dung                                                     |
| -------------------------- | ------------------------------------------------------------ |
| **Phần 1: Baseline**       | 4 model trên dữ liệu **thô** (với `class_weight="balanced"`) |
| → So sánh LR               | So sánh `class_weight="balanced"` vs `class_weight=None`     |
| **Phần 2: Under-sampling** | 4 model trên dữ liệu đã **giảm mẫu Legit**                   |
| **Phần 3: SMOTE**          | 4 model trên dữ liệu đã **tạo thêm mẫu Fraud tổng hợp**      |
| **Phần 4: Tổng hợp**       | Bảng so sánh toàn bộ, confusion matrices grid 3×4            |

Mỗi model hiển thị: Classification Report, Confusion Matrix, bảng metrics.

### 5.3. `03_tuning_results.ipynb` — Kết quả Tuning

- Load `best_params_*.json` cho mỗi model
- Biểu đồ so sánh: Recall, F1, AUC-ROC, AUC-PR
- Feature Importance (Random Forest top-20 features)
- Bảng xếp hạng model cuối cùng (`final_model_ranking.csv`)

---

## 6. Công nghệ & Thư viện sử dụng

### 6.1. Thư viện chính

| Thư viện             | Phiên bản | Vai trò                                                              |
| -------------------- | --------- | -------------------------------------------------------------------- |
| **scikit-learn**     | ≥ 1.4     | Framework ML chính: models, metrics, preprocessing, cross-validation |
| **imbalanced-learn** | ≥ 0.12    | Xử lý imbalanced data: `SMOTE`, `RandomUnderSampler`                 |
| **Optuna**           | ≥ 3.5     | Tối ưu siêu tham số tự động (Bayesian optimization với TPE)          |
| **pandas**           | ≥ 2.1     | Xử lý dữ liệu dạng bảng                                              |
| **numpy**            | ≥ 1.26    | Tính toán số học                                                     |
| **matplotlib**       | ≥ 3.8     | Vẽ biểu đồ cơ bản                                                    |
| **seaborn**          | ≥ 0.13    | Vẽ biểu đồ thống kê đẹp hơn                                          |
| **joblib**           | ≥ 1.3     | Serialize/deserialize model (`.pkl` files)                           |
| **MLflow**           | ≥ 2.10    | (Tùy chọn) Experiment tracking                                       |
| **psycopg2**         | ≥ 2.9     | (Tùy chọn) Kết nối PostgreSQL                                        |
| **jupyter**          | ≥ 1.0     | Chạy notebooks                                                       |

### 6.2. Kỹ thuật ML quan trọng

#### `class_weight="balanced"`

- **Dùng ở:** Random Forest, Logistic Regression, SVM
- **Tại sao:** Dataset cực kỳ mất cân bằng (0,173% fraud). Nếu không có `class_weight="balanced"`, model sẽ thiên vị dự đoán class đa số (Legit) → accuracy cao nhưng Recall rất thấp (bỏ sót fraud)
- **Cách hoạt động:** Tự tính trọng số `weight = n_samples / (n_classes × n_samples_per_class)` → class Fraud được "phạt" nặng hơn ~289 lần khi model dự đoán sai
- **Ảnh hưởng lớn nhất ở Baseline** (dữ liệu thô). Ở Under-sampling/SMOTE, dữ liệu đã được cân bằng nên ảnh hưởng rất nhỏ

#### `RobustScaler`

- **Dùng ở:** `preprocessing.py` — scale cột `Amount` và `Time`
- **Tại sao không dùng StandardScaler?** `StandardScaler` dùng mean/std → dễ bị ảnh hưởng bởi outlier. `RobustScaler` dùng **median** và **IQR (khoảng tứ phân vị)** → robust hơn với outlier
- **Quan trọng vì:** Giao dịch fraud thường có `Amount` rất bất thường (outlier)

#### Stratified K-Fold Cross Validation

- **Dùng ở:** `tuner.py` khi tối ưu siêu tham số
- **Tại sao Stratified?** Đảm bảo mỗi fold có **tỷ lệ fraud/legit giống nhau** → đánh giá model công bằng và ổn định hơn

#### SMOTE vs Random Under-sampling

|                | Under-sampling        | SMOTE                          |
| -------------- | --------------------- | ------------------------------ |
| **Cách**       | Xóa bớt mẫu Legit     | Tạo thêm mẫu Fraud tổng hợp    |
| **Kích thước** | ~788 mẫu              | ~454.000 mẫu                   |
| **Ưu**         | Nhanh, đơn giản       | Giữ toàn bộ dữ liệu Legit      |
| **Nhược**      | Mất rất nhiều dữ liệu | Mẫu synthetic có thể gây noise |

#### TPE Sampler (Optuna)

- **Tree-structured Parzen Estimator** — thuật toán Bayesian optimization
- **So với Grid Search:** Grid Search thử tất cả tổ hợp (rất chậm). TPE **học từ các trial trước** để chọn tham số tiếp theo thông minh hơn → tìm được kết quả tốt hơn với ít trial hơn

---

## 7. Luồng dữ liệu tổng thể (Data Flow)

```
creditcard.csv
      │
      ▼
  load_data()          ← preprocessing.py
      │
      ▼
  scale_features()     ← RobustScaler (Amount, Time)
      │
      ▼
  split_data()         ← Stratified 80/20
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
  X_train, y_train                        X_test, y_test
      │                                    (giữ nguyên, không sampling)
      │
      ├──── Baseline (không sampling) ─────┐
      ├──── Under-sampling ────────────────┤
      └──── SMOTE ─────────────────────────┤
                                            │
                                            ▼
                                       get_model()    ← models.py
                                            │
                                            ▼
                                        model.fit()   ← trainer.py
                                            │
                                            ▼
                                    evaluate_model()   ← evaluator.py
                                            │
                                     ┌──────┴──────┐
                                     ▼              ▼
                              reports/*.csv    models/*.pkl
                              reports/*.png
```

---

## 8. Cách chạy dự án

### Cài đặt

```bash
cd "Machine Learning"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy pipeline tự động

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

### Chạy notebooks

```bash
jupyter notebook notebooks/
```

Mở lần lượt: `01_eda_and_powerbi.ipynb` → `02_baseline_vs_sampling.ipynb` → `03_tuning_results.ipynb`
