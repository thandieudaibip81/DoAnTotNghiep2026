# Credit Card Fraud Detection — DevOps & Observability

## 1. Tổng quan


1. **Containerization:** Đóng gói ứng dụng bằng Docker
2. **Container Orchestration:** Triển khai trên Kubernetes (K8s) cluster 3 nodes
3. **CI/CD Pipeline:** Tự động hóa build/deploy với Jenkins
4. **Helm Charts:** Quản lý cấu hình triển khai
5. **Centralized Logging:** ELK Stack (Elasticsearch, Filebeat, Kibana)
6. **Monitoring & Alerting:** Prometheus, Grafana, AlertManager → Telegram

---

## 2. Cấu trúc thư mục

```
Ops/
├── Dockerfile               # Docker image cho Webapp (Python 3.11)
├── build.sh                 # Script build Docker image local
│
├── helm/                    # Helm charts & values
│   ├── fraud-guard/         # Helm chart tự viết cho Webapp + PostgreSQL
│   │   ├── Chart.yaml       # Metadata chart (version 0.1.0)
│   │   ├── values.yaml      # Giá trị mặc định (image, replicas, ports)
│   │   └── templates/       # K8s manifests dạng template
│   │       ├── deployment.yaml        # Webapp Deployment (2 replicas)
│   │       ├── service.yaml           # Webapp Service (NodePort 30080)
│   │       ├── postgres.yaml          # PostgreSQL Deployment + Service
│   │       ├── configmap.yaml         # Biến môi trường không nhạy cảm
│   │       ├── secret.yaml            # API keys, DB password
│   │       └── jenkins-rolebinding.yaml  # RBAC cho Jenkins deploy
│   │
│   ├── jenkins-values.yaml            # Helm values cho Jenkins
│   ├── elasticsearch-values.yaml      # Helm values cho Elasticsearch
│   ├── kibana-values.yaml             # Helm values cho Kibana
│   ├── filebeat-values.yaml           # Helm values cho Filebeat
│   ├── prometheus-stack-values.yaml   # Helm values cho kube-prometheus-stack
│   └── postgres-exporter-values.yaml  # Helm values cho Postgres Exporter
│
└── k8s/                     # Cấu hình Kubernetes tĩnh (manual apply)
    ├── namespace.yaml        # Namespace "fraud-detection"
    ├── configmap.yaml        # ConfigMap (PG_HOST, PG_PORT, ...)
    ├── secret.yaml           # Secret (API keys, PG_PASSWORD)
    ├── postgres.yaml         # PostgreSQL Deployment + PVC + Service
    ├── deployment.yaml       # Webapp Deployment (2 replicas)
    ├── service.yaml          # Webapp Service (NodePort 30800)
    ├── jenkins.yaml          # Jenkins full stack (Namespace + SA + RBAC + PVC + Deployment + Service)
    └── deploy.sh             # Script deploy tự động theo thứ tự

Jenkinsfile                  # (Nằm ở root project) — CI/CD Pipeline definition
```

---

## 3. Containerization — Docker

### 3.1. `Dockerfile`

**Base image:** `python:3.11-slim`

**Luồng build:**

```
python:3.11-slim
    │
    ├── apt-get: libpq-dev, gcc          ← Cần cho psycopg2 (PostgreSQL driver)
    ├── pip install: requirements.txt     ← FastAPI, Uvicorn, AI SDKs
    ├── COPY webapp/app.py               ← Backend code
    ├── COPY webapp/static/              ← Frontend (HTML, CSS, JS)
    ├── COPY models/                     ← Pre-trained ML models (.pkl)
    │
    ├── EXPOSE 8000
    ├── HEALTHCHECK: GET /models/        ← K8s readiness probe dùng endpoint này
    └── CMD: uvicorn app:app --host 0.0.0.0 --port 8000
```

**Điểm quan trọng:**
- **Models được bake vào image:** File `.pkl` được COPY trực tiếp vào image → không cần mount volume khi deploy → đơn giản hóa deployment
- **Multi-platform:** Build với `--platform linux/amd64` để chạy trên K8s cluster (AMD64)
- **Health check:** Kiểm tra endpoint `/models/` mỗi 30 giây
- **Requirements:** Dockerfile dùng `webapp/requirements.txt` (FastAPI, Uvicorn, AI SDKs) — KHÔNG phải root `requirements.txt` (ML pipeline)
- **`.dockerignore`:** Loại trừ `.env`, `__pycache__`, `.git`, `*.md`, `*.db` khỏi build context để giảm kích thước image

### 3.2. `build.sh` — Script build local

```bash
./build.sh v1.0     # Build image với tag v1.0
./build.sh          # Build image với tag "latest"
```

**Cách hoạt động:**
- Chuyển về root project (`cd .. từ Ops/`)
- Build context là `"Machine Learning"` → Docker có thể access cả `webapp/` và `models/`
- Dockerfile nằm ở `Ops/Dockerfile` nhưng context ở `Machine Learning/`
- Image name: `thandieudaibip/fraud-detection-webapp`

---

## 4. Kubernetes — Container Orchestration

### 4.1. Kiến trúc Cluster

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster (3 Nodes)                  │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Master   │    │  Node 1  │    │  Node 2  │                   │
│  │(Control)  │    │(Worker)  │    │(Worker)  │                   │
│  │           │    │          │    │          │                   │
│  │ .110      │    │ .111     │    │ .112     │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                                  │
│  Namespaces:                                                     │
│  ├── fraud-detection  → Webapp (2 replicas) + PostgreSQL        │
│  ├── jenkins          → Jenkins CI/CD Server                    │
│  ├── logging          → Elasticsearch + Kibana + Filebeat       │
│  └── monitoring       → Prometheus + Grafana + AlertManager     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2. NFS — Persistent Storage

**Vấn đề:** Khi Pod restart hoặc chuyển node, dữ liệu trên `emptyDir` sẽ bị mất.

**Giải pháp:** Sử dụng `nfs-client-provisioner` (StorageClass: `nfs-client`) để tự động tạo PersistentVolume trên NFS server riêng biệt.

**NFS Server (VM 200 trên Proxmox):**

| Thông số | Giá trị |
| --- | --- |
| **IP** | `192.168.1.60` |
| **OS** | Ubuntu Server |
| **Ổ đĩa riêng** | `/dev/sdb` — 44GB mount tại `/mnt/nfs_data` |
| **Export** | `/mnt/nfs_data *(rw,sync,no_subtree_check,no_root_squash,insecure)` |

**Cài đặt NFS Provisioner trên K8s (file `helm/nfs-provisioner-values.yaml`):**
```bash
helm repo add nfs-subdir-external-provisioner \
  https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/

helm install nfs-client nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  -f helm/nfs-provisioner-values.yaml
```

**Cấu hình chính:**
- `storageClass.name: nfs-client` — Tên StorageClass dùng trong tất cả PVC
- `storageClass.defaultClass: true` — Đặt làm StorageClass mặc định
- `reclaimPolicy: Retain` — Giữ lại dữ liệu khi PVC bị xóa
- `archiveOnDelete: true` — Đổi tên thư mục thành `archived-...` thay vì xóa hẳn

**Các PVC sử dụng NFS:**

| PVC | Namespace | Kích thước | Mục đích |
| --- | --------- | ---------- | -------- |
| `postgres-pvc` | fraud-detection | 2Gi | Dữ liệu PostgreSQL |
| `jenkins-pvc` | jenkins | 5Gi | Jenkins home (jobs, plugins) |
| Elasticsearch PVC | logging | 5Gi | Log data & indices |
| Grafana PVC | monitoring | 2Gi | Dashboards & settings |
| Prometheus PVC | monitoring | 5Gi | Metrics time-series data |

### 4.3. Namespace: `fraud-detection`

**Thành phần:**

| Resource | Loại | Chi tiết |
| -------- | ---- | -------- |
| Webapp | Deployment (2 replicas) | FastAPI + ML models, RollingUpdate strategy |
| PostgreSQL | Deployment (1 replica) | `postgres:16-alpine`, UTF-8 encoding |
| Webapp Service | NodePort 30080 (Helm) / 30800 (static) | Truy cập từ bên ngoài cluster |
| Postgres Service | NodePort 30543 | Truy cập DB từ bên ngoài (debug) |
| ConfigMap | — | `PG_DBNAME`, `PG_USER`, `PG_HOST`, `PG_PORT` |
| Secret | — | `GEMINI_API_KEY`, `GROQ_API_KEY`, `PG_PASSWORD` |

**Webapp Deployment features:**
- **2 replicas:** High availability — nếu 1 pod chết, pod còn lại vẫn phục vụ
- **RollingUpdate:** `maxUnavailable=1, maxSurge=1` → zero-downtime deployment
- **Probes:**
  - `readinessProbe`: GET `/models/` mỗi 5s → chỉ route traffic khi app sẵn sàng
  - `livenessProbe`: GET `/models/` mỗi 10s → tự restart nếu app bị treo
- **Resources:** Request 256Mi RAM / 250m CPU, Limit 512Mi / 500m

**PostgreSQL features:**
- **`POSTGRES_INITDB_ARGS: --encoding=UTF8 --locale=C`:** Đảm bảo lưu tiếng Việt đúng
- **`subPath: postgres-data`:** Tránh xung đột với NFS `.lost+found`
- **Probes:** `pg_isready` kiểm tra kết nối mỗi 5-10 giây

### 4.4. `deploy.sh` — Script deploy thủ công

```bash
cd k8s/
./deploy.sh
```

**Thứ tự deploy:**
1. Tạo namespace `fraud-detection`
2. Apply secrets + configmap
3. Deploy PostgreSQL → chờ ready
4. Deploy webapp → chờ ready
5. In trạng thái pods + services + URLs truy cập

---

## 5. CI/CD Pipeline — Jenkins

### 5.1. Kiến trúc Jenkins trên K8s

Jenkins chạy trong namespace `jenkins` với:
- **ServiceAccount `jenkins-admin`:** Có quyền `cluster-admin` → có thể deploy vào mọi namespace
- **PVC trên NFS (5Gi):** Lưu Jenkins home, jobs, plugins
- **Ports:** Web UI (32000), Agent (50000)
- **Init container:** `busybox` chạy `chown` để fix permission NFS

### 5.2. `Jenkinsfile` — Pipeline Definition

```
┌──────────────────────────────────────────────────────────────┐
│                    Jenkins CI/CD Pipeline                      │
│                                                                │
│  ┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌───────┐ │
│  │Checkout  │───▶│Build Docker  │───▶│Push to   │───▶│Deploy │ │
│  │Source    │    │Image         │    │DockerHub │    │to K8s │ │
│  │Code     │    │              │    │          │    │       │ │
│  └─────────┘    └──────────────┘    └──────────┘    └───────┘ │
│                                                                │
│  Agent: Kubernetes Pod                                         │
│  ├── Container "docker": docker:27-dind (build & push)        │
│  └── Container "kubectl": alpine/k8s:1.32.4 (deploy)         │
└──────────────────────────────────────────────────────────────┘
```

**Chi tiết từng Stage:**

| Stage | Container | Hành động |
| ----- | --------- | --------- |
| **Checkout** | jnlp (default) | `git clone` từ GitHub (`main` branch) |
| **Build Docker Image** | `docker` (DinD) | Build `Ops/Dockerfile` với context `"Machine Learning"`, tag = `BUILD_NUMBER` + `latest` |
| **Push to Docker Hub** | `docker` | Login bằng credentials `dockerhub-credentials`, push cả 2 tags |
| **Deploy to K8s** | `kubectl` | `kubectl set image` → `kubectl rollout status` (timeout 300s) |

**Biến môi trường pipeline:**

| Biến | Giá trị | Mục đích |
| ---- | ------- | -------- |
| `DOCKER_IMAGE` | `thandieudaibip/fraud-detection-webapp` | Tên image trên DockerHub |
| `DOCKER_TAG` | `${BUILD_NUMBER}` | Tag theo số build Jenkins |
| `KUBE_NS` | `fraud-detection` | Namespace K8s để deploy |

**Kubernetes Dynamic Agent:**
- Jenkins KHÔNG chạy build trên master — thay vào đó tạo **Pod tạm thời** trên K8s cho mỗi build
- Pod chứa 2 containers: `docker` (DinD — Docker in Docker) và `kubectl`
- Sau khi build xong, Pod bị xóa → tiết kiệm tài nguyên

### 5.3. Helm Chart — `fraud-guard`

**Chart.yaml:** `fraud-guard`, version `0.1.0`, appVersion `1.0.0`

**Values mặc định (`values.yaml`):**

| Key | Giá trị | Mô tả |
| --- | ------- | ----- |
| `webapp.image.repository` | `thandieudaibip/fraud-detection-webapp` | Docker image |
| `webapp.image.tag` | `latest` | Image tag |
| `webapp.replicas` | `2` | Số replicas webapp |
| `webapp.service.nodePort` | `30080` | Port truy cập webapp |
| `postgres.image.tag` | `16-alpine` | PostgreSQL version |
| `postgres.database` | `fraud_guard` | Tên database |
| `postgres.persistence.storageClass` | `nfs-client` | NFS storage |

**Templates:** 6 files template tạo Deployment, Service, PostgreSQL, ConfigMap, Secret, RBAC cho Jenkins.

---

## 6. Centralized Logging — ELK Stack

### 6.1. Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                    Namespace: logging                     │
│                                                           │
│  ┌──────────┐         ┌───────────────┐    ┌──────────┐  │
│  │ Filebeat  │────────▶│ Elasticsearch │◀───│  Kibana  │  │
│  │(DaemonSet)│  logs   │  (Single Node)│    │  (Web UI)│  │
│  │mỗi node  │         │  5Gi NFS      │    │          │  │
│  └──────────┘         └───────────────┘    └──────────┘  │
│                                                           │
│  Ports: ES=31920, Kibana=31601                           │
└─────────────────────────────────────────────────────────┘
```

### 6.2. Filebeat (`filebeat-values.yaml`)

**Loại:** DaemonSet — chạy 1 instance trên **mỗi node** trong cluster

**Cấu hình:**
- **Input:** Container logs từ `/var/log/containers/*.log`
- **Processor:** `add_kubernetes_metadata` → thêm thông tin Pod name, namespace, container name vào mỗi dòng log
- **Output:** Gửi log đến Elasticsearch (`elasticsearch-master.logging.svc.cluster.local:9200`)
- **Index pattern:** `filebeat-YYYY.MM.DD` (daily index)
- **Tolerations:** `NoSchedule` → chạy được cả trên master node
- **Resources:** 50m-200m CPU, 100Mi-200Mi RAM

### 6.3. Elasticsearch (`elasticsearch-values.yaml`)

**Mode:** Single-node (tối ưu cho cluster nhỏ)

**Cấu hình:**
- **Version:** 7.17.3
- **JVM Heap:** 1GB (`-Xmx1g -Xms1g`) = 50% RAM limit
- **Storage:** 5Gi trên NFS (`nfs-client`)
- **Security:** Tắt hoàn toàn (`xpack.security.enabled: false`) — internal cluster, không cần HTTPS
- **Health check:** `wait_for_status=yellow` (single-node không cần green)
- **Node selector:** `node2` (node có nhiều RAM nhất)
- **Service:** NodePort 31920

### 6.4. Kibana (`kibana-values.yaml`)

**Cấu hình:**
- Kết nối: `http://elasticsearch-master:9200`
- Service: NodePort 31601
- Resources: 100m-500m CPU, 256Mi-1Gi RAM

**Sử dụng:** Truy cập `http://<node-ip>:31601` → tạo Index Pattern `filebeat-*` → Discover → tìm kiếm log theo thời gian, namespace, pod name, keyword.

---

## 7. Monitoring & Alerting — Prometheus + Grafana

### 7.1. Kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                     Namespace: monitoring                         │
│                                                                    │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────────┐  │
│  │  Prometheus   │───▶│   Grafana   │    │    AlertManager     │  │
│  │  (Scraper)    │    │ (Dashboards)│    │  → Telegram Bot     │  │
│  │  5Gi NFS      │    │  2Gi NFS   │    │                      │  │
│  └──────┬────────┘    └─────────────┘    └──────────────────────┘  │
│         │                                                          │
│    ┌────┴────────────────────────┐                                 │
│    │   Metrics Sources           │                                 │
│    ├── Node Exporter (mỗi node) │                                 │
│    ├── kube-state-metrics        │                                 │
│    └── postgres-exporter         │                                 │
│                                                                    │
│  Ports: Prometheus=30090, Grafana=31300, AlertManager=30903       │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2. Prometheus (`prometheus-stack-values.yaml`)

**Helm chart:** `kube-prometheus-stack` (bao gồm Prometheus, Grafana, AlertManager, Node Exporter)

**Prometheus config:**
- **Retention:** 7 ngày
- **Storage:** 5Gi trên NFS
- **Resources:** 200m-1000m CPU, 512Mi-2Gi RAM
- **Service Monitor:** Scrape tất cả namespace (`serviceMonitorSelectorNilUsesHelmValues: false`)

**Các component tắt** (không cần trên cluster nhỏ):
- `kubeEtcd`, `kubeScheduler`, `kubeControllerManager`, `kubeProxy`

### 7.3. Grafana — Dashboards

**Truy cập:** `http://<node-ip>:31300` | Admin: `admin` / `admin123`

**3 Dashboard tự động cài đặt:**

| Dashboard | ID Grafana.com | Mục đích |
| --------- | -------------- | -------- |
| **Node Exporter Full** | 1860 (rev 37) | CPU, RAM, Disk, Network của từng node |
| **PostgreSQL** | 9628 (rev 7) | Connections, cache hit ratio, queries/sec |
| **K8s Cluster Overview** | 6417 (rev 1) | Tổng quan tài nguyên cluster |

**Persistence:** 2Gi trên NFS → dashboards custom không bị mất khi restart

### 7.4. Postgres Exporter (`postgres-exporter-values.yaml`)

**Vai trò:** Thu thập metrics từ PostgreSQL → expose cho Prometheus scrape

**Kết nối:**
- Host: `postgres-service.fraud-detection.svc.cluster.local:5432`
- Database: `fraud_guard`
- User: `postgres`

**ServiceMonitor:** Label `release: prometheus` → Prometheus tự động discover

**Metrics quan trọng:**
- `pg_stat_database_numbackends` — số connections đang hoạt động
- `pg_stat_database_tup_returned` — số rows returned
- `pg_up` — PostgreSQL có sống hay không (dùng cho alert)

### 7.5. AlertManager → Telegram

**Cấu hình routing:**

| Tham số | Giá trị | Ý nghĩa |
| ------- | ------- | -------- |
| `group_by` | `alertname, namespace` | Nhóm cảnh báo cùng loại |
| `group_wait` | 30s | Chờ 30s thu thập alerts cùng nhóm trước khi gửi |
| `group_interval` | 5m | Khoảng cách giữa 2 lần gửi alert cùng nhóm |
| `repeat_interval` | 4h | Nhắc lại alert chưa resolve mỗi 4 tiếng |

**Telegram config:**
- Chat ID cá nhân → nhận thông báo trực tiếp trên điện thoại
- Format HTML → hiển thị đẹp trên Telegram

### 7.6. Custom Alert Rules

**6 alert rules được cấu hình:**

| Alert | Điều kiện | Severity | Thời gian chờ |
| ----- | --------- | -------- | ------------- |
| **PodCrashLooping** | Pod trong `fraud-detection` restart nhiều lần/5 phút | 🔴 Critical | 2 phút |
| **NodeHighMemory** | RAM sử dụng > 85% | 🟡 Warning | 5 phút |
| **NodeHighCPU** | CPU sử dụng > 85% | 🟡 Warning | 5 phút |
| **PostgreSQLDown** | `pg_up == 0` | 🔴 Critical | 1 phút |
| **PostgreSQLHighConnections** | Connections > 80% max | 🟡 Warning | 5 phút |
| **WebappReplicasLow** | Webapp < 2 replicas đang chạy | 🟡 Warning | 3 phút |

---

## 8. Jenkins Helm Values (`jenkins-values.yaml`)

**Cấu hình:**
- **Service:** NodePort 32000
- **Plugins:** kubernetes, workflow-aggregator, git, configuration-as-code, docker-workflow
- **Persistence:** 5Gi trên NFS
- **RBAC:** Tạo ServiceAccount `jenkins-admin` với full quyền
- **Startup probe:** `failureThreshold=60, periodSeconds=10` → chờ tối đa 10 phút để Jenkins khởi động (lần đầu cần cài plugins)

---

## 9. Công nghệ sử dụng

| Công cụ / Stack | Version | Vai trò |
| ---------------- | ------- | ------- |
| **Docker** | 27 (DinD) | Đóng gói ứng dụng thành container |
| **Kubernetes** | 1.32.x | Container Orchestration |
| **Helm** | 3.x | Package manager cho K8s |
| **Jenkins** | LTS | CI/CD Pipeline |
| **Elasticsearch** | 7.17.3 | Lưu trữ & index log tập trung |
| **Filebeat** | 7.17.3 | Thu thập log từ containers |
| **Kibana** | 7.17.3 | Giao diện tìm kiếm & phân tích log |
| **Prometheus** | Latest | Thu thập metrics (time-series) |
| **Grafana** | Latest | Dashboards trực quan |
| **AlertManager** | Latest | Routing & gửi cảnh báo |
| **Node Exporter** | Latest | Metrics phần cứng (CPU, RAM, Disk) |
| **Postgres Exporter** | Latest | Metrics PostgreSQL |
| **NFS Client Provisioner** | — | Dynamic PV provisioning trên NFS |

---

## 10. Hướng dẫn triển khai

### 10.1. Triển khai thủ công (Static manifests)

```bash
# 1. Deploy ứng dụng
cd Ops/k8s/
./deploy.sh

# 2. Deploy Jenkins
kubectl apply -f jenkins.yaml

# 3. Truy cập
# Webapp:    http://192.168.10.110:30800
# Jenkins:   http://192.168.10.110:32000
# PostgreSQL: psql -h 192.168.10.110 -p 30543 -U postgres -d fraud_guard
```

### 10.2. Triển khai bằng Helm

```bash
# 1. Webapp + PostgreSQL
helm install fraud-guard Ops/helm/fraud-guard/ -n fraud-detection --create-namespace

# 2. Jenkins
helm install jenkins jenkins/jenkins -f Ops/helm/jenkins-values.yaml -n jenkins --create-namespace

# 3. ELK Stack (namespace: logging)
helm install elasticsearch elastic/elasticsearch -f Ops/helm/elasticsearch-values.yaml -n logging --create-namespace
helm install kibana elastic/kibana -f Ops/helm/kibana-values.yaml -n logging
helm install filebeat elastic/filebeat -f Ops/helm/filebeat-values.yaml -n logging

# 4. Monitoring (namespace: monitoring)
helm install prometheus prometheus-community/kube-prometheus-stack -f Ops/helm/prometheus-stack-values.yaml -n monitoring --create-namespace
helm install postgres-exporter prometheus-community/prometheus-postgres-exporter -f Ops/helm/postgres-exporter-values.yaml -n monitoring
```

### 10.3. Truy cập Services

| Service | URL | Ghi chú |
| ------- | --- | ------- |
| **Webapp** | `http://<node-ip>:30080` | Helm / `30800` Static |
| **Jenkins** | `http://<node-ip>:32000` | Admin password: `kubectl get secret` |
| **Kibana** | `http://<node-ip>:31601` | Tạo Index Pattern `filebeat-*` |
| **Grafana** | `http://<node-ip>:31300` | admin / admin123 |
| **Prometheus** | `http://<node-ip>:30090` | Query PromQL |
| **AlertManager** | `http://<node-ip>:30903` | Xem active alerts |
| **Elasticsearch** | `http://<node-ip>:31920` | API endpoint |
| **PostgreSQL** | `<node-ip>:30543` | psql client |

### 10.4. Build & Push Docker Image

```bash
# Build local
cd Ops/
./build.sh v2.0

# Push lên DockerHub
docker push thandieudaibip/fraud-detection-webapp:v2.0
```

Hoặc để Jenkins tự động build/push qua pipeline — chỉ cần push code lên GitHub `main` branch.

---

## 11. Luồng CI/CD End-to-End

```
Developer push code
        │
        ▼
   GitHub (main branch)
        │
        ▼
   Jenkins detect change (webhook/poll)
        │
        ▼
   Tạo K8s Pod (docker + kubectl containers)
        │
        ├── docker build "Machine Learning/" → image:BUILD_NUMBER
        ├── docker push → DockerHub
        ├── kubectl set image → update Deployment
        └── kubectl rollout status → chờ deploy xong
        │
        ▼
   Pod tạm bị xóa (giải phóng tài nguyên)
        │
        ▼
   Webapp pods rolling update (zero-downtime)
        │
        ▼
   Filebeat thu log → Elasticsearch → Kibana
   Prometheus scrape metrics → Grafana dashboards
   AlertManager → Telegram (nếu có vấn đề)
```
