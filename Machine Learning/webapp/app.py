"""
app.py — FastAPI web server for Credit Card Fraud Detection.

Features:
    • Multi-model prediction (RF, LR, KNN, SVM across sampling strategies)
    • Dual AI chatbot (Gemini + Groq) with provider switching
    • Transaction history with PostgreSQL (SQLite fallback)
    • Neo-Brutalism UI served from static/
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import httpx
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
DB_PATH = BASE_DIR / "fraud_history.db"
SCALE_COLS = ["Amount", "Time"]  # Must match src/config.py

# PostgreSQL configuration (from .env)
PG_CONFIG = {
    "dbname": os.getenv("PG_DBNAME", "fraud_guard"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
}
DB_TYPE = "postgresql"  # Will auto-fallback to "sqlite" if connection fails

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(name)s │ %(message)s")
logger = logging.getLogger("webapp")

# ──────────────────────────────────────────────────
# Model Registry — 4 production models (SMOTE + Tuned Params)
# ──────────────────────────────────────────────────
# Chiến lược triển khai:
#   - 4 model SMOTE + Optuna Tuned Params = sản phẩm cuối cùng cho dự đoán
#   - Các model khác (baseline, undersample) vẫn nằm trong models/ để phục vụ
#     notebook so sánh / báo cáo, nhưng KHÔNG hiển thị trên giao diện web.

WEBAPP_MODELS = {
    "random_forest_smote": {
        "display_name": "Random Forest",
        "description": "Ensemble 200 cây quyết định, tuned bởi Optuna (F1=0.9999)",
    },
    "logistic_regression_smote": {
        "display_name": "Logistic Regression",
        "description": "Hồi quy logistic với ElasticNet, tuned bởi Optuna (F1=0.9524)",
    },
    "knn_smote": {
        "display_name": "KNN",
        "description": "K=3, khoảng cách Manhattan, tuned bởi Optuna (F1=0.9964)",
    },
    "svm_smote": {
        "display_name": "SVM",
        "description": "Kernel RBF, C=28.48, tuned bởi Optuna (F1=0.9988)",
    },
}


def discover_models() -> Dict[str, Any]:
    """Load the 4 production models (SMOTE + Tuned Params) for the web UI.

    Only models listed in WEBAPP_MODELS are exposed to the frontend.
    Other .pkl files in models/ (baseline, undersample) are kept for
    notebook comparisons / reports but NOT shown on the web interface.
    """
    available = {}
    scaler_path = MODELS_DIR / "robust_scaler.pkl"
    if not scaler_path.exists():
        logger.warning("Scaler not found at %s", scaler_path)
        return available

    for model_id, meta in WEBAPP_MODELS.items():
        pkl_file = MODELS_DIR / f"{model_id}.pkl"
        if pkl_file.exists():
            available[model_id] = {
                "file": str(pkl_file),
                "display_name": meta["display_name"],
                "description": meta["description"],
            }
            logger.info("Registered model: %s → %s", model_id, meta["display_name"])
        else:
            logger.warning("Model file not found: %s", pkl_file)

    return available


MODEL_REGISTRY = discover_models()
LOADED_MODELS: Dict[str, Any] = {}
SCALER = None


def load_scaler():
    global SCALER
    scaler_path = MODELS_DIR / "robust_scaler.pkl"
    if scaler_path.exists():
        SCALER = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)


def get_model(model_id: str):
    """Lazy-load and cache a model."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")

    if model_id not in LOADED_MODELS:
        import warnings
        path = MODEL_REGISTRY[model_id]["file"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = joblib.load(path)

        # Patch for sklearn version mismatch:
        # Models saved with sklearn>=1.8 may lack 'multi_class' attribute
        # which is required by sklearn<=1.6 for predict_proba()
        if hasattr(loaded, 'predict_proba') and not hasattr(loaded, 'multi_class'):
            loaded.multi_class = 'auto'

        LOADED_MODELS[model_id] = loaded
        logger.info("Loaded model: %s", model_id)

    return LOADED_MODELS[model_id]


# Load scaler on startup
load_scaler()

# ──────────────────────────────────────────────────
# Database — PostgreSQL with SQLite fallback
# ──────────────────────────────────────────────────


def get_db():
    """Get a database connection. Tries PostgreSQL first, falls back to SQLite."""
    global DB_TYPE
    if DB_TYPE == "postgresql":
        try:
            conn = psycopg2.connect(**PG_CONFIG)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.warning("PostgreSQL connection failed: %s — falling back to SQLite", e)
            DB_TYPE = "sqlite"

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the fraud_history table if it doesn't exist."""
    conn = get_db()
    cur = conn.cursor()

    if DB_TYPE == "postgresql":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                time_val FLOAT,
                amount FLOAT,
                v_features JSONB,
                prediction INTEGER,
                probability FLOAT,
                verdict TEXT
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                time_val FLOAT,
                amount FLOAT,
                v_features TEXT,
                prediction INTEGER,
                probability FLOAT,
                verdict TEXT
            )
        """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized [%s] %s",
                DB_TYPE,
                PG_CONFIG['dbname'] if DB_TYPE == 'postgresql' else DB_PATH)


init_db()

# ──────────────────────────────────────────────────
# AI Providers
# ──────────────────────────────────────────────────

SYSTEM_PROMPT_CHAT = """Bạn là chuyên gia bảo mật tài chính cấp cao của hệ thống Fraud Guard Pro.
Dữ liệu bạn đang phân tích dựa trên bộ dữ liệu giao dịch thẻ tín dụng của Châu Âu với các đặc trưng PCA (V1-V28).
Hãy trả lời ngắn gọn, chuyên nghiệp, tập trung vào khía cạnh an ninh tài chính bằng tiếng Việt."""

SYSTEM_PROMPT_ANALYZE = """Bạn là Giám đốc Phân tích Rủi ro (CRO) của Fraud Guard Pro.
Bạn nổi tiếng với khả năng phân tích dữ liệu database cực kỳ chi tiết, khoa học và không bao giờ tóm tắt quá mức.
Hãy trả lời bằng tiếng Việt."""


async def call_gemini(prompt: str, system_msg: str) -> str:
    """Call Google Gemini API."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=system_msg,
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return f"Lỗi Gemini: {str(e)}"


async def call_groq(prompt: str, system_msg: str) -> str:
    """Call Groq API."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq error: %s", e)
        return f"Lỗi Groq: {str(e)}"


async def call_ai(prompt: str, system_msg: str, provider: str = "gemini") -> str:
    """Route to the selected AI provider with automatic fallback."""
    try:
        if provider == "groq":
            return await call_groq(prompt, system_msg)
        else:
            return await call_gemini(prompt, system_msg)
    except Exception:
        # Fallback to the other provider
        fallback = "groq" if provider == "gemini" else "gemini"
        logger.warning("Provider %s failed, falling back to %s", provider, fallback)
        if fallback == "groq":
            return await call_groq(prompt, system_msg)
        else:
            return await call_gemini(prompt, system_msg)


# ──────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────


def _ph() -> str:
    """Return the SQL placeholder for the current DB type."""
    return "%s" if DB_TYPE == "postgresql" else "?"


def get_history_context() -> str:
    """Build a text summary from recent history for AI context."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT amount, verdict, probability, timestamp, model_used "
            "FROM fraud_history ORDER BY timestamp DESC LIMIT 20"
        )
        rows = cur.fetchall()

        cur.execute(
            "SELECT AVG(amount), COUNT(*) FROM fraud_history WHERE verdict = 'An toàn'"
        )
        safe_stats = cur.fetchone()

        cur.execute(
            "SELECT COUNT(*) FROM fraud_history WHERE verdict = 'Gian lận'"
        )
        fraud_count = cur.fetchone()[0]

        cur.close()
        conn.close()

        avg_safe = safe_stats[0] if safe_stats[0] else 0
        total_safe = safe_stats[1] if safe_stats[1] else 0

        ctx = "TỔNG QUAN HỆ THỐNG (Database):\n"
        ctx += f"- Tổng giao dịch an toàn: {total_safe}\n"
        ctx += f"- Số tiền trung bình an toàn: {avg_safe:.2f}$\n"
        ctx += f"- Tổng vụ gian lận đã chặn: {fraud_count}\n\n"
        ctx += "GIAO DỊCH GẦN ĐÂY:\n"
        for r in rows[:10]:
            ctx += f"- {r['timestamp']}: {r['amount']}$ [{r['model_used']}] → {r['verdict']} ({r['probability']*100:.1f}%)\n"

        return ctx
    except Exception as e:
        return f"DB context không khả dụng: {e}"


# ──────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


class PredictRequest(BaseModel):
    transaction: Transaction
    model_id: str = "random_forest_smote"


class ChatMessage(BaseModel):
    message: str
    provider: str = "gemini"
    context: Optional[dict] = None


class AnalysisRequest(BaseModel):
    data: dict
    prediction: int
    probability: float
    model_used: str = ""
    provider: str = "gemini"


# ──────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────

app = FastAPI(title="Fraud Guard Pro", version="2.0")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/models/")
def list_models():
    """Return available models for the frontend dropdown."""
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        models.append({
            "id": model_id,
            "display_name": info["display_name"],
            "description": info.get("description", ""),
        })
    return {"models": models, "default": "random_forest_smote"}


@app.post("/predict/")
def predict(req: PredictRequest):
    try:
        model_id = req.model_id
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' không tồn tại")

        ml_model = get_model(model_id)
        tx = req.transaction

        # Build DataFrame with correct column order
        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        values = [tx.Time] + [getattr(tx, f"V{i}") for i in range(1, 29)] + [tx.Amount]
        df = pd.DataFrame([values], columns=columns)

        # Scale only Amount & Time (scaler was fit on these 2 columns only)
        if SCALER is None:
            raise HTTPException(status_code=500, detail="Scaler chưa được load")
        df[SCALE_COLS] = SCALER.transform(df[SCALE_COLS])

        # Predict using the full DataFrame (already scaled where needed)
        prediction = int(ml_model.predict(df)[0])

        # Get probability
        if hasattr(ml_model, "predict_proba"):
            proba = ml_model.predict_proba(df)[0]
            probs = np.abs(proba)
            total = np.sum(probs)
            if total > 0:
                normalized = probs / total
            else:
                normalized = [0.5, 0.5]
            fraud_prob = float(np.clip(normalized[1], 0.0, 1.0)) if len(normalized) > 1 else 0.0
        else:
            fraud_prob = 1.0 if prediction == 1 else 0.0

        # Determine verdict
        if fraud_prob >= 0.75:
            verdict = "Gian lận"
        elif fraud_prob >= 0.35:
            verdict = "Nghi ngờ"
        else:
            verdict = "An toàn"

        # Save to DB
        try:
            v_dict = {f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)}
            v_json = json.dumps(v_dict)
            conn = get_db()
            cur = conn.cursor()
            p = _ph()
            cur.execute(
                f"""INSERT INTO fraud_history
                   (model_used, time_val, amount, v_features, prediction, probability, verdict)
                   VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p})""",
                (MODEL_REGISTRY[model_id]["display_name"], tx.Time, tx.Amount,
                 v_json, prediction, fraud_prob, verdict),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            logger.error("DB save failed: %s", db_err)

        return {
            "fraud_prediction": prediction,
            "probability": round(fraud_prob, 4),
            "verdict": verdict,
            "model_used": MODEL_REGISTRY[model_id]["display_name"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Predict error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history/")
def get_history():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, timestamp, amount, verdict, probability, model_used "
            "FROM fraud_history ORDER BY timestamp DESC LIMIT 50"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = []
        for r in rows:
            ts = r[1]  # timestamp
            if hasattr(ts, 'strftime'):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            result.append({
                "id": r[0],
                "timestamp": str(ts),
                "amount": r[2],
                "verdict": r[3],
                "probability": r[4],
                "model_used": r[5],
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{item_id}")
def get_history_item(item_id: int):
    try:
        conn = get_db()
        cur = conn.cursor()
        p = _ph()
        cur.execute(
            f"SELECT time_val, amount, v_features, prediction, probability, verdict, model_used "
            f"FROM fraud_history WHERE id = {p}",
            (item_id,),
        )
        r = cur.fetchone()
        cur.close()
        conn.close()

        if not r:
            raise HTTPException(status_code=404, detail="Not found")

        v_raw = r[2]  # v_features
        v_data = v_raw if isinstance(v_raw, dict) else json.loads(v_raw)
        return {
            "Time": r[0],
            "Amount": r[1],
            "V": v_data,
            "prediction": r[3],
            "probability": r[4],
            "verdict": r[5],
            "model_used": r[6],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/")
async def chat(msg: ChatMessage):
    context_str = ""
    if msg.context:
        context_str = f"\nNgữ cảnh giao dịch hiện tại: {json.dumps(msg.context, ensure_ascii=False)}"

    full_system = SYSTEM_PROMPT_CHAT + context_str
    response = await call_ai(msg.message, full_system, msg.provider)
    return {"response": response}


@app.post("/analyze/")
async def analyze(req: AnalysisRequest):
    status = "An Toàn" if req.prediction == 0 else "Gian lận"
    db_context = get_history_context()

    prompt = f"""BÁO CÁO PHÂN TÍCH FORENSIC TÀI CHÍNH
    ─────────────────────────────────
    MÔ HÌNH SỬ DỤNG: {req.model_used}
    TRẠNG THÁI: {status}
    ĐỘ TIN CẬY: {req.probability * 100:.2f}%
    DỮ LIỆU: {json.dumps(req.data, indent=2)}

    NGỮ CẢNH HỆ THỐNG:
    {db_context}

    YÊU CẦU PHÂN TÍCH CHI TIẾT:
    1. Đánh giá kỹ thuật các biến PCA (V1-V28), đặc biệt V14, V17, V12, V10, V4
    2. So sánh số tiền {req.data.get('Amount', 0)}$ với mức trung bình an toàn
    3. Đối chiếu với lịch sử giao dịch gần đây
    4. Kết luận chuyên gia chi tiết với thuật ngữ: PCA, Latent features, Anomaly Detection
    """

    response = await call_ai(prompt, SYSTEM_PROMPT_ANALYZE, req.provider)
    return {"analysis": response}


# ──────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
