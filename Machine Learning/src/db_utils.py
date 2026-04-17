"""
db_utils.py — PostgreSQL integration for Power BI dashboards.

Provides:
    • Table creation DDL (model_metrics, feature_importance, predictions_log)
    • Insert helpers for metrics and feature importance
    • Connection factory using environment variables

This module is OPTIONAL — the pipeline works without a database.
Power BI can also connect directly to the CSV exports in ``reports/``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# DDL statements
# ──────────────────────────────────────────────────

CREATE_MODEL_METRICS = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(64)   NOT NULL,
    sampling        VARCHAR(32)   NOT NULL,
    accuracy        FLOAT         NOT NULL,
    precision_score FLOAT         NOT NULL,
    recall          FLOAT         NOT NULL,
    f1_score        FLOAT         NOT NULL,
    auc_roc         FLOAT         DEFAULT 0,
    auc_pr          FLOAT         DEFAULT 0,
    created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_FEATURE_IMPORTANCE = """
CREATE TABLE IF NOT EXISTS feature_importance (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(64)   NOT NULL,
    sampling        VARCHAR(32)   NOT NULL,
    feature         VARCHAR(64)   NOT NULL,
    importance      FLOAT         NOT NULL,
    created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PREDICTIONS_LOG = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(64)   NOT NULL,
    features_json   TEXT          NOT NULL,
    prediction      INTEGER       NOT NULL,
    probability     FLOAT,
    created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_DDL = [CREATE_MODEL_METRICS, CREATE_FEATURE_IMPORTANCE, CREATE_PREDICTIONS_LOG]


# ──────────────────────────────────────────────────
# Connection factory
# ──────────────────────────────────────────────────


def get_connection() -> Any:
    """Create a psycopg2 connection from environment variables.

    Required env vars:
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

    Returns
    -------
    psycopg2 connection object

    Raises
    ------
    ImportError
        If psycopg2 is not installed.
    EnvironmentError
        If required env vars are missing.
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 is required for database operations. "
            "Install with: pip install psycopg2-binary"
        )

    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables for DB connection: {missing}"
        )

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    logger.info("Connected to PostgreSQL at %s:%s", os.getenv("DB_HOST"), os.getenv("DB_PORT"))
    return conn


# ──────────────────────────────────────────────────
# Table creation
# ──────────────────────────────────────────────────


def create_tables(conn: Optional[Any] = None) -> None:
    """Execute all CREATE TABLE statements.

    Parameters
    ----------
    conn : psycopg2 connection | None
        If None, creates one via ``get_connection()``.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    cur = conn.cursor()
    for ddl in ALL_DDL:
        cur.execute(ddl)
    conn.commit()
    cur.close()

    if own_conn:
        conn.close()

    logger.info("Database tables created / verified.")


# ──────────────────────────────────────────────────
# Data insertion
# ──────────────────────────────────────────────────


def insert_metrics(conn: Any, data: Dict[str, Any]) -> None:
    """Insert a single model evaluation result.

    Parameters
    ----------
    data : dict
        Keys: model_name, sampling, accuracy, precision_score,
              recall, f1_score, auc_roc, auc_pr
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_metrics
            (model_name, sampling, accuracy, precision_score,
             recall, f1_score, auc_roc, auc_pr)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            data["model_name"], data["sampling"],
            data["accuracy"], data["precision_score"],
            data["recall"], data["f1_score"],
            data.get("auc_roc", 0), data.get("auc_pr", 0),
        ),
    )
    conn.commit()
    cur.close()
    logger.info("Inserted metrics for %s (%s)", data["model_name"], data["sampling"])


def insert_feature_importance(
    conn: Any,
    model_name: str,
    sampling: str,
    features: List[str],
    importances: List[float],
) -> None:
    """Bulk-insert feature importance scores.

    Parameters
    ----------
    features : list[str]
        Feature names.
    importances : list[float]
        Corresponding importance values.
    """
    cur = conn.cursor()
    rows = [
        (model_name, sampling, feat, float(imp))
        for feat, imp in zip(features, importances)
    ]
    cur.executemany(
        """
        INSERT INTO feature_importance
            (model_name, sampling, feature, importance)
        VALUES (%s, %s, %s, %s)
        """,
        rows,
    )
    conn.commit()
    cur.close()
    logger.info(
        "Inserted %d feature importance rows for %s (%s)",
        len(rows), model_name, sampling,
    )


def insert_metrics_from_csv(csv_path: str, conn: Optional[Any] = None) -> None:
    """Read a model_comparison CSV and insert all rows into the DB.

    Parameters
    ----------
    csv_path : str
        Path to the ``model_comparison_*.csv`` file.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        insert_metrics(conn, {
            "model_name": row["model"],
            "sampling": row["sampling"],
            "accuracy": row["accuracy"],
            "precision_score": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "auc_roc": row.get("auc_roc", 0),
            "auc_pr": row.get("auc_pr", 0),
        })

    if own_conn:
        conn.close()

    logger.info("Bulk-inserted %d metric rows from %s", len(df), csv_path)
