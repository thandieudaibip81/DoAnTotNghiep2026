"""
drift_monitor.py — Data Drift Detection with Evidently AI.

Compares training data (reference) against production/current data
to detect distribution shifts that may degrade model performance.

Usage:
    # Normal mode (compare train vs current production data):
    python -m src.drift_monitor

    # Simulate drift for demo:
    python -m src.drift_monitor --simulate

Output:
    reports/data_drift_report.html  — Interactive HTML dashboard
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Feature columns used by the model
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_reference_data() -> pd.DataFrame:
    """Load original training data as the reference baseline."""
    logger.info("Loading reference data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return df[FEATURE_COLS]


def simulate_drifted_data(reference: pd.DataFrame, n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic 'production' data with intentional drift for demo.

    Simulates a real-world scenario where:
    - Transaction amounts suddenly increase (e.g. new market/currency)
    - Some PCA features shift distribution (e.g. new fraud patterns)
    """
    rng = np.random.default_rng(seed=42)

    # Sample from reference as base
    current = reference.sample(n=n_samples, random_state=42).copy()

    # ── Drift 1: Amount increases significantly (x3 mean shift) ──
    current["Amount"] = current["Amount"] * rng.uniform(2.0, 5.0, size=n_samples)
    logger.info("💥 Injected drift: Amount multiplied by 2x-5x")

    # ── Drift 2: V1-V4 shift (simulating new fraud patterns) ──
    for col in ["V1", "V2", "V3", "V4"]:
        original_std = reference[col].std()
        current[col] = current[col] + rng.normal(loc=original_std * 1.5, scale=original_std * 0.5, size=n_samples)
    logger.info("💥 Injected drift: V1-V4 shifted by 1.5σ")

    # ── Drift 3: V14 inverted (key fraud indicator) ──
    current["V14"] = -current["V14"] + rng.normal(0, 0.3, size=n_samples)
    logger.info("💥 Injected drift: V14 distribution inverted")

    return current


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Generate Evidently Data Drift report comparing reference vs current data.

    Parameters
    ----------
    reference : pd.DataFrame
        Training data (baseline distribution).
    current : pd.DataFrame
        Production/current data to check for drift.
    output_path : Path | None
        Where to save the HTML report; defaults to reports/data_drift_report.html.

    Returns
    -------
    Path
        Path to the saved HTML report.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "data_drift_report.html"

    logger.info("Generating drift report (reference: %d rows, current: %d rows)...",
                len(reference), len(current))

    report = Report([
        DataDriftPreset(),
    ])

    # Evidently 0.7: run() returns a Snapshot object with save_html()
    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    snapshot.save_html(str(output_path))
    logger.info("✅ Data Drift report saved → %s", output_path)

    print("\n" + "=" * 60)
    print("📊 DATA DRIFT REPORT GENERATED")
    print("=" * 60)
    print(f"   Reference data:  {len(reference):,} rows")
    print(f"   Current data:    {len(current):,} rows")
    print(f"   Features:        {len(FEATURE_COLS)}")
    print(f"   Report saved:    {output_path}")
    print("=" * 60)

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Data Drift Monitoring")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Inject synthetic drift into production data for demo purposes",
    )
    parser.add_argument(
        "--current-csv", type=str, default=None,
        help="Path to a CSV file containing current/production data",
    )
    args = parser.parse_args()

    # 1. Load reference (training) data
    reference = load_reference_data()

    # 2. Get current data
    if args.current_csv:
        logger.info("Loading current data from %s", args.current_csv)
        current = pd.read_csv(args.current_csv)[FEATURE_COLS]
    elif args.simulate:
        print("\n🧪 SIMULATION MODE: Injecting artificial drift for demo...\n")
        current = simulate_drifted_data(reference)
    else:
        # Default: sample from training data (should show NO drift)
        print("\n📊 NORMAL MODE: Comparing training data against itself (expect no drift)...\n")
        current = reference.sample(n=5000, random_state=99)

    # 3. Generate report
    report_path = generate_drift_report(reference, current)

    print(f"\n🔗 Open the report in your browser:")
    print(f"   file://{report_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
