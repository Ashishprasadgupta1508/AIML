import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

from aiml_engine.ai.database import get_connection


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved"

MODEL_PATH = MODEL_DIR / "time_model.joblib"


def load_completed_projects():

    query = """
        SELECT
            project_id,
            agency,
            ministry,
            sector,
            state,
            start_date,
            original_completion_date,
            original_cost,
            physical_progress,
            actual_completion_date
        FROM project
        WHERE actual_completion_date IS NOT NULL
          AND start_date IS NOT NULL
          AND original_completion_date IS NOT NULL;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    columns = [
        "project_id",
        "agency",
        "ministry",
        "sector",
        "state",
        "start_date",
        "original_completion_date",
        "original_cost",
        "physical_progress",
        "actual_completion_date",
    ]

    return pd.DataFrame(rows, columns=columns)


def build_features(df):

    df = df.copy()

    df["start_date"] = pd.to_datetime(
        df["start_date"]
    )

    df["original_completion_date"] = pd.to_datetime(
        df["original_completion_date"]
    )

    df["planned_duration_days"] = (
        df["original_completion_date"]
        - df["start_date"]
    ).dt.total_seconds() / 86400

    df["start_year"] = df["start_date"].dt.year
    df["start_month"] = df["start_date"].dt.month

    return df[
        [
            "agency",
            "ministry",
            "sector",
            "state",
            "start_date",
            "original_completion_date",
            "original_cost",
            "physical_progress",
            "start_year",
            "start_month",
            "planned_duration_days",
        ]
    ]


def calculate_actual_delay(df):

    actual_duration = (
        df["actual_completion_date"]
        - df["start_date"]
    ).dt.total_seconds() / 86400

    planned_duration = (
        df["original_completion_date"]
        - df["start_date"]
    ).dt.total_seconds() / 86400

    return actual_duration - planned_duration


def main():

    print("=" * 60)
    print("TIME MODEL CALIBRATION")
    print("=" * 60)

    df = load_completed_projects()

    print(
        f"\nCompleted projects: {len(df)}"
    )

    if len(df) < 10:

        print(
            "\nWARNING: Very small dataset."
        )

    model = load(MODEL_PATH)

    X = build_features(df)

    actual_delay = calculate_actual_delay(df)

    predictions = model.predict(X)

    errors = (
        actual_delay.values
        - predictions
    )

    absolute_errors = np.abs(errors)

    print("\nModel error statistics")
    print("-" * 40)

    print(
        f"MAE: "
        f"{np.mean(absolute_errors):.2f} days"
    )

    print(
        f"Median absolute error: "
        f"{np.median(absolute_errors):.2f} days"
    )

    print(
        f"75th percentile error: "
        f"{np.percentile(absolute_errors, 75):.2f} days"
    )

    print(
        f"90th percentile error: "
        f"{np.percentile(absolute_errors, 90):.2f} days"
    )

    print(
        f"95th percentile error: "
        f"{np.percentile(absolute_errors, 95):.2f} days"
    )

    print(
        f"Maximum absolute error: "
        f"{np.max(absolute_errors):.2f} days"
    )

    print("\nProject-level results")
    print("-" * 40)

    results = pd.DataFrame({
        "project_id": df["project_id"],
        "actual_delay_days": actual_delay.round(2),
        "ml_prediction_days": np.round(
            predictions,
            2
        ),
        "absolute_error_days": np.round(
            absolute_errors,
            2
        ),
    })

    print(
        results.to_string(index=False)
    )

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
