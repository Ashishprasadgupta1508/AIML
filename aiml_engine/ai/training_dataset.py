import pandas as pd
from aiml_engine.ai.database import get_connection


def load_training_data():
    query = """
        SELECT
            project_id,
            agency,
            ministry,
            sector,
            state,
            start_date,
            original_completion_date,
            revised_completion_date,
            original_cost,
            revised_cost,
            physical_progress,
            cumulative_expenditure,
            actual_completion_date
        FROM project
        WHERE actual_completion_date IS NOT NULL
          AND original_cost IS NOT NULL
          AND cumulative_expenditure IS NOT NULL
          AND start_date IS NOT NULL
          AND original_completion_date IS NOT NULL;
    """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    # -----------------------------
    # TARGET 1: COST OVER-RUN
    # -----------------------------
    df["cost_overrun_percent"] = (
        (df["cumulative_expenditure"] - df["original_cost"])
        / df["original_cost"]
    ) * 100

    # -----------------------------
    # TARGET 2: TIME DELAY
    # -----------------------------
    df["planned_duration_days"] = (
        df["original_completion_date"] - df["start_date"]
    ).dt.total_seconds() / 86400

    df["actual_duration_days"] = (
        df["actual_completion_date"] - df["start_date"]
    ).dt.total_seconds() / 86400

    df["delay_days"] = (
        df["actual_duration_days"]
        - df["planned_duration_days"]
    )

    # ----------------------------------------
    # LEAKAGE PREVENTION
    # ----------------------------------------
    # These columns reveal the final outcome.
    # They must NOT be model input features.
    leakage_columns = [
        "actual_completion_date",
        "cumulative_expenditure",
        "cost_overrun_percent",
        "actual_duration_days",
        "delay_days",
    ]

    feature_columns = [
        "project_id",
        "agency",
        "ministry",
        "sector",
        "state",
        "start_date",
        "original_completion_date",
        "revised_completion_date",
        "original_cost",
        "revised_cost",
        "physical_progress",
    ]

    X = df[feature_columns].copy()

    y_cost = df["cost_overrun_percent"].copy()
    y_time = df["delay_days"].copy()

    return X, y_cost, y_time, df


if __name__ == "__main__":
    X, y_cost, y_time, df = load_training_data()

    print("\n==============================")
    print("TRAINING DATASET")
    print("==============================")

    print("Projects:", len(df))

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nCost target:")
    print(y_cost.describe())

    print("\nTime target:")
    print(y_time.describe())

    print("\nFirst 5 records:")
    print(X.head())