import numpy as np
import pandas as pd

from aiml_engine.ai.database import get_connection


def load_completed_projects():

    query = """
        SELECT
            project_id,
            original_cost,
            cumulative_expenditure
        FROM project
        WHERE actual_completion_date IS NOT NULL
          AND original_cost IS NOT NULL
          AND original_cost > 0
          AND cumulative_expenditure IS NOT NULL;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    columns = [
        "project_id",
        "original_cost",
        "cumulative_expenditure",
    ]

    return pd.DataFrame(
        rows,
        columns=columns
    )


def main():

    print("=" * 60)
    print("COST PREDICTION CALIBRATION")
    print("=" * 60)

    df = load_completed_projects()

    print(
        f"\nCompleted projects: {len(df)}"
    )

    df["original_cost"] = (
        df["original_cost"]
        .astype(float)
    )

    df["cumulative_expenditure"] = (
        df["cumulative_expenditure"]
        .astype(float)
    )

    # Actual historical cost overrun
    df["actual_overrun_percent"] = (
        (
            df["cumulative_expenditure"]
            - df["original_cost"]
        )
        / df["original_cost"]
    ) * 100

    print("\nHistorical cost-overrun statistics")
    print("-" * 40)

    values = df[
        "actual_overrun_percent"
    ].values

    print(
        f"Mean overrun: "
        f"{np.mean(values):.2f}%"
    )

    print(
        f"Median overrun: "
        f"{np.median(values):.2f}%"
    )

    print(
        f"Minimum overrun: "
        f"{np.min(values):.2f}%"
    )

    print(
        f"Maximum overrun: "
        f"{np.max(values):.2f}%"
    )

    print(
        f"25th percentile: "
        f"{np.percentile(values, 25):.2f}%"
    )

    print(
        f"75th percentile: "
        f"{np.percentile(values, 75):.2f}%"
    )

    print(
        f"90th percentile: "
        f"{np.percentile(values, 90):.2f}%"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(values, 95):.2f}%"
    )

    print("\nProject-level historical outcomes")
    print("-" * 40)

    results = df[
        [
            "project_id",
            "original_cost",
            "cumulative_expenditure",
            "actual_overrun_percent",
        ]
    ].copy()

    results["actual_overrun_percent"] = (
        results["actual_overrun_percent"]
        .round(2)
    )

    print(
        results.to_string(index=False)
    )

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()

