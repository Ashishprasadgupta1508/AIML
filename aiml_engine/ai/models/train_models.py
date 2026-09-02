import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from aiml_engine.ai.training_dataset import load_training_data


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "aiml_engine" / "ai" / "models" / "saved"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train():

    X, y_cost, y_time, df = load_training_data()

    # ----------------------------------------------
    # REMOVE IDENTIFIER + FUTURE INFORMATION
    # ----------------------------------------------

    safe_features = [
        "agency",
        "ministry",
        "sector",
        "state",
        "start_date",
        "original_completion_date",
        "original_cost",
        "physical_progress",
    ]

    X = X[safe_features].copy()

    # ----------------------------------------------
    # DATE FEATURES
    # ----------------------------------------------

    date_columns = [
        "start_date",
        "original_completion_date",
    ]

    for column in date_columns:
        X[column] = pd.to_datetime(
            X[column],
            errors="coerce"
        )

    X["start_year"] = X["start_date"].dt.year
    X["start_month"] = X["start_date"].dt.month

    X["planned_duration_days"] = (
        X["original_completion_date"]
        - X["start_date"]
    ).dt.days

    X.drop(
        columns=date_columns,
        inplace=True
    )

    # ----------------------------------------------
    # FEATURE TYPES
    # ----------------------------------------------

    categorical_features = [
        "agency",
        "ministry",
        "sector",
        "state",
    ]

    numeric_features = [
        "original_cost",
        "physical_progress",
        "start_year",
        "start_month",
        "planned_duration_days",
    ]

    # ----------------------------------------------
    # PREPROCESSOR
    # ----------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # ----------------------------------------------
    # TRAIN / TEST SPLIT
    # ----------------------------------------------

    X_train, X_test, y_cost_train, y_cost_test, \
        y_time_train, y_time_test = train_test_split(
            X,
            y_cost,
            y_time,
            test_size=0.20,
            random_state=42
        )

    # ----------------------------------------------
    # COST MODEL
    # ----------------------------------------------

    cost_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42
            )
        )
    ])

    cost_model.fit(
        X_train,
        y_cost_train
    )

    cost_predictions = cost_model.predict(
        X_test
    )

    cost_mae = mean_absolute_error(
        y_cost_test,
        cost_predictions
    )

    cost_r2 = r2_score(
        y_cost_test,
        cost_predictions
    )

    # ----------------------------------------------
    # TIME MODEL
    # ----------------------------------------------

    time_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42
            )
        )
    ])

    time_model.fit(
        X_train,
        y_time_train
    )

    time_predictions = time_model.predict(
        X_test
    )

    time_mae = mean_absolute_error(
        y_time_test,
        time_predictions
    )

    time_r2 = r2_score(
        y_time_test,
        time_predictions
    )

    # ----------------------------------------------
    # SAVE MODELS
    # ----------------------------------------------

    joblib.dump(
        cost_model,
        MODEL_DIR / "cost_model.joblib"
    )

    joblib.dump(
        time_model,
        MODEL_DIR / "time_model.joblib"
    )

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    print("\n====================================")
    print("MODEL TRAINING COMPLETE")
    print("====================================")

    print("\nDataset:")
    print(f"Total projects: {len(X)}")
    print(f"Training projects: {len(X_train)}")
    print(f"Testing projects: {len(X_test)}")

    print("\nCOST MODEL")
    print("------------------------------------")
    print(f"MAE: {cost_mae:.2f}%")
    print(f"R² : {cost_r2:.4f}")

    print("\nTIME MODEL")
    print("------------------------------------")
    print(f"MAE: {time_mae:.2f} days")
    print(f"R² : {time_r2:.4f}")

    print("\nModels saved:")
    print(MODEL_DIR / "cost_model.joblib")
    print(MODEL_DIR / "time_model.joblib")


if __name__ == "__main__":
    train()
