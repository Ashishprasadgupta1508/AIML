from pathlib import Path
from statistics import median

import pandas as pd
from joblib import load




BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved"

MODEL_PATH = MODEL_DIR / "time_model.joblib"

time_model = load(MODEL_PATH)

MIN_SIMILARITY = 0.60
SEARCH_LIMIT = 10
MODEL_ERROR_P90 = 875
MODEL_ERROR_P75 = 525


def predict_time(project, completed_similar_projects=None):
    """
    Predict project delay using:
    1. Existing ML model
    2. Similar completed historical projects
    3. Historical uncertainty/range

    Runtime is READ-ONLY.
    No project or ai_prediction table is modified.
    """

    start_date = pd.to_datetime(
        project.get("start_date"),
        errors="coerce"
    )

    completion_date = pd.to_datetime(
        project.get("original_completion_date"),
        errors="coerce"
    )

    if pd.isna(start_date) or pd.isna(completion_date):
        raise ValueError(
            "start_date and original_completion_date "
            "are required for time prediction."
        )

    planned_duration_days = (
        completion_date - start_date
    ).days

    features = pd.DataFrame([{
        "agency": project.get("agency"),
        "ministry": project.get("ministry"),
        "sector": project.get("sector"),
        "state": project.get("state"),
        "start_date": start_date,
        "original_completion_date": completion_date,
        "original_cost": project.get("original_cost"),
        "physical_progress": project.get("physical_progress"),
        "start_year": start_date.year,
        "start_month": start_date.month,
        "planned_duration_days": planned_duration_days,
    }])

    ml_prediction = float(
        time_model.predict(features)[0]
    )

    historical = _get_historical_delay_estimate(
        project,
        completed_similar_projects
    )
    historical_prediction = historical["predicted_delay_days"]
    historical_delays = historical["delays"]

    avg_similarity = historical["average_similarity"]

    # --------------------------------------------------
    # Combine ML + historical evidence
    # --------------------------------------------------

    if historical_prediction is not None:

        if avg_similarity >= 0.75:
            historical_weight = 0.60
        elif avg_similarity >= 0.65:
            historical_weight = 0.40
        else:
            historical_weight = 0.20

        ml_weight = 1.0 - historical_weight

        final_prediction = (
            ml_prediction * ml_weight
            + historical_prediction * historical_weight
        )

    else:
        final_prediction = ml_prediction

    final_prediction = max(0, final_prediction)

    # --------------------------------------------------
    # Prediction range
    # --------------------------------------------------

    # --------------------------------------------------
    # Calibrated prediction range
    # --------------------------------------------------

    if len(historical_delays) >= 5:

        historical_q25 = _percentile(
            historical_delays,
            25
        )

        historical_q75 = _percentile(
            historical_delays,
            75
        )

        # Base model uncertainty comes from the
        # empirical 90th percentile absolute error
        # calculated from completed projects.
        uncertainty = MODEL_ERROR_P90

        range_min = max(
            0,
            min(
                final_prediction - uncertainty,
                historical_q25
            )
        )

        range_max = max(
            final_prediction + uncertainty,
            historical_q75
        )

    else:

        # Not enough historical evidence.
        uncertainty = MODEL_ERROR_P90

        range_min = max(
            0,
            final_prediction - uncertainty
        )

        range_max = (
            final_prediction + uncertainty
        )


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    confidence = _calculate_confidence(
        historical_count=len(historical_delays),
        average_similarity=avg_similarity,
        ml_prediction=ml_prediction,
        historical_prediction=historical_prediction,
    )

    # --------------------------------------------------
    # Warning
    # --------------------------------------------------

    warning = None

    if historical_prediction is not None:

        disagreement = abs(
            ml_prediction - historical_prediction
        )

        if disagreement > 730:
            warning = (
                "ML prediction and historical similar-project "
                "evidence show large disagreement."
            )

        elif disagreement > 365:
            warning = (
                "ML prediction and historical evidence "
                "show moderate disagreement."
            )

    return {
        "predicted_delay_days": round(
            final_prediction,
            2
        ),

        "expected_delay_range": {
            "min_days": round(
                range_min,
                2
            ),
            "max_days": round(
                range_max,
                2
            ),
        },

        "planned_duration_days": planned_duration_days,

        "ml_predicted_delay_days": round(
            ml_prediction,
            2
        ),

        "historical_predicted_delay_days": (
            round(
                historical_prediction,
                2
            )
            if historical_prediction is not None
            else None
        ),

        "historical_projects_used": len(
            historical_delays
        ),

        "average_similarity": round(
            avg_similarity,
            4
        ),

        "confidence": confidence,

        "warning": warning,
    }


def _get_historical_delay_estimate(
    project,
    similar_projects=None
):

    if similar_projects is None:
        similar_projects = []

    similar_projects = [
        item
        for item in similar_projects
        if float(item.get("similarity", 0)) >= MIN_SIMILARITY
        and item.get("project_id") != project.get("project_id")
    ]

    if not similar_projects:
        return {
            "predicted_delay_days": None,
            "delays": [],
            "average_similarity": 0.0,
        }

    project_ids = [
        item["project_id"]
        for item in similar_projects
    ]

    similarity_map = {
        item["project_id"]: float(item["similarity"])
        for item in similar_projects
    }

    placeholders = ",".join(["%s"] * len(project_ids))

    query = f"""
        SELECT
            project_id,
            start_date,
            original_completion_date,
            actual_completion_date
        FROM project
        WHERE project_id IN ({placeholders})
          AND start_date IS NOT NULL
          AND original_completion_date IS NOT NULL
          AND actual_completion_date IS NOT NULL
    """

    # IMPORTANT:
    # We still need one READ-ONLY database query here
    # to obtain actual historical delay outcomes.
    from aiml_engine.ai.database import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, project_ids)
            rows = cur.fetchall()

    historical = []

    for (
        project_id,
        start_date,
        original_completion_date,
        actual_completion_date,
    ) in rows:

        planned_duration = (
            original_completion_date - start_date
        ).days

        actual_duration = (
            actual_completion_date - start_date
        ).days

        delay_days = actual_duration - planned_duration

        historical.append({
            "project_id": project_id,
            "similarity": similarity_map.get(
                project_id,
                0.0
            ),
            "delay_days": delay_days,
        })

    if not historical:
        return {
            "predicted_delay_days": None,
            "delays": [],
            "average_similarity": 0.0,
        }

    total_weight = 0.0
    weighted_delay = 0.0

    for item in historical:

        similarity = item["similarity"]
        weight = similarity ** 4

        weighted_delay += (
            item["delay_days"] * weight
        )

        total_weight += weight

    if total_weight == 0:

        predicted_delay = median(
            item["delay_days"]
            for item in historical
        )

    else:

        predicted_delay = (
            weighted_delay / total_weight
        )

    delays = [
        item["delay_days"]
        for item in historical
    ]

    average_similarity = (
        sum(
            item["similarity"]
            for item in historical
        )
        / len(historical)
    )

    return {
        "predicted_delay_days": float(
            predicted_delay
        ),
        "delays": delays,
        "average_similarity": average_similarity,
    }

def _calculate_confidence(
    historical_count,
    average_similarity,
    ml_prediction,
    historical_prediction,
):

    if historical_prediction is None:

        return "LOW"

    disagreement = abs(
        ml_prediction
        - historical_prediction
    )

    if (
        historical_count >= 5
        and average_similarity >= 0.75
        and disagreement <= 365
    ):
        return "HIGH"

    if (
        historical_count >= 3
        and average_similarity >= 0.65
        and disagreement <= 730
    ):
        return "MEDIUM"

    return "LOW"


def _percentile(values, percentile):

    values = sorted(values)

    if not values:
        return 0

    if len(values) == 1:
        return values[0]

    index = (
        (len(values) - 1)
        * percentile
        / 100
    )

    lower_index = int(index)
    upper_index = min(
        lower_index + 1,
        len(values) - 1
    )

    fraction = (
        index - lower_index
    )

    return (
        values[lower_index]
        + (
            values[upper_index]
            - values[lower_index]
        )
        * fraction
    )


def _build_project_text(project):

    return f"""
    Project Name: {project.get('project_name', '')}
    Agency: {project.get('agency', '')}
    Ministry: {project.get('ministry', '')}
    Sector: {project.get('sector', '')}
    State: {project.get('state', '')}
    Original Cost: {project.get('original_cost', '')}
    Physical Progress: {project.get('physical_progress', '')}
    Start Date: {project.get('start_date', '')}
    Original Completion Date:
        {project.get('original_completion_date', '')}
    """.strip()