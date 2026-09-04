from pathlib import Path
from statistics import median

import pandas as pd
from joblib import load


# =========================================================
# MODEL CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved"

MODEL_PATH = MODEL_DIR / "time_model.joblib"

time_model = load(MODEL_PATH)


MIN_SIMILARITY = 0.60
SEARCH_LIMIT = 10

MODEL_ERROR_P90 = 875
MODEL_ERROR_P75 = 525


# =========================================================
# DYNAMIC TIME WARNING GENERATOR
# =========================================================

def _build_time_warning(
    planned_duration_days,
    final_prediction,
    ml_prediction,
    historical_prediction,
    historical_delays,
    average_similarity,
    confidence,
    range_min,
    range_max,
):
    """
    Generate a concise, deterministic and project-specific
    schedule warning.

    Warning uses:
        - predicted delay
        - planned project duration
        - ML prediction
        - historical prediction
        - historical evidence
        - prediction range
        - similarity
        - confidence

    Maximum two sentences.

    Same project + same data produces the same warning.

    Runtime is READ-ONLY.
    """

    historical_count = len(historical_delays)

    final_prediction = max(
        0.0,
        float(final_prediction),
    )

    ml_prediction = float(ml_prediction)

    if historical_prediction is not None:
        historical_prediction = float(
            historical_prediction
        )

    average_similarity = float(
        average_similarity or 0.0
    )

    range_min = float(range_min)
    range_max = float(range_max)

    # =====================================================
    # 1. NO HISTORICAL EVIDENCE
    # =====================================================

    if historical_prediction is None:

        if final_prediction > 1095:
            return (
                f"The projected delay of approximately "
                f"{final_prediction:.0f} days indicates a prolonged "
                "schedule risk. Comparable completed-project evidence "
                "is insufficient to validate the exact timeline."
            )

        elif final_prediction > 730:
            return (
                f"The schedule outlook indicates a significant delay "
                f"of about {final_prediction:.0f} days, but comparable "
                "historical evidence is limited."
            )

        elif final_prediction > 365:
            return (
                f"An estimated delay of approximately "
                f"{final_prediction:.0f} days indicates meaningful "
                "schedule slippage, while historical evidence is "
                "insufficient for strong validation."
            )

        elif final_prediction > 90:
            return (
                f"Some schedule slippage of around "
                f"{final_prediction:.0f} days is anticipated, although "
                "historical evidence is unavailable for validation."
            )

        return (
            "The current model does not indicate a major schedule "
            "problem, but the absence of comparable completed-project "
            "evidence limits confidence in the forecast."
        )

    # =====================================================
    # 2. DELAY SEVERITY
    # =====================================================

    if final_prediction > 1825:

        primary = (
            f"The projected delay of approximately "
            f"{final_prediction:.0f} days represents an exceptionally "
            "prolonged extension beyond the planned timeline."
        )

    elif final_prediction > 1095:

        primary = (
            f"The schedule outlook indicates a prolonged delay of about "
            f"{final_prediction:.0f} days, suggesting substantial pressure "
            "on the original completion plan."
        )

    elif final_prediction > 730:

        primary = (
            f"The expected delay of approximately "
            f"{final_prediction:.0f} days is likely to materially affect "
            "the original completion plan."
        )

    elif final_prediction > 365:

        primary = (
            f"The current schedule outlook indicates a meaningful "
            f"extension of around {final_prediction:.0f} days beyond "
            "the planned timeline."
        )

    elif final_prediction > 180:

        primary = (
            f"The project is showing noticeable schedule slippage, "
            f"with approximately {final_prediction:.0f} days of delay "
            "projected."
        )

    elif final_prediction > 90:

        primary = (
            f"The assessment indicates moderate schedule slippage of "
            f"about {final_prediction:.0f} days that should be monitored."
        )

    elif final_prediction > 30:

        primary = (
            f"A limited schedule extension of approximately "
            f"{final_prediction:.0f} days is currently anticipated."
        )

    elif final_prediction > 0:

        primary = (
            f"Some schedule slippage of approximately "
            f"{final_prediction:.0f} days is currently projected."
        )

    else:

        primary = (
            "The current assessment does not indicate a significant "
            "delay against the planned schedule."
        )

    # =====================================================
    # 3. MODEL DISAGREEMENT
    # =====================================================

    disagreement = abs(
        ml_prediction - historical_prediction
    )

    # =====================================================
    # 4. PREDICTION RANGE
    # =====================================================

    range_width = max(
        0.0,
        range_max - range_min,
    )

    if planned_duration_days > 0:

        uncertainty_ratio = (
            range_width
            / planned_duration_days
        )

        delay_ratio = (
            final_prediction
            / planned_duration_days
        )

    else:

        uncertainty_ratio = 0.0
        delay_ratio = 0.0

    # =====================================================
    # 5. STRONGEST SUPPORTING SIGNAL
    # =====================================================

    # -----------------------------------------------------
    # Very large disagreement
    # -----------------------------------------------------

    if disagreement > 1460:

        support = (
            "The ML and historical estimates differ by more than "
            "four years, making the exact completion timeline "
            "highly uncertain."
        )

    elif disagreement > 1095:

        support = (
            "The ML and historical estimates differ by more than "
            "three years, creating substantial uncertainty around "
            "the expected timeline."
        )

    elif disagreement > 730:

        support = (
            "The ML and historical estimates differ substantially, "
            "so the projected completion timeline should be "
            "validated carefully."
        )

    elif disagreement > 365:

        support = (
            "The model and historical evidence point to noticeably "
            "different delay outcomes, reducing confidence in the "
            "exact timeline."
        )

    # -----------------------------------------------------
    # Very broad prediction range
    # -----------------------------------------------------

    elif uncertainty_ratio > 3:

        support = (
            "The possible delay outcomes span a very broad range "
            "relative to the original project duration, limiting "
            "forecast precision."
        )

    elif uncertainty_ratio > 2:

        support = (
            "The expected delay range is wide relative to the "
            "planned execution period, adding substantial uncertainty "
            "to the forecast."
        )

    elif uncertainty_ratio > 1:

        support = (
            "The expected schedule range is substantial relative "
            "to the planned execution period, making milestone "
            "validation important."
        )

    elif uncertainty_ratio > 0.5:

        support = (
            "The projected schedule carries moderate uncertainty "
            "relative to the original execution period."
        )

    # -----------------------------------------------------
    # Historical evidence
    # -----------------------------------------------------

    elif historical_count < 3:

        support = (
            f"Only {historical_count} comparable completed "
            "project(s) were available, so the historical signal "
            "should be interpreted cautiously."
        )

    elif average_similarity < 0.65:

        support = (
            "The available historical projects have relatively "
            "weak similarity to the current project, limiting "
            "validation strength."
        )

    elif average_similarity >= 0.85:

        support = (
            f"The schedule outlook is supported by {historical_count} "
            "highly similar completed projects, providing strong "
            "historical evidence."
        )

    elif average_similarity >= 0.80:

        support = (
            f"{historical_count} comparable completed projects provide "
            "strong historical support for the current schedule outlook."
        )

    elif average_similarity >= 0.70:

        support = (
            f"{historical_count} historical comparisons provide useful "
            "support for the current schedule assessment."
        )

    else:

        support = (
            f"{historical_count} historical comparisons provide some "
            "support, but their similarity limits validation strength."
        )

    # =====================================================
    # 6. LOW CONFIDENCE
    # =====================================================

    if confidence == "LOW":

        # Don't add confidence repeatedly when a major uncertainty
        # signal has already been selected.

        if (
            disagreement <= 365
            and uncertainty_ratio <= 2
            and historical_count >= 3
        ):

            support = (
                f"{support.rstrip('.')} ; overall confidence "
                "in the exact delay estimate remains low."
            )

    # =====================================================
    # 7. MEDIUM CONFIDENCE
    # =====================================================

    elif confidence == "MEDIUM":

        if (
            disagreement <= 365
            and uncertainty_ratio <= 1
        ):

            support = (
                f"{support.rstrip('.')}, with moderate confidence "
                "in the projected timeline."
            )

    # =====================================================
    # 8. VERY LARGE DELAY RELATIVE TO PLAN
    # =====================================================

    if (
        delay_ratio > 2
        and disagreement <= 365
        and uncertainty_ratio <= 2
        and historical_count >= 3
        and confidence != "LOW"
    ):

        support = (
            "The projected delay is more than twice the original "
            "planned execution duration, indicating substantial "
            "schedule exposure."
        )

    # =====================================================
    # 9. FINAL WARNING
    # =====================================================

    if primary == support:
        return primary

    return f"{primary} {support}"


# =========================================================
# MAIN TIME PREDICTION
# =========================================================

def predict_time(
    project,
    completed_similar_projects=None,
):
    """
    Predict project delay using:

    1. Existing ML model
    2. Similar completed historical projects
    3. Historical uncertainty/range
    4. Dynamic project-specific warning

    Runtime is READ-ONLY.

    No project or ai_prediction table is modified.
    """

    # =====================================================
    # PROJECT DATES
    # =====================================================

    start_date = pd.to_datetime(
        project.get("start_date"),
        errors="coerce",
    )

    completion_date = pd.to_datetime(
        project.get("original_completion_date"),
        errors="coerce",
    )

    if (
        pd.isna(start_date)
        or pd.isna(completion_date)
    ):

        raise ValueError(
            "start_date and original_completion_date "
            "are required for time prediction."
        )

    planned_duration_days = (
        completion_date - start_date
    ).days

    # =====================================================
    # ML FEATURES
    # =====================================================

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

    # =====================================================
    # ML PREDICTION
    # =====================================================

    ml_prediction = float(
        time_model.predict(features)[0]
    )

    # =====================================================
    # HISTORICAL PREDICTION
    # =====================================================

    historical = _get_historical_delay_estimate(
        project,
        completed_similar_projects,
    )

    historical_prediction = historical[
        "predicted_delay_days"
    ]

    historical_delays = historical[
        "delays"
    ]

    avg_similarity = historical[
        "average_similarity"
    ]

    # =====================================================
    # COMBINE ML + HISTORICAL EVIDENCE
    # =====================================================

    if historical_prediction is not None:

        if avg_similarity >= 0.75:

            historical_weight = 0.60

        elif avg_similarity >= 0.65:

            historical_weight = 0.40

        else:

            historical_weight = 0.20

        ml_weight = (
            1.0 - historical_weight
        )

        final_prediction = (
            ml_prediction * ml_weight
            + historical_prediction
            * historical_weight
        )

    else:

        final_prediction = ml_prediction

    final_prediction = max(
        0,
        final_prediction,
    )

    # =====================================================
    # PREDICTION RANGE
    # =====================================================

    if len(historical_delays) >= 5:

        historical_q25 = _percentile(
            historical_delays,
            25,
        )

        historical_q75 = _percentile(
            historical_delays,
            75,
        )

        uncertainty = MODEL_ERROR_P90

        range_min = max(
            0,
            min(
                final_prediction - uncertainty,
                historical_q25,
            ),
        )

        range_max = max(
            final_prediction + uncertainty,
            historical_q75,
        )

    else:

        uncertainty = MODEL_ERROR_P90

        range_min = max(
            0,
            final_prediction - uncertainty,
        )

        range_max = (
            final_prediction
            + uncertainty
        )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = _calculate_confidence(
        historical_count=len(
            historical_delays
        ),
        average_similarity=avg_similarity,
        ml_prediction=ml_prediction,
        historical_prediction=historical_prediction,
    )

    # =====================================================
    # DYNAMIC WARNING
    # =====================================================

    warning = _build_time_warning(
        planned_duration_days=planned_duration_days,
        final_prediction=final_prediction,
        ml_prediction=ml_prediction,
        historical_prediction=historical_prediction,
        historical_delays=historical_delays,
        average_similarity=avg_similarity,
        confidence=confidence,
        range_min=range_min,
        range_max=range_max,
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "predicted_delay_days": round(
            final_prediction,
            2,
        ),

        "expected_delay_range": {
            "min_days": round(
                range_min,
                2,
            ),
            "max_days": round(
                range_max,
                2,
            ),
        },

        "planned_duration_days": (
            planned_duration_days
        ),

        "ml_predicted_delay_days": round(
            ml_prediction,
            2,
        ),

        "historical_predicted_delay_days": (
            round(
                historical_prediction,
                2,
            )
            if historical_prediction is not None
            else None
        ),

        "historical_projects_used": len(
            historical_delays
        ),

        "average_similarity": round(
            avg_similarity,
            4,
        ),

        "confidence": confidence,

        "warning": warning,
    }


# =========================================================
# HISTORICAL DELAY ESTIMATION
# =========================================================

def _get_historical_delay_estimate(
    project,
    similar_projects=None,
):

    if similar_projects is None:
        similar_projects = []

    similar_projects = [
        item
        for item in similar_projects
        if float(
            item.get("similarity", 0)
        ) >= MIN_SIMILARITY
        and item.get("project_id")
        != project.get("project_id")
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
        item["project_id"]: float(
            item["similarity"]
        )
        for item in similar_projects
    }

    placeholders = ",".join(
        ["%s"] * len(project_ids)
    )

    # =====================================================
    # READ-ONLY DATABASE QUERY
    # =====================================================

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

    from aiml_engine.ai.database import get_connection

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                project_ids,
            )

            rows = cur.fetchall()

    # =====================================================
    # CALCULATE HISTORICAL DELAYS
    # =====================================================

    historical = []

    for (
        project_id,
        start_date,
        original_completion_date,
        actual_completion_date,
    ) in rows:

        planned_duration = (
            original_completion_date
            - start_date
        ).days

        actual_duration = (
            actual_completion_date
            - start_date
        ).days

        delay_days = (
            actual_duration
            - planned_duration
        )

        historical.append({
            "project_id": project_id,

            "similarity": similarity_map.get(
                project_id,
                0.0,
            ),

            "delay_days": delay_days,
        })

    if not historical:

        return {
            "predicted_delay_days": None,
            "delays": [],
            "average_similarity": 0.0,
        }

    # =====================================================
    # SIMILARITY-WEIGHTED HISTORICAL PREDICTION
    # =====================================================

    total_weight = 0.0
    weighted_delay = 0.0

    for item in historical:

        similarity = item["similarity"]

        weight = similarity ** 4

        weighted_delay += (
            item["delay_days"]
            * weight
        )

        total_weight += weight

    if total_weight == 0:

        predicted_delay = median(
            item["delay_days"]
            for item in historical
        )

    else:

        predicted_delay = (
            weighted_delay
            / total_weight
        )

    # =====================================================
    # HISTORICAL DELAY VALUES
    # =====================================================

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

        "average_similarity": (
            average_similarity
        ),
    }


# =========================================================
# CONFIDENCE CALCULATION
# =========================================================

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


# =========================================================
# PERCENTILE
# =========================================================

def _percentile(
    values,
    percentile,
):

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
        len(values) - 1,
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


# =========================================================
# PROJECT TEXT
# =========================================================

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