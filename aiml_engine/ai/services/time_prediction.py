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
    Generate a project-specific schedule warning.

    The warning is based on:
    - predicted delay
    - planned project duration
    - ML prediction
    - historical prediction
    - historical evidence
    - similarity
    - uncertainty
    - confidence

    Runtime is READ-ONLY.
    """

    historical_count = len(historical_delays)

    # =====================================================
    # 1. NO HISTORICAL EVIDENCE
    # =====================================================

    if historical_prediction is None:

        if final_prediction > 1095:

            return (
                "The project is facing a prolonged schedule risk, "
                "and sufficiently comparable completed projects are "
                "not available to strongly validate the expected "
                "delay. The current timeline should therefore be "
                "treated with caution."
            )

        elif final_prediction > 730:

            return (
                "The projected schedule indicates significant delay "
                "pressure, but comparable completed-project evidence "
                "is limited. The expected completion timeline should "
                "be monitored closely."
            )

        elif final_prediction > 365:

            return (
                "The current assessment indicates meaningful schedule "
                "slippage, while limited historical evidence reduces "
                "confidence in the exact delay estimate."
            )

        elif final_prediction > 90:

            return (
                "Some schedule slippage is anticipated, although the "
                "available historical evidence is insufficient to "
                "strongly validate the projected delay."
            )

        else:

            return (
                "The current model does not indicate a major schedule "
                "problem, but the limited historical evidence means "
                "the assessment should be monitored as the project "
                "progresses."
            )

    # =====================================================
    # 2. ML VS HISTORICAL COMPARISON
    # =====================================================

    disagreement = abs(
        ml_prediction - historical_prediction
    )

    # =====================================================
    # 3. HISTORICAL EVIDENCE QUALITY
    # =====================================================

    if historical_count == 1:

        evidence_message = (
            "The schedule assessment is supported by only one "
            "completed comparable project, which limits historical "
            "validation."
        )

    elif historical_count == 2:

        evidence_message = (
            "Only two completed comparable projects were available "
            "for schedule comparison, so the historical signal "
            "should be interpreted cautiously."
        )

    elif historical_count < 5:

        evidence_message = (
            f"{historical_count} completed comparable projects "
            "provide some historical support, although the evidence "
            "base remains relatively small."
        )

    elif average_similarity >= 0.85:

        evidence_message = (
            "The schedule outlook is strongly supported by highly "
            "similar completed projects."
        )

    elif average_similarity >= 0.80:

        evidence_message = (
            "The available historical projects provide strong "
            "comparative support for the current schedule outlook."
        )

    elif average_similarity >= 0.70:

        evidence_message = (
            "The historical comparison provides a reasonable basis "
            "for assessing the expected schedule behaviour."
        )

    elif average_similarity >= 0.65:

        evidence_message = (
            "Historical projects provide moderate support, although "
            "their similarity to the current project is not especially strong."
        )

    else:

        evidence_message = (
            "The available historical comparisons have relatively "
            "weak similarity to the current project."
        )

    # =====================================================
    # 4. DELAY SEVERITY
    # =====================================================

    if final_prediction > 1825:

        delay_message = (
            "The projected schedule indicates an exceptionally "
            "prolonged delay that could substantially affect project "
            "completion and downstream activities."
        )

    elif final_prediction > 1095:

        delay_message = (
            "The projected completion outlook indicates a prolonged "
            "delay, suggesting that substantial schedule recovery "
            "measures may be required."
        )

    elif final_prediction > 730:

        delay_message = (
            "The project is showing significant schedule pressure, "
            "with the expected delay likely to materially affect "
            "the original completion plan."
        )

    elif final_prediction > 365:

        delay_message = (
            "The current schedule outlook indicates a meaningful "
            "extension beyond the planned completion timeline."
        )

    elif final_prediction > 180:

        delay_message = (
            "The project is showing noticeable schedule slippage "
            "that could affect planned completion if the trend "
            "continues."
        )

    elif final_prediction > 90:

        delay_message = (
            "The assessment indicates moderate schedule slippage "
            "that should be monitored through upcoming milestones."
        )

    elif final_prediction > 30:

        delay_message = (
            "A limited schedule extension is currently anticipated, "
            "with continued milestone monitoring recommended."
        )

    elif final_prediction > 0:

        delay_message = (
            "The project shows some expected schedule slippage, "
            "but the projected extension remains comparatively limited."
        )

    else:

        delay_message = (
            "The current assessment does not indicate a significant "
            "delay against the planned schedule."
        )

    # =====================================================
    # 5. ML VS HISTORICAL RELATIONSHIP
    # =====================================================

    if disagreement > 1460:

        agreement_message = (
            "The ML estimate and historical project behaviour differ "
            "by more than four years, creating substantial uncertainty "
            "around the expected completion timeline."
        )

    elif disagreement > 1095:

        agreement_message = (
            "The model and historical evidence indicate a very large "
            "difference in expected delay, making the final schedule "
            "outlook particularly uncertain."
        )

    elif disagreement > 730:

        agreement_message = (
            "The ML estimate and historical project behaviour differ "
            "substantially, increasing uncertainty around the expected "
            "completion timeline."
        )

    elif disagreement > 365:

        agreement_message = (
            "The model and historical evidence indicate noticeably "
            "different schedule outcomes, so the predicted timeline "
            "should be reviewed carefully."
        )

    elif disagreement > 180:

        agreement_message = (
            "The model and historical evidence are broadly aligned, "
            "although their estimated delay outcomes show meaningful "
            "variation."
        )

    elif disagreement > 90:

        agreement_message = (
            "The two prediction approaches show some difference, "
            "but both provide a broadly comparable schedule outlook."
        )

    elif disagreement > 30:

        agreement_message = (
            "The ML estimate follows the historical signal reasonably "
            "closely, with only a modest difference between the two."
        )

    else:

        agreement_message = (
            "The ML estimate closely follows the behaviour observed "
            "in comparable completed projects."
        )

    # =====================================================
    # 6. SCHEDULE UNCERTAINTY
    # =====================================================

    range_width = max(
        0,
        float(range_max) - float(range_min)
    )

    if planned_duration_days > 0:

        uncertainty_ratio = (
            range_width / planned_duration_days
        )

    else:

        uncertainty_ratio = 0

    if uncertainty_ratio > 4:

        uncertainty_message = (
            "The possible delay outcomes span an extremely broad "
            "range compared with the original project duration, "
            "indicating very high schedule uncertainty."
        )

    elif uncertainty_ratio > 3:

        uncertainty_message = (
            "The possible delay outcomes span a very broad range "
            "relative to the original project duration."
        )

    elif uncertainty_ratio > 2:

        uncertainty_message = (
            "The expected delay range remains wide compared with "
            "the project's planned execution period."
        )

    elif uncertainty_ratio > 1:

        uncertainty_message = (
            "There is noticeable variation in the possible schedule "
            "outcome, making milestone-level monitoring important."
        )

    elif uncertainty_ratio > 0.5:

        uncertainty_message = (
            "The projected schedule has a moderate uncertainty "
            "range relative to the original execution period."
        )

    else:

        uncertainty_message = (
            "The estimated schedule range remains relatively "
            "contained compared with the original project duration."
        )

    # =====================================================
    # 7. CONFIDENCE
    # =====================================================

    if confidence == "LOW":

        confidence_message = (
            "Overall confidence in the exact delay estimate is low."
        )

    elif confidence == "MEDIUM":

        confidence_message = (
            "The prediction has moderate confidence and should be "
            "validated against actual milestone performance."
        )

    else:

        confidence_message = (
            "The available model and historical evidence provide "
            "relatively strong support for the schedule estimate."
        )

    # =====================================================
    # 8. PLANNED-DURATION RELATIONSHIP
    # =====================================================

    if planned_duration_days > 0:

        delay_ratio = (
            final_prediction / planned_duration_days
        )

    else:

        delay_ratio = 0

    if delay_ratio > 2:

        duration_message = (
            "The projected delay is more than twice the originally "
            "planned execution duration, indicating substantial "
            "schedule exposure."
        )

    elif delay_ratio > 1:

        duration_message = (
            "The projected delay is comparable to or greater than "
            "the original planned execution duration."
        )

    elif delay_ratio > 0.50:

        duration_message = (
            "The expected delay represents a substantial portion "
            "of the project's original planned duration."
        )

    elif delay_ratio > 0.25:

        duration_message = (
            "The anticipated delay represents a noticeable portion "
            "of the original project schedule."
        )

    else:

        duration_message = (
            "The projected delay remains relatively small compared "
            "with the original planned execution period."
        )

    # =====================================================
    # 9. BUILD FINAL WARNING
    # =====================================================

    messages = [
        delay_message,
        evidence_message,
        agreement_message,
        uncertainty_message,
        duration_message,
        confidence_message,
    ]

    # Remove exact duplicates.
    messages = list(dict.fromkeys(messages))

    return " ".join(messages)


# =========================================================
# MAIN TIME PREDICTION
# =========================================================

def predict_time(project, completed_similar_projects=None):
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
        completed_similar_projects
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

        ml_weight = 1.0 - historical_weight

        final_prediction = (
            ml_prediction * ml_weight
            + historical_prediction * historical_weight
        )

    else:

        final_prediction = ml_prediction

    final_prediction = max(
        0,
        final_prediction
    )

    # =====================================================
    # PREDICTION RANGE
    # =====================================================

    if len(historical_delays) >= 5:

        historical_q25 = _percentile(
            historical_delays,
            25
        )

        historical_q75 = _percentile(
            historical_delays,
            75
        )

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

        uncertainty = MODEL_ERROR_P90

        range_min = max(
            0,
            final_prediction - uncertainty
        )

        range_max = (
            final_prediction + uncertainty
        )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = _calculate_confidence(
        historical_count=len(historical_delays),
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


# =========================================================
# HISTORICAL DELAY ESTIMATION
# =========================================================

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

    # IMPORTANT:
    # This query is READ ONLY.
    from aiml_engine.ai.database import get_connection

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                project_ids
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