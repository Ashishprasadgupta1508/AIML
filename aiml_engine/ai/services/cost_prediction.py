from statistics import median
from aiml_engine.ai.database import get_connection


MIN_SIMILARITY = 0.60
SEARCH_LIMIT = 50

# Prototype calibration from 25 completed projects.
# These values are not true out-of-sample confidence intervals.
HISTORICAL_P25 = -37.90
HISTORICAL_P75 = 3.68
HISTORICAL_P90 = 26.58
HISTORICAL_P95 = 59.55


def _no_history_result(original_cost):
    return {
        "predicted_final_cost": round(original_cost, 2),
        "predicted_cost_overrun_percent": 0.0,
        "expected_cost_range": {
            "min_cost": round(
                original_cost * (1 + HISTORICAL_P25 / 100),
                2
            ),
            "max_cost": round(
                original_cost * (1 + HISTORICAL_P90 / 100),
                2
            ),
        },
        "confidence": "LOW",
        "historical_projects_used": 0,
        "historical_projects_found": 0,
        "average_similarity": 0.0,
        "historical_spread_percent": None,
        "warning": "No suitable completed historical projects were found.",
        "historical_projects": [],
    }


def predict_cost(project, similar_projects=None, limit=SEARCH_LIMIT):

    original_cost = project.get("original_cost")

    if original_cost is None or float(original_cost) <= 0:
        return {
            "predicted_final_cost": None,
            "predicted_cost_overrun_percent": None,
            "expected_cost_range": None,
            "confidence": "LOW",
            "historical_projects_used": 0,
            "historical_projects_found": 0,
            "average_similarity": 0.0,
            "historical_spread_percent": None,
            "warning": "Original project cost is missing or invalid.",
            "historical_projects": [],
        }

    original_cost = float(original_cost)

    if similar_projects is None:
        similar_projects = []

    similar_projects = [
        item
        for item in similar_projects
        if float(item.get("similarity", 0)) >= MIN_SIMILARITY
    ]

    if not similar_projects:
        return _no_history_result(original_cost)

    project_ids = [
        item["project_id"]
        for item in similar_projects
    ]

    similarity_map = {
        item["project_id"]: float(item.get("similarity", 0))
        for item in similar_projects
    }

    placeholders = ",".join(["%s"] * len(project_ids))

    # READ ONLY query.
    query = f"""
        SELECT
            project_id,
            original_cost,
            cumulative_expenditure,
            actual_completion_date
        FROM project
        WHERE project_id IN ({placeholders})
          AND actual_completion_date IS NOT NULL
          AND original_cost IS NOT NULL
          AND original_cost > 0
          AND cumulative_expenditure IS NOT NULL
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, project_ids)
            rows = cur.fetchall()

    historical = []

    for (
        project_id,
        historical_original_cost,
        expenditure,
        actual_completion_date
    ) in rows:

        historical_original_cost = float(historical_original_cost)
        expenditure = float(expenditure)

        overrun = (
            (expenditure - historical_original_cost)
            / historical_original_cost
        ) * 100

        historical.append({
            "project_id": project_id,
            "similarity": similarity_map.get(project_id, 0.0),
            "cost_overrun_percent": overrun,
        })

    if not historical:
        result = _no_history_result(original_cost)

        result["warning"] = (
            "Similar projects were found, but no completed "
            "historical projects with actual cost outcomes "
            "were available."
        )

        return result

    # ---------------------------------------------------------
    # Remove extreme historical outliers using MAD.
    # ---------------------------------------------------------

    values = [
        item["cost_overrun_percent"]
        for item in historical
    ]

    median_value = median(values)

    deviations = [
        abs(value - median_value)
        for value in values
    ]

    mad = median(deviations)

    if mad == 0:

        filtered = historical

    else:

        filtered = []

        for item in historical:

            modified_z = (
                0.6745
                * (
                    item["cost_overrun_percent"]
                    - median_value
                )
                / mad
            )

            if abs(modified_z) <= 3.5:
                filtered.append(item)

    # Don't throw away too much historical evidence.
    if len(filtered) < 3:
        filtered = historical

    # ---------------------------------------------------------
    # Similarity-weighted historical estimate.
    # Higher similarity gets more influence.
    # ---------------------------------------------------------

    total_weight = 0.0
    weighted_sum = 0.0

    for item in filtered:

        similarity = item["similarity"]

        weight = similarity ** 4

        weighted_sum += (
            item["cost_overrun_percent"] * weight
        )

        total_weight += weight

    if total_weight == 0:

        predicted_overrun = median(
            item["cost_overrun_percent"]
            for item in filtered
        )

    else:

        predicted_overrun = (
            weighted_sum / total_weight
        )

    # ---------------------------------------------------------
    # Historical statistics.
    # ---------------------------------------------------------

    filtered_values = [
        item["cost_overrun_percent"]
        for item in filtered
    ]

    average_similarity = (
        sum(item["similarity"] for item in filtered)
        / len(filtered)
    )

    spread = (
        max(filtered_values)
        - min(filtered_values)
        if len(filtered_values) > 1
        else 0.0
    )

    # ---------------------------------------------------------
    # Final cost prediction.
    # ---------------------------------------------------------

    predicted_final_cost = (
        original_cost
        * (1 + predicted_overrun / 100)
    )

    # ---------------------------------------------------------
    # Expected cost range.
    #
    # Prototype range based on historical percentile
    # calibration.
    # ---------------------------------------------------------

    range_min_overrun = min(
        HISTORICAL_P25,
        predicted_overrun
    )

    range_max_overrun = max(
        HISTORICAL_P90,
        predicted_overrun
    )

    range_min_cost = max(
        0.0,
        original_cost
        * (1 + range_min_overrun / 100)
    )

    range_max_cost = max(
        predicted_final_cost,
        original_cost
        * (1 + range_max_overrun / 100)
    )

    # ---------------------------------------------------------
    # Confidence.
    # ---------------------------------------------------------

    if (
        len(filtered) >= 5
        and average_similarity >= 0.80
        and spread <= 50
    ):
        confidence = "HIGH"

    elif (
        len(filtered) >= 3
        and average_similarity >= 0.75
        and spread <= 100
    ):
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    # ---------------------------------------------------------
    # Warning.
    # ---------------------------------------------------------

    warning = None

    if spread > 100:

        warning = (
            "Historical completed projects show very high "
            "cost variability."
        )

    elif spread > 50:

        warning = (
            "Historical completed projects show moderate "
            "cost variability."
        )

    return {
        "predicted_final_cost": round(
            predicted_final_cost,
            2
        ),

        "predicted_cost_overrun_percent": round(
            predicted_overrun,
            2
        ),

        "expected_cost_range": {
            "min_cost": round(
                range_min_cost,
                2
            ),
            "max_cost": round(
                range_max_cost,
                2
            ),
        },

        "confidence": confidence,

        "historical_projects_used": len(filtered),

        "historical_projects_found": len(historical),

        "average_similarity": round(
            average_similarity,
            4
        ),

        "historical_spread_percent": round(
            spread,
            2
        ),

        "warning": warning,

        "historical_projects": filtered,
    }
