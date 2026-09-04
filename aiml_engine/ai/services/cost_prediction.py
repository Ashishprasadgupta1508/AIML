from statistics import median

from aiml_engine.ai.database import get_connection


MIN_SIMILARITY = 0.60
SEARCH_LIMIT = 50

# Prototype calibration from completed projects.
HISTORICAL_P25 = -37.90
HISTORICAL_P75 = 3.68
HISTORICAL_P90 = 26.58
HISTORICAL_P95 = 59.55


def _build_cost_warning(
    original_cost,
    predicted_final_cost,
    predicted_overrun,
    spread,
    average_similarity,
    historical_count,
    confidence,
):
    """
    Generate a concise, deterministic and data-driven cost warning.

    Warning generation uses:
    - predicted budget movement
    - historical variability
    - number of completed comparable projects
    - average similarity
    - confidence

    Same project + same data produces the same warning.
    No randomness is introduced.

    Runtime is READ-ONLY.
    """

    if historical_count == 0:
        return (
            "No sufficiently comparable completed project provides a "
            "reliable cost outcome for validation, so the estimate "
            "should be treated with high uncertainty."
        )

    predicted_overrun = float(predicted_overrun)
    spread = float(spread)
    average_similarity = float(average_similarity)
    historical_count = int(historical_count)

    # ---------------------------------------------------------
    # 1. Budget position
    # ---------------------------------------------------------

    if predicted_overrun >= 50:
        budget_message = (
            f"The projected final cost is approximately "
            f"{predicted_overrun:.1f}% above the original budget, "
            "indicating substantial financial exposure."
        )

    elif predicted_overrun >= 25:
        budget_message = (
            f"The projected final cost is estimated to be about "
            f"{predicted_overrun:.1f}% above the original budget, "
            "indicating a significant potential cost increase."
        )

    elif predicted_overrun >= 10:
        budget_message = (
            f"The projected final cost is around "
            f"{predicted_overrun:.1f}% above the original budget, "
            "indicating a noticeable cost pressure."
        )

    elif predicted_overrun >= 3:
        budget_message = (
            f"A cost increase of approximately "
            f"{predicted_overrun:.1f}% is projected against the "
            "original budget."
        )

    elif predicted_overrun <= -25:
        budget_message = (
            f"The projected final cost is approximately "
            f"{abs(predicted_overrun):.1f}% below the original budget, "
            "indicating a potentially substantial saving."
        )

    elif predicted_overrun <= -10:
        budget_message = (
            f"The projected final cost indicates a potential saving "
            f"of around {abs(predicted_overrun):.1f}% against the "
            "original budget."
        )

    elif predicted_overrun <= -3:
        budget_message = (
            f"The projected final cost is approximately "
            f"{abs(predicted_overrun):.1f}% below the original budget."
        )

    else:
        budget_message = (
            "The projected final cost remains close to the "
            "original project budget."
        )

    # ---------------------------------------------------------
    # 2. Historical evidence
    # ---------------------------------------------------------

    if historical_count == 1:
        evidence_message = (
            "The estimate is based on only one completed comparable "
            "project, so historical validation is very limited."
        )

    elif historical_count == 2:
        evidence_message = (
            "Only two completed comparable projects are available, "
            "so the historical cost signal should be interpreted cautiously."
        )

    elif historical_count < 5:
        evidence_message = (
            f"The estimate is supported by {historical_count} completed "
            "comparables, giving only a limited historical evidence base."
        )

    elif average_similarity >= 0.85:
        evidence_message = (
            f"The estimate is supported by {historical_count} highly "
            "similar completed projects, giving the historical comparison "
            "strong relevance."
        )

    elif average_similarity >= 0.80:
        evidence_message = (
            f"{historical_count} closely matched completed projects "
            "provide strong comparative support for the cost outlook."
        )

    elif average_similarity >= 0.70:
        evidence_message = (
            f"{historical_count} completed comparable projects provide "
            "a reasonable historical basis for the cost assessment."
        )

    elif average_similarity >= 0.65:
        evidence_message = (
            f"{historical_count} completed projects provide moderate "
            "historical support, although project similarity is limited."
        )

    else:
        evidence_message = (
            f"The {historical_count} historical comparisons have relatively "
            "weak similarity to the current project, reducing validation strength."
        )

    # ---------------------------------------------------------
    # 3. Historical variability
    # ---------------------------------------------------------

    if spread >= 150:
        variability_message = (
            f"Historical cost outcomes vary by approximately "
            f"{spread:.1f} percentage points, showing very high cost "
            "uncertainty."
        )

    elif spread >= 100:
        variability_message = (
            f"Historical cost outcomes span approximately "
            f"{spread:.1f} percentage points, making the final-cost "
            "estimate less precise."
        )

    elif spread >= 50:
        variability_message = (
            f"Comparable projects show considerable cost variation "
            f"of about {spread:.1f} percentage points."
        )

    elif spread >= 25:
        variability_message = (
            f"Historical cost outcomes vary by approximately "
            f"{spread:.1f} percentage points, introducing some uncertainty."
        )

    else:
        variability_message = (
            "Comparable completed projects show relatively consistent "
            "cost outcomes."
        )

    # ---------------------------------------------------------
    # 4. Confidence
    # ---------------------------------------------------------

    if confidence == "LOW":
        confidence_message = (
            "The available historical evidence provides limited confidence "
            "in the exact final-cost estimate."
        )

    elif confidence == "MEDIUM":
        confidence_message = (
            "The estimate has moderate confidence and should be checked "
            "against current expenditure."
        )

    else:
        confidence_message = (
            "The estimate has comparatively strong historical support."
        )

    # ---------------------------------------------------------
    # 5. Select the strongest project-specific signals
    # ---------------------------------------------------------

    # Major cost increase:
    if predicted_overrun >= 25:

        primary = budget_message

        if spread >= 100:
            secondary = variability_message
        elif confidence == "LOW":
            secondary = confidence_message
        elif average_similarity < 0.70:
            secondary = evidence_message
        else:
            secondary = evidence_message

    # Major projected saving:
    elif predicted_overrun <= -25:

        primary = budget_message

        if spread >= 100:
            secondary = variability_message
        elif confidence == "LOW":
            secondary = confidence_message
        elif average_similarity < 0.70:
            secondary = evidence_message
        else:
            secondary = evidence_message

    # Moderate cost increase:
    elif predicted_overrun >= 10:

        primary = budget_message

        if spread >= 100:
            secondary = variability_message
        elif confidence == "LOW":
            secondary = confidence_message
        elif average_similarity < 0.70:
            secondary = evidence_message
        else:
            secondary = evidence_message

    # Moderate saving:
    elif predicted_overrun <= -10:

        primary = budget_message

        if spread >= 100:
            secondary = variability_message
        elif confidence == "LOW":
            secondary = confidence_message
        elif average_similarity < 0.70:
            secondary = evidence_message
        else:
            secondary = evidence_message

    # Smaller budget movement:
    elif abs(predicted_overrun) >= 3:

        primary = budget_message

        if spread >= 150:
            secondary = variability_message
        elif confidence == "LOW":
            secondary = confidence_message
        elif average_similarity < 0.65:
            secondary = evidence_message
        elif spread >= 50:
            secondary = variability_message
        else:
            secondary = evidence_message

    # Near-budget prediction:
    else:

        if spread >= 150:
            primary = budget_message
            secondary = (
                f"Historical outcomes vary by about {spread:.1f} "
                "percentage points, so the apparently stable budget "
                "position should be interpreted cautiously."
            )

        elif spread >= 100:
            primary = budget_message
            secondary = variability_message

        elif confidence == "LOW":
            primary = budget_message
            secondary = confidence_message

        elif average_similarity < 0.65:
            primary = budget_message
            secondary = evidence_message

        elif spread >= 50:
            primary = budget_message
            secondary = variability_message

        else:
            primary = budget_message
            secondary = evidence_message

    # ---------------------------------------------------------
    # 6. Avoid duplicate information
    # ---------------------------------------------------------

    if primary == secondary:
        return primary

    return f"{primary} {secondary}"


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

        "warning": (
            "No completed project with a reliable actual cost outcome "
            "was sufficiently comparable to this project, so the cost "
            "prediction should be treated as highly uncertain."
        ),

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

            "warning": (
                "A reliable cost warning cannot be generated because "
                "the original project cost is missing or invalid."
            ),

            "historical_projects": [],
        }

    original_cost = float(original_cost)

    if similar_projects is None:
        similar_projects = []

    similar_projects = [
        item
        for item in similar_projects
        if float(item.get("similarity", 0)) >= MIN_SIMILARITY
        and item.get("project_id") != project.get("project_id")
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
            "Similar projects were identified, but none had a usable "
            "completed cost outcome for comparison with this project."
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
    # Dynamic project-specific warning.
    # ---------------------------------------------------------

    warning = _build_cost_warning(
        original_cost=original_cost,
        predicted_final_cost=predicted_final_cost,
        predicted_overrun=predicted_overrun,
        spread=spread,
        average_similarity=average_similarity,
        historical_count=len(filtered),
        confidence=confidence,
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