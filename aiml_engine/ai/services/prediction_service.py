from aiml_engine.ai.embedding_service import generate_embedding

from aiml_engine.ai.services.vector_search import (
    search_similar_projects_by_embedding,
    search_completed_similar_projects_by_embedding,
)

from aiml_engine.ai.services.cost_prediction import predict_cost
from aiml_engine.ai.services.time_prediction import predict_time
from aiml_engine.ai.services.risk_engine import calculate_risk


# =========================================================
# Project Text Builder
# =========================================================

def _build_project_text(project):
    """
    Convert project information into a text representation
    used for semantic similarity search.

    READ ONLY.
    """

    fields = [
        ("Project Name", project.get("project_name")),
        ("Agency", project.get("agency")),
        ("Ministry", project.get("ministry")),
        ("Sector", project.get("sector")),
        ("State", project.get("state")),
        ("Progress Status", project.get("progress_status")),
        ("Physical Progress", project.get("physical_progress")),
        ("Original Cost", project.get("original_cost")),
        ("Revised Cost", project.get("revised_cost")),
        ("Start Date", project.get("start_date")),
        ("Original Completion Date", project.get("original_completion_date")),
        ("Revised Completion Date", project.get("revised_completion_date")),
    ]

    parts = []

    for label, value in fields:
        if value is None:
            continue

        parts.append(f"{label}: {value}")

    return "\n".join(parts)


# =========================================================
# Similar Project Retrieval
# =========================================================

def retrieve_similar_projects(
    project,
    embedding,
    limit=10,
):
    """
    Retrieve general similar projects.

    The current project is excluded from the search.

    READ ONLY.
    """

    project_id = project.get("project_id")

    return search_similar_projects_by_embedding(
        embedding,
        limit=limit,
        exclude_project_id=project_id,
    )


def retrieve_completed_similar_projects(
    project,
    embedding,
    limit=50,
):
    """
    Retrieve completed historical projects
    similar to the current project.

    The current project is excluded.

    READ ONLY.
    """

    project_id = project.get("project_id")

    return search_completed_similar_projects_by_embedding(
        embedding,
        limit=limit,
        exclude_project_id=project_id,
    )


# =========================================================
# COMPLETE AI / ML PIPELINE
# =========================================================

def predict_project(project):
    """
    Complete AI/ML prediction pipeline.

    Flow:

        Project
           ↓
        Build project text
           ↓
        Generate 384D embedding ONCE
           ↓
        ┌─────────────────────────────┐
        │                             │
        ↓                             ↓
    General Similar            Completed Similar
       Projects                  Projects
        │                             │
        └──────────────┬──────────────┘
                       ↓
                 Cost Prediction
                       ↓
                 Time Prediction
                       ↓
                  Risk Engine
                       ↓
                Final API Response

    Runtime is READ-ONLY with respect to Supabase.

    NOTE:
    Similar project data is used internally for AI prediction,
    but is intentionally NOT exposed in the final API response.
    """

    # -----------------------------------------------------
    # 1. Build project text
    # -----------------------------------------------------

    project_text = _build_project_text(project)

    # -----------------------------------------------------
    # 2. Generate embedding ONLY ONCE
    # -----------------------------------------------------

    embedding = generate_embedding(project_text)

    # -----------------------------------------------------
    # 3. Retrieve general similar projects
    # -----------------------------------------------------

    similar_projects = retrieve_similar_projects(
        project,
        embedding,
        limit=10,
    )

    # -----------------------------------------------------
    # 4. Retrieve completed historical projects
    # -----------------------------------------------------

    completed_similar_projects = (
        retrieve_completed_similar_projects(
            project,
            embedding,
            limit=50,
        )
    )

    # -----------------------------------------------------
    # 5. Cost prediction
    # -----------------------------------------------------

    cost_result = predict_cost(
        project,
        similar_projects=completed_similar_projects,
    )

    # -----------------------------------------------------
    # 6. Time prediction
    # -----------------------------------------------------

    predicted_delay = predict_time(
        project,
        completed_similar_projects=completed_similar_projects,
    )

    # -----------------------------------------------------
    # 7. Risk assessment
    # -----------------------------------------------------

    risk_result = calculate_risk(
        project,

        predicted_delay_days=(
            predicted_delay.get(
                "predicted_delay_days"
            )
        ),

        similar_projects=similar_projects,

        completed_similar_projects=(
            completed_similar_projects
        ),

        predicted_cost_overrun_percent=(
            cost_result.get(
                "predicted_cost_overrun_percent"
            )
        ),

        cost_prediction_confidence=(
            cost_result.get(
                "confidence",
                "LOW",
            )
        ),

        time_prediction_confidence=(
            predicted_delay.get(
                "confidence",
                "LOW",
            )
        ),

        cost_range=(
            cost_result.get(
                "expected_cost_range"
            )
        ),

        time_range=(
            predicted_delay.get(
                "expected_delay_range"
            )
        ),
    )

    # =====================================================
    # FINAL API RESPONSE
    # =====================================================
    #
    # Similar projects are intentionally NOT returned here.
    #
    # They remain available internally for:
    #   - cost prediction
    #   - time prediction
    #   - risk assessment
    #
    # =====================================================

    return {
        "project_id": project.get(
            "project_id"
        ),

        # =================================================
        # COST PREDICTION
        # =================================================

        "cost_prediction": {

            "predicted_final_cost": (
                cost_result.get(
                    "predicted_final_cost"
                )
            ),

            "predicted_cost_overrun_percent": (
                cost_result.get(
                    "predicted_cost_overrun_percent"
                )
            ),

            "expected_cost_range": (
                cost_result.get(
                    "expected_cost_range"
                )
            ),

            "confidence": (
                cost_result.get(
                    "confidence"
                )
            ),

            "historical_projects_used": (
                cost_result.get(
                    "historical_projects_used"
                )
            ),

            "historical_projects_found": (
                cost_result.get(
                    "historical_projects_found"
                )
            ),

            "average_similarity": (
                cost_result.get(
                    "average_similarity"
                )
            ),

            "historical_spread_percent": (
                cost_result.get(
                    "historical_spread_percent"
                )
            ),

            "warning": (
                cost_result.get(
                    "warning"
                )
            ),
        },

        # =================================================
        # TIME PREDICTION
        # =================================================

        "time_prediction": {

            "predicted_delay_days": (
                predicted_delay.get(
                    "predicted_delay_days"
                )
            ),

            "expected_delay_range": (
                predicted_delay.get(
                    "expected_delay_range"
                )
            ),

            "planned_duration_days": (
                predicted_delay.get(
                    "planned_duration_days"
                )
            ),

            "ml_predicted_delay_days": (
                predicted_delay.get(
                    "ml_predicted_delay_days"
                )
            ),

            "historical_predicted_delay_days": (
                predicted_delay.get(
                    "historical_predicted_delay_days"
                )
            ),

            "historical_projects_used": (
                predicted_delay.get(
                    "historical_projects_used"
                )
            ),

            "average_similarity": (
                predicted_delay.get(
                    "average_similarity"
                )
            ),

            "confidence": (
                predicted_delay.get(
                    "confidence"
                )
            ),

            "warning": (
                predicted_delay.get(
                    "warning"
                )
            ),
        },

        # =================================================
        # RISK ASSESSMENT
        # =================================================

        "risk": {

            "risk_level": (
                risk_result.get(
                    "risk_level"
                )
            ),

            "issue_id": (
                risk_result.get(
                    "issue_id"
                )
            ),

            "reason": (
                risk_result.get(
                    "reason"
                )
            ),

            "recommended_solution": (
                risk_result.get(
                    "recommended_solution"
                )
            ),

            "risk_score": (
                risk_result.get(
                    "risk_score"
                )
            ),

            "detected_issues": (
                risk_result.get(
                    "detected_issues",
                    [],
                )
            ),
        },
    }