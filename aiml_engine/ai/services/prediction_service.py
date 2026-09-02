import pandas as pd
from joblib import load
from pathlib import Path
from aiml_engine.ai.services.time_prediction import predict_time
from aiml_engine.ai.services.vector_search import (
    search_similar_projects,
    search_completed_similar_projects,
)
from aiml_engine.ai.services.cost_prediction import predict_cost
from aiml_engine.ai.services.risk_engine import calculate_risk


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved"

time_model = load(
    MODEL_DIR / "time_model.joblib"
)





def retrieve_similar_projects(project, limit=5):

    text = _build_project_text(project)

    return search_similar_projects(
        text,
        limit=limit,
        exclude_project_id=project.get("project_id")
    )


def retrieve_completed_similar_projects(
    project,
    limit=10
):
    """
    Retrieve only completed historical projects.

    These projects have actual completion and actual
    expenditure data available.
    """

    text = _build_project_text(project)

    return search_completed_similar_projects(
        text,
        limit=limit,
        exclude_project_id=project.get("project_id")
    )


def _clean_similar_project(item):
    """
    Convert raw vector-search content into a small
    API-safe object.
    """

    content = item.get("content", "")

    def extract_field(field_name):
        prefix = f"{field_name}:"

        for line in content.splitlines():
            line = line.strip()

            if line.startswith(prefix):
                return line[len(prefix):].strip()

        return None

    return {
        "project_id": item.get("project_id"),
        "similarity": round(
            float(item.get("similarity", 0)),
            4
        ),
        "project_name": extract_field(
            "Project Name"
        ),
        "sector": extract_field(
            "Sector"
        ),
        "state": extract_field(
            "State"
        ),
        "agency": extract_field(
            "Agency"
        ),
    }


def predict_project(project):

    # -----------------------------------------
    # 1. General similar projects
    # -----------------------------------------

    similar_projects = retrieve_similar_projects(
        project,
        limit=10
    )

    # -----------------------------------------
    # 2. Completed historical projects
    # -----------------------------------------

    completed_similar_projects = (
        retrieve_completed_similar_projects(
            project,
            limit=50
        )
    )

    # -----------------------------------------
    # 3. Cost prediction
    # -----------------------------------------

    cost_result = predict_cost(
        project,
        similar_projects=completed_similar_projects,
        
    )

    # -----------------------------------------
    # 4. Time prediction
    # -----------------------------------------

    predicted_delay = predict_time(
    project,
    completed_similar_projects=completed_similar_projects
)

    # -----------------------------------------
    # 5. Risk assessment
    # -----------------------------------------

    risk_result = calculate_risk(
        project,
        predicted_delay_days=predicted_delay["predicted_delay_days"],
        similar_projects=similar_projects,
        completed_similar_projects=completed_similar_projects,
        predicted_cost_overrun_percent=cost_result.get(
            "predicted_cost_overrun_percent"
        ),
        cost_prediction_confidence=cost_result.get(
            "confidence",
            "LOW"
        ),
        time_prediction_confidence=predicted_delay.get(
            "confidence",
            "LOW"
        ),
        cost_range=cost_result.get(
            "expected_cost_range"
        ),
        time_range=predicted_delay.get(
            "expected_delay_range"
        ),
)

    # -----------------------------------------
    # 6. Clean general similar projects
    # -----------------------------------------

    clean_similar_projects = [
        _clean_similar_project(item)
        for item in similar_projects
    ]

    # -----------------------------------------
    # 7. Clean completed historical projects
    # -----------------------------------------

    clean_completed_projects = [
        _clean_similar_project(item)
        for item in completed_similar_projects
    ]

    # -----------------------------------------
    # 8. Final response
    # -----------------------------------------

    return {
    "project_id": project.get("project_id"),

    "cost_prediction": {
        "predicted_final_cost": cost_result.get(
            "predicted_final_cost"
        ),
        "predicted_cost_overrun_percent": cost_result.get(
            "predicted_cost_overrun_percent"
        ),
        "expected_cost_range": cost_result.get(
            "expected_cost_range"
        ),
        "confidence": cost_result.get("confidence"),
        "historical_projects_used": cost_result.get(
            "historical_projects_used"
        ),
        "warning": cost_result.get("warning"),
    },

    "time_prediction": {
        "predicted_delay_days": predicted_delay.get(
            "predicted_delay_days"
        ),
        "expected_delay_range": predicted_delay.get(
            "expected_delay_range"
        ),
        "planned_duration_days": predicted_delay.get(
            "planned_duration_days"
        ),
        "confidence": predicted_delay.get("confidence"),
        "historical_projects_used": predicted_delay.get(
            "historical_projects_used"
        ),
        "average_similarity": predicted_delay.get(
            "average_similarity"
        ),
        "warning": predicted_delay.get("warning"),
    },

    "risk": {
        "risk_level": risk_result.get("risk_level"),
        "issue_id": risk_result.get("issue_id"),
        "reason": risk_result.get("reason"),
        "recommended_solution": risk_result.get(
            "recommended_solution"
        ),
        "risk_score": risk_result.get("risk_score"),
        "detected_issues": risk_result.get(
            "detected_issues",
            []
        ),
    }
}


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