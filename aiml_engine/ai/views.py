from rest_framework.decorators import (
    api_view,
    authentication_classes,
)
from rest_framework.response import Response
from rest_framework import status

from .authentication import AIMLAPIKeyAuthentication

from aiml_engine.ai.services.prediction_service import (
    predict_project,
    _build_project_text,
    retrieve_similar_projects,
    retrieve_completed_similar_projects,
)

from aiml_engine.ai.services.cost_prediction import predict_cost
from aiml_engine.ai.services.time_prediction import predict_time
from aiml_engine.ai.services.risk_engine import calculate_risk

from aiml_engine.ai.embedding_service import generate_embedding
from aiml_engine.ai.services.project_reader import get_project_by_id


# =========================================================
# HEALTH CHECK
# =========================================================

@api_view(["GET"])
def health_check(request):
    return Response({
        "status": "ok",
        "service": "Django AI/ML Engine"
    })


# =========================================================
# EXISTING API
# =========================================================
# IMPORTANT:
# This API is intentionally kept unchanged.
# Existing URL, input and output remain the same.
# =========================================================

@api_view(["GET"])
@authentication_classes([AIMLAPIKeyAuthentication])
def predict_project_api(request):

    project_id = request.query_params.get("project_id")

    if project_id is None:
        return Response(
            {
                "error": "project_id is required."
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        project_id = int(project_id)
    except (TypeError, ValueError):
        return Response(
            {
                "error": "project_id must be a valid integer."
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        project = get_project_by_id(project_id)

    except Exception as e:
        import traceback

        print("========== PROJECT READ ERROR ==========")
        print(f"Project ID: {project_id}")
        print(f"Error: {e}")

        traceback.print_exc()

        print("========================================")

        return Response(
            {
                "error": "Failed to read project data.",
                "details": str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    if project is None:
        return Response(
            {
                "error": "Project not found.",
                "project_id": project_id
            },
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        result = predict_project(project)

        return Response(
            result,
            status=status.HTTP_200_OK
        )

    except Exception:
        import traceback

        traceback.print_exc()

        return Response(
            {
                "error": "Prediction failed."
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# =========================================================
# HELPER FUNCTIONS FOR NEW API RESPONSE
# =========================================================

def _safe_float(value):
    """
    Safely convert a value to float.
    """
    try:
        if value is None:
            return None

        return float(value)

    except (TypeError, ValueError):
        return value


def _clean_project_content(content):
    """
    Convert stored project text into a cleaner API response.

    This function only formats response data.
    It does NOT change prediction logic or database data.
    """

    if not content:
        return {}

    project_data = {}

    lines = str(content).splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if not key:
            continue

        project_data[key] = value

    return project_data


def _clean_similar_project(item):
    """
    Format a similar project for frontend/Postman.

    Raw embedding is never returned.
    """

    content = item.get("content")

    parsed_content = _clean_project_content(content)

    return {
        "project_id": item.get("project_id"),
        "similarity": _safe_float(
            item.get("similarity")
        ),
        "project_details": parsed_content,
    }


def _clean_similar_projects(projects):
    """
    Format list of similar projects.
    """

    if not projects:
        return []

    return [
        _clean_similar_project(item)
        for item in projects
    ]


# =========================================================
# NEW API
# =========================================================
# GET API FOR NEW PROJECT
#
# Existing API is NOT changed.
# Prediction logic is NOT changed.
# =========================================================

@api_view(["GET"])
@authentication_classes([AIMLAPIKeyAuthentication])
def predict_new_project_api(request):

    # -----------------------------------------------------
    # READ PROJECT DATA
    # -----------------------------------------------------

    project = {
        "project_id": request.query_params.get("project_id"),

        "project_name": request.query_params.get(
            "project_name"
        ),

        "agency": request.query_params.get(
            "agency"
        ),

        "ministry": request.query_params.get(
            "ministry"
        ),

        "sector": request.query_params.get(
            "sector"
        ),

        "state": request.query_params.get(
            "state"
        ),

        "progress_status": request.query_params.get(
            "progress_status"
        ),

        "physical_progress": request.query_params.get(
            "physical_progress"
        ),

        "original_cost": request.query_params.get(
            "original_cost"
        ),

        "revised_cost": request.query_params.get(
            "revised_cost"
        ),

        "start_date": request.query_params.get(
            "start_date"
        ),

        "original_completion_date": request.query_params.get(
            "original_completion_date"
        ),

        "revised_completion_date": request.query_params.get(
            "revised_completion_date"
        ),
    }

    # -----------------------------------------------------
    # VALIDATE REQUIRED FIELDS
    # -----------------------------------------------------

    required_fields = [
        "project_name",
        "agency",
        "ministry",
        "sector",
        "state",
        "start_date",
        "original_completion_date",
        "original_cost",
        "physical_progress",
    ]

    missing_fields = [
        field
        for field in required_fields
        if project.get(field) in (None, "")
    ]

    if missing_fields:
        return Response(
            {
                "error": "Required project fields are missing.",
                "missing_fields": missing_fields
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # -----------------------------------------------------
    # CONVERT NUMERIC VALUES
    # -----------------------------------------------------

    try:

        project["physical_progress"] = float(
            project["physical_progress"]
        )

        project["original_cost"] = float(
            project["original_cost"]
        )

        if project["revised_cost"] not in (None, ""):

            project["revised_cost"] = float(
                project["revised_cost"]
            )

    except (TypeError, ValueError):

        return Response(
            {
                "error": (
                    "physical_progress, original_cost and "
                    "revised_cost must be valid numbers."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # -----------------------------------------------------
    # CONVERT PROJECT ID
    # -----------------------------------------------------

    if project["project_id"] in (None, ""):

        project["project_id"] = None

    else:

        try:

            project["project_id"] = int(
                project["project_id"]
            )

        except (TypeError, ValueError):

            return Response(
                {
                    "error": "project_id must be a valid integer."
                },
                status=status.HTTP_400_BAD_REQUEST
            )

    # =====================================================
    # AI / ML PIPELINE
    # =====================================================

    try:

        # -------------------------------------------------
        # 1. BUILD PROJECT TEXT
        # -------------------------------------------------

        project_text = _build_project_text(
            project
        )

        # -------------------------------------------------
        # 2. GENERATE EMBEDDING ONCE
        # -------------------------------------------------

        embedding = generate_embedding(
            project_text
        )

        # -------------------------------------------------
        # 3. GENERAL SIMILAR PROJECTS
        # -------------------------------------------------

        similar_projects = retrieve_similar_projects(
            project,
            embedding,
            limit=10,
        )

        # -------------------------------------------------
        # 4. COMPLETED SIMILAR PROJECTS
        # -------------------------------------------------

        completed_similar_projects = (
            retrieve_completed_similar_projects(
                project,
                embedding,
                limit=50,
            )
        )

        # -------------------------------------------------
        # 5. COST PREDICTION
        # -------------------------------------------------

        cost_result = predict_cost(
            project,
            similar_projects=completed_similar_projects,
        )

        # -------------------------------------------------
        # 6. TIME PREDICTION
        # -------------------------------------------------

        predicted_delay = predict_time(
            project,
            completed_similar_projects=(
                completed_similar_projects
            ),
        )

        # -------------------------------------------------
        # 7. RISK ASSESSMENT
        # -------------------------------------------------

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
                    "LOW"
                )
            ),

            time_prediction_confidence=(
                predicted_delay.get(
                    "confidence",
                    "LOW"
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

        # =================================================
        # CLEAN SIMILAR PROJECTS
        # =================================================

        clean_similar_projects = (
            _clean_similar_projects(
                similar_projects
            )
        )

        clean_completed_similar_projects = (
            _clean_similar_projects(
                completed_similar_projects
            )
        )

        # =================================================
        # FINAL PROFESSIONAL RESPONSE
        # =================================================

        response_data = {

            # -------------------------------------------------
            # PROJECT
            # -------------------------------------------------

            "project": {

                "project_id": project.get(
                    "project_id"
                ),

                "project_name": project.get(
                    "project_name"
                ),

                "agency": project.get(
                    "agency"
                ),

                "ministry": project.get(
                    "ministry"
                ),

                "sector": project.get(
                    "sector"
                ),

                "state": project.get(
                    "state"
                ),

                "progress_status": project.get(
                    "progress_status"
                ),

                "physical_progress": project.get(
                    "physical_progress"
                ),

                "original_cost": project.get(
                    "original_cost"
                ),

                "revised_cost": project.get(
                    "revised_cost"
                ),

                "start_date": project.get(
                    "start_date"
                ),

                "original_completion_date": project.get(
                    "original_completion_date"
                ),

                "revised_completion_date": project.get(
                    "revised_completion_date"
                ),
            },

            # -------------------------------------------------
            # PREDICTIONS
            # -------------------------------------------------

            "predictions": {

                # =============================================
                # COST
                # =============================================

                "cost": {

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

                # =============================================
                # TIME
                # =============================================

                "time": {

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

                # =============================================
                # RISK
                # =============================================

                "risk": {

                    "risk_level": (
                        risk_result.get(
                            "risk_level"
                        )
                    ),

                    "risk_score": (
                        risk_result.get(
                            "risk_score"
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

                    "detected_issues": (
                        risk_result.get(
                            "detected_issues",
                            []
                        )
                    ),
                },
            },

            # -------------------------------------------------
            # SIMILAR PROJECT ANALYSIS
            # -------------------------------------------------

            "similar_project_analysis": {

                "total_similar_projects": len(
                    clean_similar_projects
                ),

                "total_completed_similar_projects": len(
                    clean_completed_similar_projects
                ),

                "similar_projects": (
                    clean_similar_projects
                ),

                "completed_similar_projects": (
                    clean_completed_similar_projects
                ),
            },
        }

        return Response(
            response_data,
            status=status.HTTP_200_OK
        )

    # =====================================================
    # VALIDATION ERROR
    # =====================================================

    except ValueError as e:

        return Response(
            {
                "error": str(e)
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # =====================================================
    # INTERNAL ERROR
    # =====================================================

    except Exception:

        import traceback

        print(
            "========== NEW PROJECT PREDICTION ERROR =========="
        )

        traceback.print_exc()

        print(
            "==================================================="
        )

        return Response(
            {
                "error": "Prediction failed."
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )