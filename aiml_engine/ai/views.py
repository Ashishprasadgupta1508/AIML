from rest_framework.decorators import (
    api_view,
    authentication_classes,
)
from functools import lru_cache
from rest_framework.response import Response
from rest_framework import status

from .authentication import AIMLAPIKeyAuthentication
from aiml_engine.ai.services.project_recommendation import (
    generate_project_recommendations,
)
from aiml_engine.ai.services.early_warning import (
    generate_early_warning,
)
from aiml_engine.ai.services.benchmarking import (
    generate_benchmarking,
)

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
from aiml_engine.ai.services.project_assistant import (
    ask_project_assistant,
    AssistantConfigurationError,
    AssistantProviderError,
)


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

        # -----------------------------------------------------
        # COST ESCALATION DRIVER ANALYSIS
        # -----------------------------------------------------
        # Small backward-compatible extension.
        # Existing prediction fields are preserved.
        # -----------------------------------------------------
        cost_result = result.get("cost_prediction") if isinstance(result, dict) else None

        if isinstance(cost_result, dict) and "cost_escalation_analysis" not in cost_result:
            historical_projects = cost_result.get("historical_projects") or []

            historical_overruns = [
                item.get("cost_overrun_percent")
                for item in historical_projects
                if isinstance(item, dict)
                and item.get("cost_overrun_percent") is not None
            ]

            try:
                historical_average_overrun = (
                    sum(float(value) for value in historical_overruns)
                    / len(historical_overruns)
                    if historical_overruns
                    else None
                )
            except (TypeError, ValueError):
                historical_average_overrun = None

            cost_result = dict(cost_result)
            cost_result["cost_escalation_analysis"] = _build_cost_escalation_analysis(
                project=project,
                predicted_overrun=cost_result.get("predicted_cost_overrun_percent"),
                spread=cost_result.get("historical_spread_percent"),
                average_similarity=cost_result.get("average_similarity", 0.0),
                historical_average_overrun=historical_average_overrun,
                confidence=cost_result.get("confidence", "LOW"),
            )

            result = dict(result)
            result["cost_prediction"] = cost_result

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

def _build_cost_escalation_analysis(
    project,
    predicted_overrun,
    spread,
    average_similarity=0.0,
    historical_average_overrun=None,
    confidence="LOW",
):
    """
    Build a small, deterministic cost-escalation driver analysis.

    This helper is intentionally kept in views.py so the existing
    cost_prediction.py contract remains unchanged.
    """
    try:
        predicted_overrun = (
            None if predicted_overrun is None else float(predicted_overrun)
        )
    except (TypeError, ValueError):
        predicted_overrun = None

    try:
        spread = None if spread is None else float(spread)
    except (TypeError, ValueError):
        spread = None

    try:
        average_similarity = float(average_similarity or 0.0)
    except (TypeError, ValueError):
        average_similarity = 0.0

    try:
        historical_average_overrun = (
            None
            if historical_average_overrun is None
            else float(historical_average_overrun)
        )
    except (TypeError, ValueError):
        historical_average_overrun = None

    original_cost = project.get("original_cost")
    revised_cost = project.get("revised_cost")
    expenditure = project.get("cumulative_expenditure")

    try:
        original_cost = (
            None if original_cost in (None, "") else float(original_cost)
        )
    except (TypeError, ValueError):
        original_cost = None

    try:
        revised_cost = (
            None if revised_cost in (None, "") else float(revised_cost)
        )
    except (TypeError, ValueError):
        revised_cost = None

    try:
        expenditure = (
            None if expenditure in (None, "") else float(expenditure)
        )
    except (TypeError, ValueError):
        expenditure = None

    # Highest-priority driver: the model projects a material budget increase.
    if predicted_overrun is not None and predicted_overrun >= 25:
        return {
            "code": "HIGH_PREDICTED_COST_PRESSURE",
            "explanation": (
                f"The model projects approximately {predicted_overrun:.1f}% "
                "cost overrun against the original budget, indicating strong "
                "cost escalation pressure."
            ),
            "impact": "HIGH",
        }

    # Current expenditure already above the approved/original cost.
    if (
        original_cost is not None
        and original_cost > 0
        and expenditure is not None
        and expenditure > original_cost
    ):
        return {
            "code": "EXPENDITURE_ABOVE_ORIGINAL_COST",
            "explanation": (
                "Cumulative expenditure is already above the original project "
                "cost, indicating realized cost escalation."
            ),
            "impact": "HIGH",
        }

    # Revised cost is higher than the original approved cost.
    if (
        original_cost is not None
        and original_cost > 0
        and revised_cost is not None
        and revised_cost > original_cost
    ):
        revised_increase = ((revised_cost - original_cost) / original_cost) * 100
        return {
            "code": "REVISED_COST_PRESSURE",
            "explanation": (
                f"The revised project cost is approximately {revised_increase:.1f}% "
                "above the original cost, indicating an established budget "
                "revision."
            ),
            "impact": "HIGH" if revised_increase >= 25 else "MODERATE",
        }

    # Strong historical variability makes escalation risk less predictable.
    if spread is not None and spread >= 100:
        return {
            "code": "HIGH_HISTORICAL_COST_VARIABILITY",
            "explanation": (
                f"Comparable completed projects show approximately {spread:.1f} "
                "percentage points of cost-outcome variation, making future "
                "cost escalation highly uncertain."
            ),
            "impact": "HIGH",
        }

    # Historical comparables indicate a positive cost pressure.
    if (
        historical_average_overrun is not None
        and historical_average_overrun >= 10
    ):
        return {
            "code": "HISTORICAL_SECTOR_COST_PRESSURE",
            "explanation": (
                f"Comparable historical projects show an average cost overrun "
                f"of approximately {historical_average_overrun:.1f}%, indicating "
                "persistent cost pressure in the historical evidence."
            ),
            "impact": "MODERATE",
        }

    # Low confidence is itself an escalation-management signal.
    if confidence == "LOW":
        return {
            "code": "LOW_COST_PREDICTION_CONFIDENCE",
            "explanation": (
                "The cost prediction has low confidence, so the project should "
                "be monitored closely for emerging expenditure or escalation."
            ),
            "impact": "MODERATE",
        }

    return {
        "code": "NO_MAJOR_COST_DRIVER_DETECTED",
        "explanation": (
            "No major cost escalation driver was detected from the available "
            "prediction, expenditure, revised-cost, and historical signals."
        ),
        "impact": "LOW",
    }


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

        # -------------------------------------------------
        # COST ESCALATION DRIVER ANALYSIS
        # -------------------------------------------------
        if isinstance(cost_result, dict):
            historical_projects = cost_result.get("historical_projects") or []

            historical_overruns = [
                item.get("cost_overrun_percent")
                for item in historical_projects
                if isinstance(item, dict)
                and item.get("cost_overrun_percent") is not None
            ]

            try:
                historical_average_overrun = (
                    sum(float(value) for value in historical_overruns)
                    / len(historical_overruns)
                    if historical_overruns
                    else None
                )
            except (TypeError, ValueError):
                historical_average_overrun = None

            cost_result = dict(cost_result)
            cost_result["cost_escalation_analysis"] = _build_cost_escalation_analysis(
                project=project,
                predicted_overrun=cost_result.get("predicted_cost_overrun_percent"),
                spread=cost_result.get("historical_spread_percent"),
                average_similarity=cost_result.get("average_similarity", 0.0),
                historical_average_overrun=historical_average_overrun,
                confidence=cost_result.get("confidence", "LOW"),
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

                    "cost_escalation_analysis": (
                        cost_result.get(
                            "cost_escalation_analysis"
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
@api_view(["GET"])
@authentication_classes([AIMLAPIKeyAuthentication])
def early_warning_api(request):

    project_id = request.query_params.get(
        "project_id"
    )

    if project_id is None:

        return Response(
            {
                "error": "project_id is required."
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        project_id = int(
            project_id
        )

    except (TypeError, ValueError):

        return Response(
            {
                "error": (
                    "project_id must be a valid integer."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        project = get_project_by_id(
            project_id
        )

    except Exception:

        import traceback

        traceback.print_exc()

        return Response(
            {
                "error": (
                    "Failed to read project data."
                )
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

        # -------------------------------------------------
        # Existing prediction pipeline
        # -------------------------------------------------
        #
        # We use the existing result internally.
        # We DO NOT modify or return it here.
        #

        prediction_result = predict_project(
            project
        )

        early_warning = generate_early_warning(
            prediction_result
        )

        return Response(
            {
                "project_id": project_id,
                "early_warning": early_warning
            },
            status=status.HTTP_200_OK
        )

    except Exception:

        import traceback

        traceback.print_exc()

        return Response(
            {
                "error": (
                    "Early warning generation failed."
                )
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
@api_view(["GET"])
@authentication_classes([AIMLAPIKeyAuthentication])
def project_recommendation_api(request):

    project_id = request.query_params.get("project_id")

    if not project_id:
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

    project = get_project_by_id(project_id)

    if not project:
        return Response(
            {
                "error": "Project not found."
            },
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        prediction_result = predict_project(project)

        recommendation_result = (
            generate_project_recommendations(
                prediction_result
            )
        )

        return Response(
            {
                "project_id": project_id,
                "recommendations": recommendation_result,
            },
            status=status.HTTP_200_OK
        )

    except Exception as exc:
        return Response(
            {
                "error": "Unable to generate recommendations.",
                "details": str(exc),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
@api_view(["GET"])
@authentication_classes([AIMLAPIKeyAuthentication])
def project_benchmarking_api(request):

    project_id = request.query_params.get(
        "project_id"
    )

    if not project_id:
        return Response(
            {
                "error": "project_id is required."
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        project_id = int(project_id)

    except (TypeError, ValueError):
        return Response(
            {
                "error":
                    "project_id must be a valid integer."
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:

        project = get_project_by_id(
            project_id
        )

        if project is None:
            return Response(
                {
                    "error": "Project not found.",
                    "project_id": project_id,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Build exactly the same project text
        project_text = _build_project_text(
            project
        )

        # Generate embedding only for this
        # separate analytics endpoint.
        embedding = generate_embedding(
            project_text
        )

        # Existing historical retrieval.
        # READ ONLY.
        completed_similar_projects = (
            retrieve_completed_similar_projects(
                project,
                embedding,
                limit=50,
            )
        )

        benchmarking = generate_benchmarking(
            project=project,
            completed_similar_projects=(
                completed_similar_projects
            ),
        )

        return Response(
            {
                "project_id": project_id,
                "benchmarking": benchmarking,
            },
            status=status.HTTP_200_OK,
        )

    except ValueError as exc:

        return Response(
            {
                "error": str(exc)
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Exception:

        import traceback

        traceback.print_exc()

        return Response(
            {
                "error":
                    "Benchmarking generation failed."
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# =========================================================
# PROJECT INTELLIGENCE ASSISTANT
# =========================================================
# POST /api/aiml/project-assistant/
#
# Supported modes:
# 1. project_id + message -> backend generates current AI/ML analysis.
# 2. analysis + message -> reuse an existing analysis response.
# 3. projects + message -> compare supplied project analyses.
# =========================================================

def _enrich_result_for_assistant(result, project):
    """Add the existing cost-escalation analysis to assistant context."""
    if not isinstance(result, dict):
        return result

    cost_result = result.get("cost_prediction")
    if not isinstance(cost_result, dict):
        return result

    if "cost_escalation_analysis" in cost_result:
        return result

    historical_projects = cost_result.get("historical_projects") or []
    historical_overruns = [
        item.get("cost_overrun_percent")
        for item in historical_projects
        if isinstance(item, dict)
        and item.get("cost_overrun_percent") is not None
    ]

    try:
        historical_average_overrun = (
            sum(float(value) for value in historical_overruns)
            / len(historical_overruns)
            if historical_overruns
            else None
        )
    except (TypeError, ValueError):
        historical_average_overrun = None

    enriched_cost = dict(cost_result)
    enriched_cost["cost_escalation_analysis"] = _build_cost_escalation_analysis(
        project=project,
        predicted_overrun=cost_result.get("predicted_cost_overrun_percent"),
        spread=cost_result.get("historical_spread_percent"),
        average_similarity=cost_result.get("average_similarity", 0.0),
        historical_average_overrun=historical_average_overrun,
        confidence=cost_result.get("confidence", "LOW"),
    )

    enriched_result = dict(result)
    enriched_result["cost_prediction"] = enriched_cost
    return enriched_result


@lru_cache(maxsize=128)
def _build_analysis_from_project_id(project_id):
    project = get_project_by_id(project_id)
    if project is None:
        return None, None

    # Cache the expensive ML analysis for repeated assistant questions.
    # This does not change /predict-project/ behavior or its response.
    result = predict_project(project)
    result = _enrich_result_for_assistant(result, project)
    return project, result


@api_view(["POST"])
@authentication_classes([AIMLAPIKeyAuthentication])
def project_assistant_api(request):
    body = request.data if isinstance(request.data, dict) else {}
    message = body.get("message") or body.get("question")

    if not isinstance(message, str) or not message.strip():
        return Response(
            {"error": "message is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        analysis = body.get("analysis")
        projects = body.get("projects")
        project_id = body.get("project_id")

        if project_id not in (None, ""):
            try:
                project_id = int(project_id)
            except (TypeError, ValueError):
                return Response(
                    {"error": "project_id must be a valid integer."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            project, generated_analysis = _build_analysis_from_project_id(project_id)
            if project is None:
                return Response(
                    {
                        "error": "Project not found.",
                        "project_id": project_id,
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            analysis = generated_analysis

        if analysis is not None and not isinstance(analysis, dict):
            return Response(
                {"error": "analysis must be a JSON object."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if projects is not None and not isinstance(projects, list):
            return Response(
                {"error": "projects must be a JSON array."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if analysis is None and not projects:
            return Response(
                {
                    "error": (
                        "Provide project_id, analysis, or projects as context."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = ask_project_assistant(
            question=message,
            analysis=analysis,
            projects=projects,
        )

        return Response(
            {
                "question": message.strip(),
                "answer": result["answer"],
                "project_id": project_id,
            },
            status=status.HTTP_200_OK,
        )

    except AssistantConfigurationError as exc:
        return Response(
            {
                "error": str(exc),
                "code": "LLM_NOT_CONFIGURED",
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except AssistantProviderError as exc:
        return Response(
            {
                "error": str(exc),
                "code": "LLM_PROVIDER_ERROR",
            },
            status=status.HTTP_502_BAD_GATEWAY,
        )
    except ValueError as exc:
        return Response(
            {"error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return Response(
            {"error": "Project assistant failed."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
