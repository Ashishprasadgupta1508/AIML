from rest_framework.decorators import (
    api_view,
    authentication_classes,
)
from functools import lru_cache
import time
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

        # Reuse the prediction result already computed by this request.
        # This does not change the prediction response.
        if isinstance(result, dict):
            _cache_assistant_analysis(project_id, result)

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


def _fmt_number(value, decimals=1):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"


def _fmt_currency(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"₹{number:,.2f}"


def _build_local_project_assistant_answer(analysis, project=None):
    """Deterministic fallback when Gemini is unavailable/quota-exhausted.

    Uses only the already-computed project analysis. This keeps the endpoint
    useful even when the external LLM provider returns 429/503/5xx.
    """
    analysis = analysis if isinstance(analysis, dict) else {}
    project_block = analysis.get("project") or {}

    def _project_value(*names):
        if isinstance(project, dict):
            for name in names:
                value = project.get(name)
                if value not in (None, ""):
                    return value
        else:
            for name in names:
                value = getattr(project, name, None)
                if value not in (None, ""):
                    return value
        for name in names:
            if isinstance(project_block, dict):
                value = project_block.get(name)
                if value not in (None, ""):
                    return value
        return None

    project_name = _project_value("project_name", "name") or "Infrastructure Project"
    project_id = _project_value("project_id", "id") or "N/A"

    cost = analysis.get("cost_prediction") or {}
    time_data = analysis.get("time_prediction") or {}
    risk = analysis.get("risk") or {}

    risk_level = risk.get("risk_level") or "N/A"
    risk_score = risk.get("risk_score")
    physical_progress = _project_value("physical_progress", "physical_progress_percent")

    predicted_cost = cost.get("predicted_final_cost")
    cost_overrun = cost.get("predicted_cost_overrun_percent")
    cost_range = cost.get("expected_cost_range") or {}
    delay = time_data.get("predicted_delay_days")
    delay_range = time_data.get("expected_delay_range") or {}
    planned_duration = time_data.get("planned_duration_days")
    cost_confidence = cost.get("confidence") or "N/A"
    time_confidence = time_data.get("confidence") or "N/A"
    historical_used = cost.get("historical_projects_used") or cost.get("historical_projects_found")
    similarity = cost.get("average_similarity")
    spread = cost.get("historical_spread_percent")
    escalation = cost.get("cost_escalation_analysis") or {}

    ml_delay = time_data.get("ml_predicted_delay_days")
    historical_delay = time_data.get("historical_predicted_delay_days")
    recommended = risk.get("recommended_solution") or "Review the latest project records and validate the model indicators with current project status."
    detected = risk.get("detected_issues") or []

    executive = (
        f"Project **{project_id}** has an overall risk level of **{risk_level}**"
        + (f" (Risk Score: **{_fmt_number(risk_score, 0)}/10**)" if risk_score is not None else "")
        + ". "
    )
    if physical_progress is not None:
        executive += f"Recorded physical progress is **{_fmt_number(physical_progress, 1)}%**. "
    if delay is not None:
        executive += f"The schedule assessment indicates an estimated delay of approximately **{_fmt_number(delay, 1)} days**. "
    if predicted_cost is not None:
        executive += f"The projected final cost is **{_fmt_currency(predicted_cost)}**."

    lines = [
        "## Project Intelligence Summary",
        "",
        f"**Project Name / ID:** {project_name} | **Project ID:** {project_id}",
        f"**Overall Risk:** **{risk_level}**" + (f" (Risk Score: {_fmt_number(risk_score, 0)}/10)" if risk_score is not None else ""),
        "",
        "### Executive Summary",
        executive,
        "",
        "### Key Risk Indicators",
        "",
        "| Metric | Value | Status |",
        "| :--- | :--- | :--- |",
    ]

    if predicted_cost is not None:
        cost_value = f"{_fmt_currency(predicted_cost)}"
        if cost_overrun is not None:
            cost_value += f" ({_fmt_number(cost_overrun, 2)}%)"
        lines.append(f"| **Predicted Final Cost** | {cost_value} | {cost_confidence} confidence |")
    if delay is not None:
        delay_value = f"{_fmt_number(delay, 1)} days"
        if delay_range:
            delay_value += f" (Range: {_fmt_number(delay_range.get('min_days'), 1)}–{_fmt_number(delay_range.get('max_days'), 1)} days)"
        lines.append(f"| **Predicted Delay** | {delay_value} | {time_confidence} confidence |")
    if physical_progress is not None:
        lines.append(f"| **Physical Progress** | {_fmt_number(physical_progress, 1)}% | Recorded |")
    if planned_duration is not None:
        lines.append(f"| **Planned Duration** | {_fmt_number(planned_duration, 1)} days | Baseline |")
    if escalation:
        impact = escalation.get("impact") or "N/A"
        explanation = escalation.get("explanation") or escalation.get("code") or "Recorded cost pressure"
        lines.append(f"| **Cost Escalation** | {explanation} | {impact} impact |")
    if spread is not None:
        lines.append(f"| **Historical Cost Spread** | {_fmt_number(spread, 2)}% | Uncertainty indicator |")

    lines += ["", "### Major Risks"]
    if detected:
        for issue in detected[:6]:
            lines.append(f"- **{issue}**")
    else:
        if delay is not None and delay > 0:
            lines.append(f"- **Schedule Delay:** approximately {_fmt_number(delay, 1)} days projected.")
        if spread is not None and spread > 50:
            lines.append(f"- **Cost Uncertainty:** historical spread of {_fmt_number(spread, 2)}%.")
        if not detected and not lines[-1].startswith("-"):
            lines.append("- No additional risk indicators were available in the supplied analysis.")

    lines += ["", "### Expected Impact"]
    if delay is not None and delay > 0:
        lines.append(f"- **Timeline:** The projected delay of approximately {_fmt_number(delay, 1)} days may materially affect the planned completion schedule.")
    if predicted_cost is not None and cost_range:
        lines.append(
            f"- **Financial:** The estimated final cost is {_fmt_currency(predicted_cost)}, with a modeled range of "
            f"{_fmt_currency(cost_range.get('min_cost'))} to {_fmt_currency(cost_range.get('max_cost'))}."
        )
    if not any(line.startswith("-") for line in lines[lines.index("### Expected Impact") + 1:]):
        lines.append("- Impact should be validated against the latest project records.")

    lines += ["", "### Recommended Management Actions"]
    for item in [recommended]:
        if item:
            for sentence in str(item).replace("\n", " ").split(". "):
                sentence = sentence.strip().rstrip(".")
                if sentence:
                    lines.append(f"- {sentence}.")
    lines += [
        "- Reconcile current project status with the model's schedule and financial indicators.",
        "- Continue monitoring expenditure, pending milestones, and closure activities.",
    ]

    lines += ["", "### Historical Evidence"]
    if historical_used is not None:
        evidence = f"- Analysis uses **{_fmt_number(historical_used, 0)}** comparable historical project(s)."
        if similarity is not None:
            evidence += f" Average similarity is **{_fmt_number(similarity, 4)}**."
        lines.append(evidence)
    if ml_delay is not None or historical_delay is not None:
        lines.append(
            f"- Schedule evidence: ML estimate **{_fmt_number(ml_delay, 1)} days** vs historical estimate **{_fmt_number(historical_delay, 1)} days**."
        )
    if spread is not None:
        lines.append(f"- Historical cost outcome spread is **{_fmt_number(spread, 2)}%**.")
    if len(lines) == 0:
        lines.append("- No historical evidence was available in the supplied analysis.")

    lines += ["", "### Overall Assessment"]
    lines.append(
        f"Project **{project_id}** currently presents a **{risk_level}** risk profile. "
        "Management should validate the model indicators against the latest project records and prioritize the highest-risk schedule and financial issues."
    )

    return "\n".join(lines)


@lru_cache(maxsize=256)
def _get_assistant_project_context(project_id):
    """Cheap project lookup. Never run full ML inference here."""
    return get_project_by_id(project_id)


_ASSISTANT_ANALYSIS_CACHE = {}
_ASSISTANT_ANALYSIS_CACHE_MAX = 128
_ASSISTANT_ANALYSIS_CACHE_TTL = 300

def _cache_assistant_analysis(project_id, analysis):
    if not isinstance(analysis, dict):
        return
    if len(_ASSISTANT_ANALYSIS_CACHE) >= _ASSISTANT_ANALYSIS_CACHE_MAX:
        oldest_key = next(iter(_ASSISTANT_ANALYSIS_CACHE), None)
        if oldest_key is not None:
            _ASSISTANT_ANALYSIS_CACHE.pop(oldest_key, None)
    _ASSISTANT_ANALYSIS_CACHE[project_id] = (time.monotonic(), analysis)

def _get_cached_assistant_analysis(project_id):
    item = _ASSISTANT_ANALYSIS_CACHE.get(project_id)
    if not item:
        return None
    created, analysis = item
    if time.monotonic() - created > _ASSISTANT_ANALYSIS_CACHE_TTL:
        _ASSISTANT_ANALYSIS_CACHE.pop(project_id, None)
        return None
    return analysis

def _build_analysis_from_project_id(project_id):
    """Cheap assistant context only; never invoke predict_project()."""
    project = _get_assistant_project_context(project_id)
    if project is None:
        return None, None

    return project, _get_cached_assistant_analysis(project_id)


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

            # Explicit analysis always wins; otherwise reuse cached ML output.
            if analysis is None and generated_analysis is not None:
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

        if analysis is None and not projects and project is not None:
            # No cached prediction yet: use only the current project record.
            # Crucially, do not trigger the expensive ML pipeline.
            analysis = {
                "project": project,
                "source": "current_project_record_only",
            }

        if analysis is None and not projects:
            return Response(
                {
                    "error": (
                        "Provide project_id, analysis, or projects as context."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Fast path: common risk/action questions are answered directly from
        # the existing ML analysis. This avoids an unnecessary LLM round-trip
        # and keeps the common Project Assistant path sub-5-second.
        normalized_message = message.strip().lower()
        fast_keywords = (
            "risk", "action", "recommend", "recommendation",
            "what should be done", "what should we do",
            "next step", "next steps", "delay", "cost overrun",
        )
        use_fast_path = bool(analysis) and any(k in normalized_message for k in fast_keywords)

        try:
            if use_fast_path:
                answer = _build_local_project_assistant_answer(
                    analysis=analysis,
                    project=project if "project" in locals() else None,
                )
            else:
                result = ask_project_assistant(
                    question=message,
                    analysis=analysis,
                    projects=projects,
                )
                answer = result["answer"]
        except (AssistantConfigurationError, AssistantProviderError):
            answer = _build_local_project_assistant_answer(
                analysis=analysis,
                project=project if "project" in locals() else None,
            )
        except Exception:
            # Never let formatting/provider edge cases turn the assistant into
            # a 500. Return a deterministic answer from available ML data.
            answer = _build_local_project_assistant_answer(
                analysis=analysis,
                project=project if "project" in locals() else None,
            )

        return Response(
            {
                "question": message.strip(),
                "answer": answer,
                "project_id": project_id,
            },
            status=status.HTTP_200_OK,
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
