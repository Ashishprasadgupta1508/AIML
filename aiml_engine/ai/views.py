from rest_framework.decorators import (
    api_view,
    authentication_classes,
)
from rest_framework.response import Response
from rest_framework import status

from .authentication import AIMLAPIKeyAuthentication

from aiml_engine.ai.services.prediction_service import predict_project
from aiml_engine.ai.services.project_reader import get_project_by_id


@api_view(["GET"])
def health_check(request):
    return Response({
        "status": "ok",
        "service": "Django AI/ML Engine"
    })


@api_view(["POST"])
@authentication_classes([AIMLAPIKeyAuthentication])
def predict_project_api(request):

    project_id = request.data.get("project_id")

    if project_id is None:
        return Response(
            {"error": "project_id is required."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        project_id = int(project_id)
    except (TypeError, ValueError):
        return Response(
            {"error": "project_id must be a valid integer."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        project = get_project_by_id(project_id)

    except Exception:
        return Response(
            {"error": "Failed to read project data."},
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
            {"error": "Prediction failed."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )