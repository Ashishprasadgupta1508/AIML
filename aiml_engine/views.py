from django.http import JsonResponse
from .ai.analysis import generate_analysis
from .ai.database import get_connection


def analyze_project(request, project_id):

    # Temporary data.
    # Later this will come from PostgreSQL.
    project_data = {
        "project_id": project_id
    }

    result = generate_analysis(project_data)

    return JsonResponse({
        "project_id": project_id,
        "analysis": result
    })

def test_database(request):

    try:
        connection = get_connection()
        connection.close()

        return JsonResponse({
            "status": "success",
            "message": "PostgreSQL connection successful"
        })

    except Exception as error:

        return JsonResponse({
            "status": "error",
            "message": str(error)
        }, status=500)