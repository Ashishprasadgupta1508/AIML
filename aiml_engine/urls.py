from django.urls import path

from .views import analyze_project, test_database


urlpatterns = [
    path(
        "analyze/<int:project_id>/",
        analyze_project,
        name="analyze_project"
    ),
    path(
        "test-db/",
        test_database,
        name="test_database"
    ),
]

