from django.urls import path

from . import views


urlpatterns = [
    path(
        "health/",
        views.health_check,
        name="health-check",
    ),

    path(
        "predict-project/",
        views.predict_project_api,
        name="predict-project",
    ),

    path(
        "predict-new-project/",
        views.predict_new_project_api,
        name="predict-new-project",
    ),
    path(
        "early-warning/",
        views.early_warning_api,
        name="early-warning"
),
path(
    "project-recommendation/",
    views.project_recommendation_api,
    name="project-recommendation",
),
path(
    "project-benchmarking/",
    views.project_benchmarking_api,
    name="project-benchmarking",
),
]