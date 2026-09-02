from django.urls import path

from .views import (
    predict_project_api,
    health_check,
)

urlpatterns = [
    path(
        "predict/",
        predict_project_api,
        name="predict-project"
    ),
    path(
        "health/",
        health_check,
        name="health-check"
    ),
]