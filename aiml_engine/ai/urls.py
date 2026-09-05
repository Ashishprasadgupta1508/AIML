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
]