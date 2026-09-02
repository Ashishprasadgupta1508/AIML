from .preprocessing import preprocess_project_data
from .prediction import predict_project
from .model import AIModel


def generate_analysis(project_data):
    """
    Complete project AI analysis pipeline.
    """

    # 1. Clean/prepare data
    cleaned_data = preprocess_project_data(project_data)

    # 2. Numerical predictions
    predictions = predict_project(cleaned_data)

    # 3. AI analysis
    ai_model = AIModel()
    ai_analysis = ai_model.analyze(cleaned_data)

    # 4. Final response
    return {
        "cost_estimation": predictions["predicted_cost"],
        "time_estimation": predictions["predicted_time"],
        "risk": ai_analysis["risk"],
        "reason": ai_analysis["reason"],
        "issue_assessment": ai_analysis["issue_assessment"],
        "solution": ai_analysis["solution"],
    }