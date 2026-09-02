def preprocess_project_data(project_data):
    """
    Clean and prepare project data
    before sending it to AI/ML models.
    """

    if not project_data:
        return {}

    cleaned_data = {}

    for key, value in project_data.items():
        if value is not None:
            cleaned_data[key] = value

    return cleaned_data