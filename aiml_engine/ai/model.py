class AIModel:
    """
    Interface for the AI model.

    Later this will load the fine-tuned/local model
    and generate risk, reason, issue assessment
    and solutions.
    """

    def load(self):
        pass

    def analyze(self, project_data, context=None):
        return {
            "risk": None,
            "reason": None,
            "issue_assessment": None,
            "solution": None,
        }