"""
Early Warning Service

This module is completely independent from the existing
prediction response.

IMPORTANT:
- Existing prediction APIs are NOT modified.
- Existing prediction output is NOT modified.
- This service only interprets the existing prediction result.
"""


def generate_early_warning(prediction_result):
    """
    Generate structured early-warning information from
    the existing prediction result.

    No ML model is changed.
    No database write is performed.
    """

    cost_prediction = prediction_result.get(
        "cost_prediction",
        {}
    )

    time_prediction = prediction_result.get(
        "time_prediction",
        {}
    )

    risk = prediction_result.get(
        "risk",
        {}
    )

    warnings = []

    # =====================================================
    # 1. DETECT ISSUES ALREADY IDENTIFIED BY RISK ENGINE
    # =====================================================

    detected_issues = risk.get(
        "detected_issues",
        []
    )

    if detected_issues:
        warnings.extend(detected_issues)

    # =====================================================
    # 2. TIME WARNING
    # =====================================================

    predicted_delay = time_prediction.get(
        "predicted_delay_days"
    )

    if predicted_delay is not None:

        try:
            predicted_delay = float(
                predicted_delay
            )

            if predicted_delay >= 1095:
                if "SEVERE_TIME_DELAY" not in warnings:
                    warnings.append(
                        "SEVERE_TIME_DELAY"
                    )

            elif predicted_delay >= 730:
                if "HIGH_TIME_DELAY" not in warnings:
                    warnings.append(
                        "HIGH_TIME_DELAY"
                    )

            elif predicted_delay >= 365:
                if "TIME_DELAY" not in warnings:
                    warnings.append(
                        "TIME_DELAY"
                    )

        except (TypeError, ValueError):
            pass

    # =====================================================
    # 3. COST UNCERTAINTY
    # =====================================================

    cost_confidence = cost_prediction.get(
        "confidence"
    )

    historical_spread = cost_prediction.get(
        "historical_spread_percent"
    )

    if cost_confidence == "LOW":

        if "LOW_PREDICTION_CONFIDENCE" not in warnings:
            warnings.append(
                "LOW_PREDICTION_CONFIDENCE"
            )

    if historical_spread is not None:

        try:
            historical_spread = float(
                historical_spread
            )

            if historical_spread > 100:

                if "HIGH_COST_UNCERTAINTY" not in warnings:
                    warnings.append(
                        "HIGH_COST_UNCERTAINTY"
                    )

        except (TypeError, ValueError):
            pass

    # =====================================================
    # 4. TIME CONFIDENCE
    # =====================================================

    time_confidence = time_prediction.get(
        "confidence"
    )

    if time_confidence == "LOW":

        if "LOW_TIME_PREDICTION_CONFIDENCE" not in warnings:
            warnings.append(
                "LOW_TIME_PREDICTION_CONFIDENCE"
            )

    # =====================================================
    # 5. DETERMINE SEVERITY
    # =====================================================

    risk_level = str(
        risk.get(
            "risk_level",
            "LOW"
        )
    ).upper()

    if risk_level == "CRITICAL":

        severity = "CRITICAL"

    elif risk_level == "HIGH":

        severity = "HIGH"

    elif risk_level == "MEDIUM":

        severity = "MEDIUM"

    else:

        severity = "LOW"

    # =====================================================
    # 6. DETERMINE ALERT
    # =====================================================

    alert = (
        severity in {
            "HIGH",
            "CRITICAL"
        }
        or len(warnings) >= 2
    )

    # =====================================================
    # 7. EXPECTED IMPACT
    # =====================================================

    if severity == "CRITICAL":

        expected_impact = (
            "The project shows critical warning signals "
            "that may materially affect cost, schedule, "
            "or execution outcomes."
        )

    elif severity == "HIGH":

        expected_impact = (
            "The project shows significant warning signals "
            "that may affect the planned execution timeline "
            "and project performance."
        )

    elif severity == "MEDIUM":

        expected_impact = (
            "The project shows moderate warning signals "
            "that should be monitored closely."
        )

    else:

        expected_impact = (
            "No major early-warning condition is currently "
            "indicated by the available prediction signals."
        )

    # =====================================================
    # 8. RECOMMENDED INTERVENTION
    # =====================================================

    recommended_intervention = risk.get(
        "recommended_solution"
    )

    if not recommended_intervention:

        recommended_intervention = (
            "Continue regular project monitoring and "
            "review cost, schedule and physical progress "
            "against planned milestones."
        )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "alert": alert,

        "severity": severity,

        "warnings": list(
            dict.fromkeys(warnings)
        ),

        "expected_impact": expected_impact,

        "recommended_intervention": (
            recommended_intervention
        ),
    }