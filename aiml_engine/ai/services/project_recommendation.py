def generate_project_recommendations(
    prediction_result,
    similar_projects=None,
    completed_similar_projects=None,
):
    """
    Generate actionable recommendations using
    existing AI/ML prediction and historical evidence.

    READ ONLY.
    """

    cost = prediction_result.get("cost_prediction", {})
    time = prediction_result.get("time_prediction", {})
    risk = prediction_result.get("risk", {})

    recommendations = []

    # -------------------------------------------------
    # 1. COST RECOMMENDATION
    # -------------------------------------------------

    overrun = cost.get("predicted_cost_overrun_percent")

    if overrun is not None:
        if overrun > 25:
            recommendations.append({
                "priority": "HIGH",
                "area": "COST",
                "recommendation": (
                    "Conduct an immediate budget review and "
                    "closely monitor expenditure against the "
                    "original project cost."
                )
            })

        elif overrun > 10:
            recommendations.append({
                "priority": "MEDIUM",
                "area": "COST",
                "recommendation": (
                    "Monitor expenditure regularly and review "
                    "major cost drivers before further budget "
                    "commitments."
                )
            })

    # -------------------------------------------------
    # 2. TIME RECOMMENDATION
    # -------------------------------------------------

    delay = time.get("predicted_delay_days")

    if delay is not None:

        if delay > 730:
            recommendations.append({
                "priority": "CRITICAL",
                "area": "SCHEDULE",
                "recommendation": (
                    "Perform an immediate schedule recovery "
                    "review and identify critical activities "
                    "causing potential long-term delay."
                )
            })

        elif delay > 365:
            recommendations.append({
                "priority": "HIGH",
                "area": "SCHEDULE",
                "recommendation": (
                    "Review the project execution schedule and "
                    "prepare corrective actions for activities "
                    "with high delay exposure."
                )
            })

        elif delay > 180:
            recommendations.append({
                "priority": "MEDIUM",
                "area": "SCHEDULE",
                "recommendation": (
                    "Increase milestone-level monitoring and "
                    "review activities that may affect the "
                    "planned completion date."
                )
            })

    # -------------------------------------------------
    # 3. RISK RECOMMENDATION
    # -------------------------------------------------

    risk_level = risk.get("risk_level")

    if risk_level == "CRITICAL":
        recommendations.append({
            "priority": "CRITICAL",
            "area": "RISK",
            "recommendation": risk.get(
                "recommended_solution",
                "Immediate senior-level intervention is recommended."
            )
        })

    elif risk_level == "HIGH":
        recommendations.append({
            "priority": "HIGH",
            "area": "RISK",
            "recommendation": risk.get(
                "recommended_solution",
                "Enhanced project monitoring and corrective action are recommended."
            )
        })

    # -------------------------------------------------
    # 4. CONFIDENCE / EVIDENCE
    # -------------------------------------------------

    cost_confidence = cost.get("confidence")
    time_confidence = time.get("confidence")

    if cost_confidence == "LOW":
        recommendations.append({
            "priority": "MEDIUM",
            "area": "COST",
            "recommendation": (
                "Validate the cost forecast against current "
                "expenditure because historical evidence is limited."
            )
        })

    if time_confidence == "LOW":
        recommendations.append({
            "priority": "MEDIUM",
            "area": "SCHEDULE",
            "recommendation": (
                "Validate the projected timeline against "
                "current project milestones because prediction "
                "confidence is limited."
            )
        })

    # -------------------------------------------------
    # 5. HISTORICAL EVIDENCE
    # -------------------------------------------------

    historical_count = time.get(
        "historical_projects_used",
        0
    )

    average_similarity = time.get(
        "average_similarity",
        0
    )

    historical_insight = None

    if historical_count >= 5 and average_similarity >= 0.75:
        historical_insight = (
            "Strong historical evidence is available from "
            "comparable completed projects."
        )

    elif historical_count > 0:
        historical_insight = (
            "Some historical evidence is available, but "
            "recommendations should be validated against "
            "current project conditions."
        )

    else:
        historical_insight = (
            "Limited comparable historical evidence is available."
        )

    # -------------------------------------------------
    # 6. DEFAULT RECOMMENDATION
    # -------------------------------------------------

    if not recommendations:
        recommendations.append({
            "priority": "LOW",
            "area": "MONITORING",
            "recommendation": (
                "Continue regular monitoring of project cost, "
                "schedule and execution milestones."
            )
        })

    # -------------------------------------------------
    # 7. SORT BY PRIORITY
    # -------------------------------------------------

    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    recommendations.sort(
        key=lambda item: priority_order.get(
            item.get("priority"),
            3
        )
    )

    return {
        "recommendations": recommendations,
        "historical_insight": historical_insight,
        "recommendation_count": len(recommendations),
    }