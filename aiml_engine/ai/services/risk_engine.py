def calculate_risk(
    project,
    predicted_delay_days=0,
    similar_projects=None,
    completed_similar_projects=None,
    predicted_cost_overrun_percent=0,
    cost_prediction_confidence="LOW",
    time_prediction_confidence="LOW",
    cost_range=None,
    time_range=None,
):
    """
    Risk Engine V3

    Uses:
    - physical progress
    - predicted time delay
    - predicted cost overrun
    - prediction confidence
    - prediction uncertainty
    - historical similarity evidence

    This function is completely read-only.
    """

    risk_score = 0
    issues = []

    similar_projects = similar_projects or []
    completed_similar_projects = completed_similar_projects or []

    physical_progress = float(
        project.get("physical_progress") or 0
    )

    predicted_delay_days = float(
        predicted_delay_days or 0
    )

    predicted_cost_overrun_percent = float(
        predicted_cost_overrun_percent or 0
    )

    # =========================================================
    # 1. PHYSICAL PROGRESS
    # =========================================================

    if physical_progress < 25:
        risk_score += 3
        issues.append("LOW_PHYSICAL_PROGRESS")

    elif physical_progress < 50:
        risk_score += 2
        issues.append("MODERATE_PHYSICAL_PROGRESS")

    elif physical_progress < 75:
        risk_score += 1
        issues.append("BELOW_TARGET_PHYSICAL_PROGRESS")

    # =========================================================
    # 2. TIME DELAY
    # =========================================================

    if predicted_delay_days > 1095:
        risk_score += 4
        issues.append("SEVERE_TIME_DELAY")

    elif predicted_delay_days > 730:
        risk_score += 3
        issues.append("HIGH_TIME_DELAY")

    elif predicted_delay_days > 365:
        risk_score += 2
        issues.append("TIME_DELAY")

    elif predicted_delay_days > 90:
        risk_score += 1
        issues.append("MINOR_TIME_DELAY")

    # =========================================================
    # 3. COST OVERRUN
    # =========================================================

    if predicted_cost_overrun_percent > 50:
        risk_score += 4
        issues.append("SEVERE_COST_OVERRUN")

    elif predicted_cost_overrun_percent > 25:
        risk_score += 3
        issues.append("HIGH_COST_OVERRUN")

    elif predicted_cost_overrun_percent > 10:
        risk_score += 2
        issues.append("COST_OVERRUN")

    elif predicted_cost_overrun_percent > 5:
        risk_score += 1
        issues.append("MINOR_COST_OVERRUN")

    # =========================================================
    # 4. COST UNCERTAINTY
    # =========================================================

    if cost_range:
        min_cost = cost_range.get("min_cost")
        max_cost = cost_range.get("max_cost")

        original_cost = project.get("original_cost")

        if (
            min_cost is not None
            and max_cost is not None
            and original_cost is not None
            and float(original_cost) > 0
        ):
            original_cost = float(original_cost)
            min_cost = float(min_cost)
            max_cost = float(max_cost)

            range_width_percent = (
                (max_cost - min_cost)
                / original_cost
            ) * 100

            if range_width_percent > 100:
                risk_score += 2
                issues.append("HIGH_COST_UNCERTAINTY")

            elif range_width_percent > 50:
                risk_score += 1
                issues.append("COST_UNCERTAINTY")

    # =========================================================
    # 5. TIME UNCERTAINTY
    # =========================================================

    if time_range:
        min_days = time_range.get("min_days")
        max_days = time_range.get("max_days")

        if (
            min_days is not None
            and max_days is not None
        ):
            time_range_width = (
                float(max_days)
                - float(min_days)
            )

            if time_range_width > 1800:
                risk_score += 2
                issues.append("HIGH_TIME_UNCERTAINTY")

            elif time_range_width > 730:
                risk_score += 1
                issues.append("TIME_UNCERTAINTY")

    # =========================================================
    # 6. LOW CONFIDENCE
    # =========================================================

    if (
        cost_prediction_confidence == "LOW"
        and time_prediction_confidence == "LOW"
    ):
        risk_score += 1
        issues.append("LOW_PREDICTION_CONFIDENCE")

    # =========================================================
    # 7. HISTORICAL SIMILARITY
    # =========================================================

    high_similarity_historical = [
        item
        for item in completed_similar_projects
        if float(item.get("similarity", 0)) >= 0.75
    ]

    if len(high_similarity_historical) >= 5:
        risk_score += 1
        issues.append("STRONG_HISTORICAL_EVIDENCE")

    elif len(high_similarity_historical) >= 3:
        risk_score += 1
        issues.append("SIMILAR_HISTORICAL_PROJECTS")

    # =========================================================
    # 8. COMBINED SIGNALS
    # =========================================================

    severe_time = (
        "SEVERE_TIME_DELAY" in issues
        or "HIGH_TIME_DELAY" in issues
    )

    cost_pressure = (
        "SEVERE_COST_OVERRUN" in issues
        or "HIGH_COST_OVERRUN" in issues
        or "COST_OVERRUN" in issues
        or "MINOR_COST_OVERRUN" in issues
    )

    progress_problem = (
        "LOW_PHYSICAL_PROGRESS" in issues
        or "MODERATE_PHYSICAL_PROGRESS" in issues
    )

    if severe_time and cost_pressure:
        risk_score += 2
        issues.append("TIME_AND_COST_PRESSURE")

    if progress_problem and severe_time:
        risk_score += 2
        issues.append("EXECUTION_DELAY_RISK")

    # =========================================================
    # 9. RISK LEVEL
    # =========================================================

    if risk_score >= 10:
        risk_level = "CRITICAL"

    elif risk_score >= 7:
        risk_level = "HIGH"

    elif risk_score >= 4:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # =========================================================
    # 10. PRIMARY ISSUE
    # =========================================================

    priority_order = [
        "TIME_AND_COST_PRESSURE",
        "EXECUTION_DELAY_RISK",
        "SEVERE_TIME_DELAY",
        "HIGH_TIME_DELAY",
        "SEVERE_COST_OVERRUN",
        "HIGH_COST_OVERRUN",
        "LOW_PHYSICAL_PROGRESS",
        "MODERATE_PHYSICAL_PROGRESS",
        "HIGH_COST_UNCERTAINTY",
        "HIGH_TIME_UNCERTAINTY",
        "COST_OVERRUN",
        "TIME_DELAY",
        "COST_UNCERTAINTY",
        "TIME_UNCERTAINTY",
        "MINOR_COST_OVERRUN",
        "MINOR_TIME_DELAY",
        "LOW_PREDICTION_CONFIDENCE",
        "STRONG_HISTORICAL_EVIDENCE",
        "SIMILAR_HISTORICAL_PROJECTS",
        "BELOW_TARGET_PHYSICAL_PROGRESS",
    ]

    issue_id = "NO_MAJOR_ISSUE"

    for issue in priority_order:
        if issue in issues:
            issue_id = issue
            break

    # =========================================================
    # 11. REASON
    # =========================================================

    if issues:
        reason = (
            "Risk identified based on: "
            + ", ".join(issues)
            + "."
        )
    else:
        reason = (
            "No major risk indicators detected "
            "from the currently available project data."
        )

    # =========================================================
    # 12. RECOMMENDED SOLUTIONS
    # =========================================================

    solutions = []

    if (
        "LOW_PHYSICAL_PROGRESS" in issues
        or "MODERATE_PHYSICAL_PROGRESS" in issues
    ):
        solutions.append(
            "Review project execution and identify "
            "the causes of slow physical progress."
        )

    if (
        "SEVERE_TIME_DELAY" in issues
        or "HIGH_TIME_DELAY" in issues
        or "TIME_DELAY" in issues
        or "MINOR_TIME_DELAY" in issues
    ):
        solutions.append(
            "Prepare a time-recovery plan and closely "
            "monitor project milestones."
        )

    if (
        "SEVERE_COST_OVERRUN" in issues
        or "HIGH_COST_OVERRUN" in issues
        or "COST_OVERRUN" in issues
        or "MINOR_COST_OVERRUN" in issues
    ):
        solutions.append(
            "Review cost escalation drivers and "
            "strengthen cost monitoring."
        )

    if (
        "HIGH_COST_UNCERTAINTY" in issues
        or "COST_UNCERTAINTY" in issues
    ):
        solutions.append(
            "Perform a detailed cost review because "
            "historical cost outcomes show high uncertainty."
        )

    if (
        "HIGH_TIME_UNCERTAINTY" in issues
        or "TIME_UNCERTAINTY" in issues
    ):
        solutions.append(
            "Perform milestone-level schedule analysis "
            "because the predicted completion delay has high uncertainty."
        )

    if "TIME_AND_COST_PRESSURE" in issues:
        solutions.append(
            "Conduct an integrated cost and schedule "
            "review and prioritize corrective actions."
        )

    if "EXECUTION_DELAY_RISK" in issues:
        solutions.append(
            "Escalate execution monitoring and establish "
            "short-term milestone recovery targets."
        )

    if (
        "STRONG_HISTORICAL_EVIDENCE" in issues
        or "SIMILAR_HISTORICAL_PROJECTS" in issues
    ):
        solutions.append(
            "Compare the project with similar historical "
            "projects and apply relevant corrective actions."
        )

    if not solutions:
        solutions.append(
            "Continue regular project monitoring."
        )

    solutions = list(dict.fromkeys(solutions))

    # =========================================================
    # 13. FINAL RESPONSE
    # =========================================================

    return {
        "risk_level": risk_level,
        "issue_id": issue_id,
        "reason": reason,
        "recommended_solution": " ".join(solutions),
        "risk_score": risk_score,
        "detected_issues": issues,
    }
