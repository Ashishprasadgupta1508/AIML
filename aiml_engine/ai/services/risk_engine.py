# =========================================================
# RISK ENGINE
# =========================================================


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
    Risk Engine

    Uses:
    - physical progress
    - predicted time delay
    - predicted cost overrun
    - prediction confidence
    - prediction uncertainty
    - historical similarity evidence

    Runtime is completely READ-ONLY.
    No project or ai_prediction table is modified.
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

    original_cost = project.get("original_cost")

    if original_cost is not None:
        original_cost = float(original_cost)

    # =====================================================
    # 1. PHYSICAL PROGRESS
    # =====================================================

    if physical_progress < 25:

        risk_score += 3
        issues.append("LOW_PHYSICAL_PROGRESS")

    elif physical_progress < 50:

        risk_score += 2
        issues.append("MODERATE_PHYSICAL_PROGRESS")

    elif physical_progress < 75:

        risk_score += 1
        issues.append("BELOW_TARGET_PHYSICAL_PROGRESS")

    # =====================================================
    # 2. TIME DELAY
    # =====================================================

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

    # =====================================================
    # 3. COST OVERRUN
    # =====================================================

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

    # =====================================================
    # 4. COST UNCERTAINTY
    # =====================================================

    cost_range_width_percent = 0.0

    if cost_range:

        min_cost = cost_range.get("min_cost")
        max_cost = cost_range.get("max_cost")

        if (
            min_cost is not None
            and max_cost is not None
            and original_cost is not None
            and original_cost > 0
        ):

            min_cost = float(min_cost)
            max_cost = float(max_cost)

            cost_range_width_percent = (
                (max_cost - min_cost)
                / original_cost
            ) * 100

            if cost_range_width_percent > 100:

                risk_score += 2
                issues.append(
                    "HIGH_COST_UNCERTAINTY"
                )

            elif cost_range_width_percent > 50:

                risk_score += 1
                issues.append(
                    "COST_UNCERTAINTY"
                )

    # =====================================================
    # 5. TIME UNCERTAINTY
    # =====================================================

    time_range_width = 0.0

    if time_range:

        min_days = time_range.get("min_days")
        max_days = time_range.get("max_days")

        if (
            min_days is not None
            and max_days is not None
        ):

            min_days = float(min_days)
            max_days = float(max_days)

            time_range_width = max_days - min_days

            if time_range_width > 1800:

                risk_score += 2
                issues.append(
                    "HIGH_TIME_UNCERTAINTY"
                )

            elif time_range_width > 730:

                risk_score += 1
                issues.append(
                    "TIME_UNCERTAINTY"
                )

    # =====================================================
    # 6. LOW CONFIDENCE
    # =====================================================

    if (
        cost_prediction_confidence == "LOW"
        and time_prediction_confidence == "LOW"
    ):

        risk_score += 1
        issues.append(
            "LOW_PREDICTION_CONFIDENCE"
        )

    # =====================================================
    # 7. HISTORICAL EVIDENCE
    # =====================================================

    high_similarity_historical = [
        item
        for item in completed_similar_projects
        if float(item.get("similarity", 0)) >= 0.75
    ]

    historical_count = len(
        completed_similar_projects
    )

    high_similarity_count = len(
        high_similarity_historical
    )

    if high_similarity_count >= 5:

        risk_score += 1
        issues.append(
            "STRONG_HISTORICAL_EVIDENCE"
        )

    elif high_similarity_count >= 3:

        risk_score += 1
        issues.append(
            "SIMILAR_HISTORICAL_PROJECTS"
        )

    # =====================================================
    # 8. COMBINED SIGNALS
    # =====================================================

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
        issues.append(
            "TIME_AND_COST_PRESSURE"
        )

    if progress_problem and severe_time:

        risk_score += 2
        issues.append(
            "EXECUTION_DELAY_RISK"
        )

    # =====================================================
    # 9. RISK LEVEL
    # =====================================================

    if risk_score >= 10:

        risk_level = "CRITICAL"

    elif risk_score >= 7:

        risk_level = "HIGH"

    elif risk_score >= 4:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # =====================================================
    # 10. PRIMARY ISSUE
    # =====================================================

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

    # =====================================================
    # 11. DYNAMIC REASON
    # =====================================================

    reason = _build_dynamic_reason(
        project=project,
        risk_level=risk_level,
        issues=issues,
        predicted_delay_days=predicted_delay_days,
        predicted_cost_overrun_percent=(
            predicted_cost_overrun_percent
        ),
        physical_progress=physical_progress,
        historical_count=historical_count,
        high_similarity_count=high_similarity_count,
        cost_prediction_confidence=(
            cost_prediction_confidence
        ),
        time_prediction_confidence=(
            time_prediction_confidence
        ),
        cost_range_width_percent=(
            cost_range_width_percent
        ),
        time_range_width=time_range_width,
    )

    # =====================================================
    # 12. DYNAMIC SOLUTION
    # =====================================================

    recommended_solution = _build_dynamic_solution(
        project=project,
        issues=issues,
        predicted_delay_days=predicted_delay_days,
        predicted_cost_overrun_percent=(
            predicted_cost_overrun_percent
        ),
        physical_progress=physical_progress,
        historical_count=historical_count,
        high_similarity_count=high_similarity_count,
        cost_prediction_confidence=(
            cost_prediction_confidence
        ),
        time_prediction_confidence=(
            time_prediction_confidence
        ),
        cost_range_width_percent=(
            cost_range_width_percent
        ),
        time_range_width=time_range_width,
    )

    # =====================================================
    # 13. FINAL RESPONSE
    # =====================================================

    return {
        "risk_level": risk_level,
        "issue_id": issue_id,
        "reason": reason,
        "recommended_solution": recommended_solution,
        "risk_score": risk_score,
        "detected_issues": issues,
    }


# =========================================================
# DYNAMIC REASON
# =========================================================

def _build_dynamic_reason(
    project,
    risk_level,
    issues,
    predicted_delay_days,
    predicted_cost_overrun_percent,
    physical_progress,
    historical_count,
    high_similarity_count,
    cost_prediction_confidence,
    time_prediction_confidence,
    cost_range_width_percent,
    time_range_width,
):
    """
    Creates a short, project-specific explanation.

    The wording is based on actual project conditions rather
    than exposing internal issue codes.
    """

    project_name = (
        project.get("project_name")
        or "The project"
    )

    sentences = []

    # -----------------------------------------------------
    # SPECIAL CASE: HIGH PROGRESS + HIGH DELAY
    # -----------------------------------------------------

    if (
        physical_progress >= 90
        and predicted_delay_days > 730
    ):

        sentences.append(
            f"{project_name} has reached "
            f"{physical_progress:.1f}% physical progress, "
            f"but the schedule assessment still indicates "
            f"approximately {predicted_delay_days:.0f} days of delay."
        )

        if historical_count >= 3:

            sentences.append(
                f"The delay assessment is supported by "
                f"{high_similarity_count} highly similar completed "
                "projects, although the schedule outcome remains "
                "uncertain."
            )

        else:

            sentences.append(
                "The projected delay should therefore be validated "
                "against the project's actual completion and closure status."
            )

        return " ".join(sentences)

    # -----------------------------------------------------
    # SPECIAL CASE: LOW PROGRESS + HIGH DELAY
    # -----------------------------------------------------

    if (
        physical_progress < 50
        and predicted_delay_days > 730
    ):

        sentences.append(
            f"Physical progress is currently "
            f"{physical_progress:.1f}%, while the model projects "
            f"approximately {predicted_delay_days:.0f} days of delay."
        )

        sentences.append(
            "The combination indicates a significant execution "
            "pressure that may affect the planned completion timeline."
        )

        return " ".join(sentences)

    # -----------------------------------------------------
    # SPECIAL CASE: COST + TIME PRESSURE
    # -----------------------------------------------------

    if (
        predicted_delay_days > 730
        and predicted_cost_overrun_percent > 25
    ):

        sentences.append(
            f"The project faces combined schedule and financial "
            f"pressure, with an estimated delay of "
            f"{predicted_delay_days:.0f} days and projected cost "
            f"growth of {predicted_cost_overrun_percent:.1f}%."
        )

        sentences.append(
            "This combination increases the likelihood of further "
            "execution and budget pressure if corrective action is delayed."
        )

        return " ".join(sentences)

    # -----------------------------------------------------
    # OVERALL RISK
    # -----------------------------------------------------

    if risk_level == "CRITICAL":

        sentences.append(
            f"{project_name} is showing multiple severe risk "
            "indicators that could materially affect completion."
        )

    elif risk_level == "HIGH":

        sentences.append(
            f"{project_name} is exposed to significant project "
            "risk based on the current execution outlook."
        )

    elif risk_level == "MEDIUM":

        sentences.append(
            f"{project_name} shows moderate risk, with specific "
            "areas requiring closer monitoring."
        )

    else:

        sentences.append(
            f"{project_name} currently shows limited major risk "
            "indicators from the available project data."
        )

    # -----------------------------------------------------
    # PRIMARY TIME SIGNAL
    # -----------------------------------------------------

    if predicted_delay_days > 1095:

        sentences.append(
            f"The predicted schedule extension of approximately "
            f"{predicted_delay_days:.0f} days represents a prolonged "
            "departure from the original completion plan."
        )

    elif predicted_delay_days > 730:

        sentences.append(
            f"The current schedule outlook indicates roughly "
            f"{predicted_delay_days:.0f} days of delay, creating "
            "substantial completion pressure."
        )

    elif predicted_delay_days > 365:

        sentences.append(
            f"The projected delay of around "
            f"{predicted_delay_days:.0f} days indicates meaningful "
            "schedule slippage."
        )

    elif predicted_delay_days > 90:

        sentences.append(
            f"A delay of approximately "
            f"{predicted_delay_days:.0f} days is currently expected."
        )

    # -----------------------------------------------------
    # PRIMARY COST SIGNAL
    # -----------------------------------------------------

    if predicted_cost_overrun_percent > 50:

        sentences.append(
            f"The projected cost is approximately "
            f"{predicted_cost_overrun_percent:.1f}% above the "
            "original budget, creating severe financial exposure."
        )

    elif predicted_cost_overrun_percent > 25:

        sentences.append(
            f"The cost outlook indicates an estimated "
            f"{predicted_cost_overrun_percent:.1f}% increase over "
            "the original project cost."
        )

    elif predicted_cost_overrun_percent > 10:

        sentences.append(
            f"Projected expenditure is currently around "
            f"{predicted_cost_overrun_percent:.1f}% above the "
            "original cost baseline."
        )

    # -----------------------------------------------------
    # PROGRESS SIGNAL
    # -----------------------------------------------------

    if physical_progress < 25:

        sentences.append(
            f"Physical progress is only "
            f"{physical_progress:.1f}%, indicating substantial "
            "execution pressure."
        )

    elif physical_progress < 50:

        sentences.append(
            f"Physical progress stands at "
            f"{physical_progress:.1f}%, requiring closer execution "
            "monitoring."
        )

    elif physical_progress < 75:

        sentences.append(
            f"Physical progress is approximately "
            f"{physical_progress:.1f}%, which remains below a "
            "strong execution position."
        )

    # -----------------------------------------------------
    # HISTORICAL EVIDENCE
    # -----------------------------------------------------

    if high_similarity_count >= 5:

        sentences.append(
            f"{high_similarity_count} highly similar completed "
            "projects provide strong historical support for the assessment."
        )

    elif high_similarity_count >= 3:

        sentences.append(
            f"{high_similarity_count} comparable completed projects "
            "provide useful historical evidence."
        )

    elif historical_count == 0:

        sentences.append(
            "There is currently insufficient completed-project "
            "evidence for strong historical validation."
        )

    # -----------------------------------------------------
    # UNCERTAINTY
    # -----------------------------------------------------

    if cost_range_width_percent > 100:

        sentences.append(
            "The financial outlook also has substantial uncertainty "
            "because the expected cost range is unusually broad."
        )

    elif cost_range_width_percent > 50:

        sentences.append(
            "Cost uncertainty remains elevated because the expected "
            "financial outcomes span a wide range."
        )

    if time_range_width > 1800:

        sentences.append(
            "The schedule estimate has particularly high uncertainty "
            "because the possible delay outcomes vary substantially."
        )

    elif time_range_width > 730:

        sentences.append(
            "The broad schedule range means that the projected delay "
            "should be validated against actual milestone performance."
        )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if (
        cost_prediction_confidence == "LOW"
        and time_prediction_confidence == "LOW"
    ):

        sentences.append(
            "Both predictions have low confidence, so the assessment "
            "should be treated as an early-warning indicator."
        )

    elif time_prediction_confidence == "LOW":

        sentences.append(
            "The schedule prediction has limited confidence and "
            "requires validation against current milestones."
        )

    elif cost_prediction_confidence == "LOW":

        sentences.append(
            "The cost prediction has limited confidence and "
            "should be validated against current expenditure."
        )

    # Maximum three sentences for production readability.
    return " ".join(sentences[:3])


# =========================================================
# DYNAMIC RECOMMENDED SOLUTION
# =========================================================

def _build_dynamic_solution(
    project,
    issues,
    predicted_delay_days,
    predicted_cost_overrun_percent,
    physical_progress,
    historical_count,
    high_similarity_count,
    cost_prediction_confidence,
    time_prediction_confidence,
    cost_range_width_percent,
    time_range_width,
):
    """
    Creates concise and condition-specific corrective actions.
    """

    solutions = []

    # =====================================================
    # COMPLETED / NEAR-COMPLETED BUT DELAYED
    # =====================================================

    if (
        physical_progress >= 90
        and predicted_delay_days > 730
    ):

        solutions.append(
            "Verify the actual completion, closure and pending "
            "administrative milestones against the original schedule."
        )

        solutions.append(
            "Investigate the source of the projected delay and "
            "validate the AI schedule estimate using the latest "
            "project records."
        )

    # =====================================================
    # LOW PROGRESS + HIGH DELAY
    # =====================================================

    elif (
        physical_progress < 50
        and predicted_delay_days > 730
    ):

        solutions.append(
            "Identify the activities responsible for slow physical "
            "progress and establish short-term recovery milestones."
        )

        solutions.append(
            "Review critical-path activities regularly and escalate "
            "execution bottlenecks that are driving the schedule delay."
        )

    # =====================================================
    # HIGH TIME DELAY
    # =====================================================

    elif predicted_delay_days > 1095:

        solutions.append(
            "Prepare a formal schedule recovery plan focused on "
            "critical activities and measurable completion milestones."
        )

    elif predicted_delay_days > 730:

        solutions.append(
            "Implement a focused time-recovery plan and closely "
            "track critical-path activities."
        )

    elif predicted_delay_days > 365:

        solutions.append(
            "Strengthen schedule monitoring and identify the main "
            "activities contributing to the projected delay."
        )

    elif predicted_delay_days > 90:

        solutions.append(
            "Increase milestone-level schedule monitoring and "
            "address emerging sources of delay early."
        )

    # =====================================================
    # PHYSICAL PROGRESS
    # =====================================================

    if physical_progress < 25:

        solutions.append(
            "Conduct an immediate execution review and identify "
            "the operational constraints affecting physical progress."
        )

    elif physical_progress < 50:

        solutions.append(
            "Review work-package progress and prioritize activities "
            "required to improve execution performance."
        )

    elif physical_progress < 75:

        solutions.append(
            "Strengthen progress tracking and focus management "
            "attention on below-target activities."
        )

    # =====================================================
    # COST PRESSURE
    # =====================================================

    if predicted_cost_overrun_percent > 50:

        solutions.append(
            "Perform an immediate financial review and identify "
            "the major drivers of projected cost escalation."
        )

    elif predicted_cost_overrun_percent > 25:

        solutions.append(
            "Review major cost-escalation drivers and strengthen "
            "budget control over remaining activities."
        )

    elif predicted_cost_overrun_percent > 10:

        solutions.append(
            "Closely monitor remaining expenditure and investigate "
            "the causes of the projected cost increase."
        )

    elif predicted_cost_overrun_percent > 5:

        solutions.append(
            "Maintain tighter cost monitoring and verify whether "
            "the current escalation trend is continuing."
        )

    # =====================================================
    # COST UNCERTAINTY
    # =====================================================

    if cost_range_width_percent > 100:

        solutions.append(
            "Carry out a detailed cost-sensitivity review because "
            "the possible financial outcomes span a very wide range."
        )

    elif cost_range_width_percent > 50:

        solutions.append(
            "Review the financial assumptions and monitor expenditure "
            "closely because cost uncertainty remains elevated."
        )

    # =====================================================
    # TIME UNCERTAINTY
    # =====================================================

    if time_range_width > 1800:

        solutions.append(
            "Perform schedule scenario analysis and establish "
            "contingency milestones because delay outcomes are highly uncertain."
        )

    elif time_range_width > 730:

        solutions.append(
            "Increase schedule review frequency and validate the "
            "projected delay against actual milestone performance."
        )

    # =====================================================
    # LOW CONFIDENCE
    # =====================================================

    if (
        cost_prediction_confidence == "LOW"
        and time_prediction_confidence == "LOW"
    ):

        solutions.append(
            "Validate both predictions against updated project "
            "performance before taking major corrective decisions."
        )

    elif cost_prediction_confidence == "LOW":

        solutions.append(
            "Validate the cost outlook against the latest expenditure "
            "and financial progress."
        )

    elif time_prediction_confidence == "LOW":

        solutions.append(
            "Validate the schedule outlook against current milestone "
            "performance and updated execution information."
        )

    # =====================================================
    # HISTORICAL EVIDENCE
    # =====================================================

    if high_similarity_count >= 3:

        solutions.append(
            "Benchmark the project against comparable completed "
            "projects and use relevant historical outcomes to guide "
            "corrective actions."
        )

    elif historical_count == 0:

        solutions.append(
            "Prioritize current project performance data because "
            "comparable completed-project evidence is limited."
        )

    # =====================================================
    # COMBINED COST + TIME
    # =====================================================

    if (
        predicted_delay_days > 730
        and predicted_cost_overrun_percent > 25
    ):

        solutions.append(
            "Conduct an integrated cost-and-schedule review and "
            "prioritize corrective actions across both areas."
        )

    # =====================================================
    # DEFAULT
    # =====================================================

    if not solutions:

        solutions.append(
            "Continue regular monitoring of cost, schedule and "
            "physical progress and reassess risk as new information "
            "becomes available."
        )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    solutions = list(
        dict.fromkeys(solutions)
    )

    # Maximum three concise recommendations.
    return " ".join(
        solutions[:3]
    )