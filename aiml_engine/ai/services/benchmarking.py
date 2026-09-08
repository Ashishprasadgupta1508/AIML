import re
from datetime import datetime
from statistics import mean


def _to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_value(content, label):
    if not content:
        return None

    pattern = rf"{re.escape(label)}:\s*([^\n\r]+)"
    match = re.search(
        pattern,
        str(content),
        re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    if value.lower() in {"none", "null", ""}:
        return None

    return value


def _parse_date(content, label):
    value = _parse_value(content, label)

    if not value:
        return None

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    return None


def _extract_metrics(project):
    if not isinstance(project, dict):
        return {}

    content = project.get("content")

    original_cost = _to_float(
        project.get("original_cost")
    )

    if original_cost is None:
        original_cost = _to_float(
            _parse_value(content, "Original Cost")
        )

    revised_cost = _to_float(
        project.get("revised_cost")
    )

    if revised_cost is None:
        revised_cost = _to_float(
            _parse_value(content, "Revised Cost")
        )

    physical_progress = _to_float(
        project.get("physical_progress")
    )

    if physical_progress is None:
        physical_progress = _to_float(
            _parse_value(content, "Physical Progress")
        )

    start_date = _parse_date(
        content,
        "Start Date"
    )

    original_completion_date = _parse_date(
        content,
        "Original Completion Date"
    )

    revised_completion_date = _parse_date(
        content,
        "Revised Completion Date"
    )

    planned_duration = None
    actual_duration = None
    delay_days = None

    if start_date and original_completion_date:
        planned_duration = (
            original_completion_date - start_date
        ).days

    if start_date and revised_completion_date:
        actual_duration = (
            revised_completion_date - start_date
        ).days

    if (
        planned_duration is not None
        and actual_duration is not None
    ):
        delay_days = (
            actual_duration - planned_duration
        )

    cost_overrun = None

    if (
        original_cost is not None
        and revised_cost is not None
        and original_cost != 0
    ):
        cost_overrun = (
            (revised_cost - original_cost)
            / original_cost
        ) * 100

    return {
        "original_cost": original_cost,
        "revised_cost": revised_cost,
        "physical_progress": physical_progress,
        "planned_duration_days": planned_duration,
        "actual_duration_days": actual_duration,
        "delay_days": delay_days,
        "cost_overrun_percent": cost_overrun,
    }


def _average(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(mean(values), 2)


def _difference(current, benchmark):
    if current is None or benchmark in (None, 0):
        return None

    return round(
        ((current - benchmark) / abs(benchmark)) * 100,
        2,
    )


def _status(
    current,
    benchmark,
    lower_is_better=True,
):
    difference = _difference(
        current,
        benchmark,
    )

    if difference is None:
        return "INSUFFICIENT_DATA"

    if abs(difference) <= 5:
        return "IN_LINE"

    if lower_is_better:
        return (
            "BETTER_THAN_BENCHMARK"
            if difference < 0
            else "WORSE_THAN_BENCHMARK"
        )

    return (
        "BETTER_THAN_BENCHMARK"
        if difference > 0
        else "WORSE_THAN_BENCHMARK"
    )


def generate_benchmarking(
    project,
    completed_similar_projects=None,
):
    completed_similar_projects = (
        completed_similar_projects or []
    )

    current = _extract_metrics(project)

    historical = []

    for item in completed_similar_projects:
        metrics = _extract_metrics(item)

        if metrics:
            historical.append(metrics)

    if not historical:
        return {
            "benchmarking_available": False,
            "historical_projects_used": 0,
            "average_similarity": None,
            "overall_status": "INSUFFICIENT_HISTORICAL_DATA",
            "summary": (
                "No completed comparable historical "
                "projects were available."
            ),
        }

    avg_original_cost = _average([
        x["original_cost"]
        for x in historical
    ])

    avg_revised_cost = _average([
        x["revised_cost"]
        for x in historical
    ])

    avg_cost_overrun = _average([
        x["cost_overrun_percent"]
        for x in historical
    ])

    avg_planned_duration = _average([
        x["planned_duration_days"]
        for x in historical
    ])

    avg_actual_duration = _average([
        x["actual_duration_days"]
        for x in historical
    ])

    avg_delay = _average([
        x["delay_days"]
        for x in historical
    ])

    avg_progress = _average([
        x["physical_progress"]
        for x in historical
    ])

    similarities = [
        _to_float(item.get("similarity"))
        for item in completed_similar_projects
    ]

    avg_similarity = _average(similarities)

    comparisons = []

    for current_value, benchmark, lower in [
        (
            current.get("original_cost"),
            avg_original_cost,
            True,
        ),
        (
            current.get("cost_overrun_percent"),
            avg_cost_overrun,
            True,
        ),
        (
            current.get("planned_duration_days"),
            avg_planned_duration,
            True,
        ),
        (
            current.get("delay_days"),
            avg_delay,
            True,
        ),
        (
            current.get("physical_progress"),
            avg_progress,
            False,
        ),
    ]:
        status = _status(
            current_value,
            benchmark,
            lower_is_better=lower,
        )

        if status != "INSUFFICIENT_DATA":
            comparisons.append(status)

    worse = comparisons.count(
        "WORSE_THAN_BENCHMARK"
    )

    better = comparisons.count(
        "BETTER_THAN_BENCHMARK"
    )

    if worse > better:
        overall_status = (
            "ABOVE_HISTORICAL_BENCHMARK"
        )
    elif better > worse:
        overall_status = (
            "BELOW_HISTORICAL_BENCHMARK"
        )
    else:
        overall_status = "GENERALLY_IN_LINE"

    return {
        "benchmarking_available": True,

        "historical_projects_found": len(
            completed_similar_projects
        ),

        "historical_projects_used": len(
            historical
        ),

        "average_similarity": avg_similarity,

        "overall_status": overall_status,

        "cost": {
            "current_original_cost":
                current.get("original_cost"),

            "historical_average_original_cost":
                avg_original_cost,

            "difference_percent":
                _difference(
                    current.get("original_cost"),
                    avg_original_cost,
                ),

            "status":
                _status(
                    current.get("original_cost"),
                    avg_original_cost,
                    lower_is_better=True,
                ),
        },

        "cost_overrun": {
            "current_cost_overrun_percent":
                current.get(
                    "cost_overrun_percent"
                ),

            "historical_average_cost_overrun_percent":
                avg_cost_overrun,

            "difference_percent":
                _difference(
                    current.get(
                        "cost_overrun_percent"
                    ),
                    avg_cost_overrun,
                ),

            "status":
                _status(
                    current.get(
                        "cost_overrun_percent"
                    ),
                    avg_cost_overrun,
                    lower_is_better=True,
                ),
        },

        "schedule": {
            "current_planned_duration_days":
                current.get(
                    "planned_duration_days"
                ),

            "historical_average_planned_duration_days":
                avg_planned_duration,

            "difference_percent":
                _difference(
                    current.get(
                        "planned_duration_days"
                    ),
                    avg_planned_duration,
                ),

            "status":
                _status(
                    current.get(
                        "planned_duration_days"
                    ),
                    avg_planned_duration,
                    lower_is_better=True,
                ),
        },

        "delay": {
            "current_delay_days":
                current.get("delay_days"),

            "historical_average_delay_days":
                avg_delay,

            "difference_percent":
                _difference(
                    current.get("delay_days"),
                    avg_delay,
                ),

            "status":
                _status(
                    current.get("delay_days"),
                    avg_delay,
                    lower_is_better=True,
                ),
        },

        "physical_progress": {
            "current_physical_progress_percent":
                current.get(
                    "physical_progress"
                ),

            "historical_average_physical_progress_percent":
                avg_progress,

            "difference_percent":
                _difference(
                    current.get(
                        "physical_progress"
                    ),
                    avg_progress,
                ),

            "status":
                _status(
                    current.get(
                        "physical_progress"
                    ),
                    avg_progress,
                    lower_is_better=False,
                ),
        },

        "summary": (
            f"Benchmarking is based on "
            f"{len(historical)} completed "
            f"comparable historical project(s)."
        ),
    }