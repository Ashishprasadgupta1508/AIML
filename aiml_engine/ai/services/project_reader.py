from aiml_engine.ai.database import get_connection


def get_project_by_id(project_id):
    """
    Read a single project from Supabase.

    IMPORTANT:
    This function performs SELECT only.
    It does not INSERT, UPDATE or DELETE anything.
    """

    try:
        project_id = int(project_id)
    except (TypeError, ValueError):
        raise ValueError("project_id must be a valid integer.")

    query = """
        SELECT
            project_id,
            agency,
            created_at,
            cumulative_expenditure,
            end_date,
            latitude,
            legacy_ocms_code,
            longitude,
            ministry,
            original_completion_date,
            original_cost,
            physical_progress,
            pmgid,
            progress_status,
            project_code,
            project_name,
            radius,
            revised_completion_date,
            revised_cost,
            sector,
            start_date,
            updated_at,
            assigned_supervisor,
            state,
            actual_completion_date
        FROM project
        WHERE project_id = %s
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (project_id,))
            row = cur.fetchone()

    if row is None:
        return None

    columns = [
        "project_id",
        "agency",
        "created_at",
        "cumulative_expenditure",
        "end_date",
        "latitude",
        "legacy_ocms_code",
        "longitude",
        "ministry",
        "original_completion_date",
        "original_cost",
        "physical_progress",
        "pmgid",
        "progress_status",
        "project_code",
        "project_name",
        "radius",
        "revised_completion_date",
        "revised_cost",
        "sector",
        "start_date",
        "updated_at",
        "assigned_supervisor",
        "state",
        "actual_completion_date",
    ]

    return dict(zip(columns, row))