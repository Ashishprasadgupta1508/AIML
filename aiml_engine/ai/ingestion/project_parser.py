import re


DATE_PATTERN = re.compile(r"^\d{2}/\d{4}$")
PAREN_DATE_PATTERN = re.compile(r"^\(\d{2}/\d{4}\)$")
NUMBER_PATTERN = re.compile(r"^[\d,]+(?:\.\d+)?$")
PAREN_NUMBER_PATTERN = re.compile(r"^\([\d,]+(?:\.\d+)?\)$")
PROJECT_ID_PATTERN = re.compile(r"^\((\d{6})\)$")


def clean_lines(text):
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def is_serial_number(line):
    return bool(re.fullmatch(r"\d{1,4}", line))


def looks_like_project_start(lines, index):

    if index + 1 >= len(lines):
        return False

    if not is_serial_number(lines[index]):
        return False

    next_line = lines[index + 1]

    invalid = {
        "Project Name",
        "PROJECT NAME",
        "Sector - Overview",
        "Ministry - Overview",
        "Agency - Overview",
    }

    if next_line in invalid:
        return False

    if NUMBER_PATTERN.fullmatch(next_line):
        return False

    return True


def normalize_parenthesized(value):
    """
    Convert:
        (01/2024) -> 01/2024
        (824.28)  -> 824.28
        (-)       -> None
    """

    value = value.strip()

    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()

    if value in {"-", ""}:
        return None

    return value


def parse_project_dates(record):

    dates = []

    for line in record:

        value = line.strip()

        if DATE_PATTERN.fullmatch(value):
            dates.append(value)

        elif PAREN_DATE_PATTERN.fullmatch(value):
            dates.append(
                normalize_parenthesized(value)
            )

    return {
        "approval_date": dates[0] if len(dates) > 0 else None,
        "start_date": dates[1] if len(dates) > 1 else None,
        "original_doc": dates[2] if len(dates) > 2 else None,
        "revised_doc": dates[3] if len(dates) > 3 else None,
    }


def parse_project_costs(record):

    values = []

    for line in record:

        value = line.strip()

        # Ignore dates
        if DATE_PATTERN.fullmatch(value):
            continue

        if PAREN_DATE_PATTERN.fullmatch(value):
            continue

        # Ignore project ID
        if PROJECT_ID_PATTERN.fullmatch(value):
            continue

        # Ignore serial number
        if is_serial_number(value):
            continue

        # Normal number
        if NUMBER_PATTERN.fullmatch(value):
            values.append(float(value.replace(",", "")))
            continue

        # Parenthesized number
        if PAREN_NUMBER_PATTERN.fullmatch(value):

            inner = value[1:-1].strip()

            if NUMBER_PATTERN.fullmatch(inner):
                values.append(float(inner.replace(",", "")))

    # Last four numeric values normally correspond to:
    #
    # Original Cost
    # Revised Cost
    # Expenditure
    # Physical Progress

    if len(values) >= 4:

        return {
            "original_cost": values[-4],
            "revised_cost": values[-3],
            "expenditure": values[-2],
            "physical_progress": values[-1],
        }

    return {
        "original_cost": None,
        "revised_cost": None,
        "expenditure": None,
        "physical_progress": None,
    }


def parse_projects(pages):

    projects = []

    for page in pages:

        text = page["text"]

        if "All Ongoing Projects" not in text:
            continue

        lines = clean_lines(text)

        starts = []

        for i in range(len(lines)):

            if looks_like_project_start(lines, i):
                starts.append(i)

        for position, start in enumerate(starts):

            if position + 1 < len(starts):
                end = starts[position + 1]
            else:
                end = len(lines)

            record = lines[start:end]

            if len(record) < 4:
                continue

            serial_number = record[0]

            # Find project ID
            project_id = None
            project_id_index = None

            for i, line in enumerate(record[1:], start=1):

                match = PROJECT_ID_PATTERN.fullmatch(line)

                if match:
                    project_id = match.group(1)
                    project_id_index = i
                    break

            if project_id is None:
                continue

            # Agency is immediately before Project ID
            agency = None

            if project_id_index > 1:
                agency = record[project_id_index - 1]

                if (
                    agency.startswith("(")
                    and agency.endswith(")")
                ):
                    agency = agency[1:-1].strip()

            # Project name is everything between
            # serial number and agency
            project_name_lines = record[
                1:project_id_index - 1
            ]

            project_name = " ".join(
                project_name_lines
            ).strip()

            # Find state
            state = None

            for i in range(
                project_id_index + 1,
                len(record)
            ):

                line = record[i]

                if line in {"(-) (-)", "(-)", "-"}:
                    continue

                if DATE_PATTERN.fullmatch(line):
                    break

                if PAREN_DATE_PATTERN.fullmatch(line):
                    continue

                state = line
                break

            dates = parse_project_dates(record)
            costs = parse_project_costs(record)

            projects.append({

                "serial_number": serial_number,

                "project_id": project_id,

                "project_name": project_name,

                "agency": agency,

                "state": state,

                "approval_date": dates[
                    "approval_date"
                ],

                "start_date": dates[
                    "start_date"
                ],

                "original_doc": dates[
                    "original_doc"
                ],

                "revised_doc": dates[
                    "revised_doc"
                ],

                "original_cost": costs[
                    "original_cost"
                ],

                "revised_cost": costs[
                    "revised_cost"
                ],

                "expenditure": costs[
                    "expenditure"
                ],

                "physical_progress": costs[
                    "physical_progress"
                ],

                "page": page["page"],

                "raw_content": "\n".join(record),
            })

    return projects