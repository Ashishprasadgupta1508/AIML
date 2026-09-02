import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATH / ENV
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    return psycopg.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )


# --------------------------------------------------
# PROJECT -> TEXT
# --------------------------------------------------

def project_to_text(project):
    return f"""
Project Name: {project["project_name"] or ""}
Project Code: {project["project_code"] or ""}
Project ID: {project["project_id"]}

Agency: {project["agency"] or ""}
Ministry: {project["ministry"] or ""}
Sector: {project["sector"] or ""}
State: {project["state"] or ""}

PMGID: {project["pmgid"] or ""}
Legacy OCMS Code: {project["legacy_ocms_code"] or ""}

Start Date: {project["start_date"] or ""}
Original Completion Date: {project["original_completion_date"] or ""}
Revised Completion Date: {project["revised_completion_date"] or ""}
Actual Completion Date: {project["actual_completion_date"] or ""}

Original Cost: {project["original_cost"] or ""}
Revised Cost: {project["revised_cost"] or ""}
Cumulative Expenditure: {project["cumulative_expenditure"] or ""}

Physical Progress: {project["physical_progress"] or ""}
Progress Status: {project["progress_status"] or ""}
""".strip()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading BGE-M3 model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Supabase PostgreSQL...")

    with get_connection() as conn:

        # ------------------------------------------
        # READ PROJECT DATA
        # ------------------------------------------

        with conn.cursor() as cur:

            cur.execute("""
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
                ORDER BY project_id;
            """)

            rows = cur.fetchall()

            columns = [desc.name for desc in cur.description]

        projects = [
            dict(zip(columns, row))
            for row in rows
        ]

        print(f"Projects found: {len(projects)}")

        # ------------------------------------------
        # CHECK ALREADY EMBEDDED PROJECTS
        # ------------------------------------------

        with conn.cursor() as cur:
            cur.execute("""
        SELECT project_id
        FROM documents
        WHERE project_id IS NOT NULL;
    """)
            existing_ids = {
                row[0]
                for row in cur.fetchall()
            }

        print(f"Already embedded: {len(existing_ids)}")

        projects_to_process = [
            project
            for project in projects
            if project["project_id"] not in existing_ids
        ]

        print(f"Remaining projects: {len(projects_to_process)}")

        if not projects_to_process:
            print("Nothing to embed.")
            return

        # ------------------------------------------
        # BATCH PROCESSING
        # ------------------------------------------

        total = len(projects_to_process)

        for start in range(0, total, BATCH_SIZE):

            batch = projects_to_process[
                start:start + BATCH_SIZE
            ]

            texts = [
                project_to_text(project)
                for project in batch
            ]

            print(
                f"Embedding projects "
                f"{start + 1} - {start + len(batch)} "
                f"of {total}"
            )

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # ------------------------------------------
            # INSERT INTO DOCUMENTS
            # ------------------------------------------

            with conn.cursor() as cur:

                for project, text, embedding in zip(
                    batch,
                    texts,
                    embeddings
                ):

                    vector_string = "[" + ",".join(
                        str(float(value))
                        for value in embedding
                    ) + "]"

                    cur.execute(
                        """
                        INSERT INTO documents
                        (
                            project_id,
                            content,
                            embedding
                        )
                        VALUES
                        (
                            %s,
                            %s,
                            %s::vector
                        );
                        """,
                        (
                            project["project_id"],
                            text,
                            vector_string,
                        )
                    )

            conn.commit()
if __name__ == "__main__":
    main()