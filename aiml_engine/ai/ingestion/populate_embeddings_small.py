from aiml_engine.ai.database import get_connection
from sentence_transformers import SentenceTransformer
import torch


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_project_text(row):
    return f"""
Project ID: {row[0]}
Project Name: {row[1]}
Project Code: {row[2]}
Agency: {row[3]}
Ministry: {row[4]}
Sector: {row[5]}
State: {row[6]}
PM GID: {row[7]}
Legacy OCMS Code: {row[8]}
Start Date: {row[9]}
Original Completion Date: {row[10]}
Revised Completion Date: {row[11]}
Original Cost: {row[12]}
Revised Cost: {row[13]}
Cumulative Expenditure: {row[14]}
Physical Progress: {row[15]}
Progress Status: {row[16]}
""".strip()


def main():
    print("Loading BGE-small model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded.")

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    project_id,
                    project_name,
                    project_code,
                    agency,
                    ministry,
                    sector,
                    state,
                    pmgid,
                    legacy_ocms_code,
                    start_date,
                    original_completion_date,
                    revised_completion_date,
                    original_cost,
                    revised_cost,
                    cumulative_expenditure,
                    physical_progress,
                    progress_status
                FROM project
                ORDER BY project_id;
            """)

            rows = cur.fetchall()

            print(f"Projects found: {len(rows)}")

            for index, row in enumerate(rows, start=1):

                project_id = row[0]
                content = build_project_text(row)

                with torch.inference_mode():
                    embedding = model.encode(
                        content,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                vector_string = "[" + ",".join(
                    str(float(value))
                    for value in embedding
                ) + "]"

                cur.execute(
                    """
                    INSERT INTO documents_small
                        (project_id, content, embedding)
                    VALUES
                        (%s, %s, %s::extensions.vector(384))
                    ON CONFLICT DO NOTHING;
                    """,
                    (
                        project_id,
                        content,
                        vector_string
                    )
                )

                if index % 50 == 0 or index == len(rows):
                    conn.commit()
                    print(f"Processed: {index}/{len(rows)}")

    print("Embedding population completed.")


if __name__ == "__main__":
    main()
