from aiml_engine.ai.database import get_connection
from aiml_engine.ai.embedding_service import generate_embedding


def _embedding_to_vector_string(embedding):
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _run_vector_search(function_name, embedding, limit, exclude_project_id=None):
    vector_string = _embedding_to_vector_string(embedding)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT project_id, content, similarity
                FROM {function_name}(
                    %s::extensions.vector(384),
                    %s,
                    %s
                );
                """,
                (
                    vector_string,
                    limit,
                    exclude_project_id,
                ),
            )

            rows = cur.fetchall()

    return [
        {
            "project_id": row[0],
            "content": row[1],
            "similarity": float(row[2]),
        }
        for row in rows
    ]


def search_similar_projects(
    text,
    limit=10,
    exclude_project_id=None,
):
    embedding = generate_embedding(text)

    return search_similar_projects_by_embedding(
        embedding,
        limit=limit,
        exclude_project_id=exclude_project_id,
    )


def search_completed_similar_projects(
    text,
    limit=50,
    exclude_project_id=None,
):
    embedding = generate_embedding(text)

    return search_completed_similar_projects_by_embedding(
        embedding,
        limit=limit,
        exclude_project_id=exclude_project_id,
    )


def search_similar_projects_by_embedding(
    embedding,
    limit=10,
    exclude_project_id=None,
):
    return _run_vector_search(
        "match_documents_small",
        embedding,
        limit,
        exclude_project_id,
    )


def search_completed_similar_projects_by_embedding(
    embedding,
    limit=50,
    exclude_project_id=None,
):
    return _run_vector_search(
        "match_completed_documents_small",
        embedding,
        limit,
        exclude_project_id,
    )