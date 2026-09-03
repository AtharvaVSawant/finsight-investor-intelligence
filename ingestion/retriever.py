import os

import psycopg

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set in the .env file."
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading BGE model...")

embeddings = HuggingFaceEmbeddings(

    model_name=MODEL_NAME,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }

)

print("✓ BGE model loaded")


# ============================================================
# RETRIEVER FUNCTION
# ============================================================

def retrieve_documents(

    query: str,

    top_k: int = 5,

    similarity_threshold: float = 0.60,

    company: str | None = None,

    year: int | None = None,

    document_type: str | None = None,

):

    """
    Retrieve semantically relevant document chunks
    from PostgreSQL + pgvector.
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty"
        )

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0"
        )

    if not 0 <= similarity_threshold <= 1:

        raise ValueError(
            "similarity_threshold "
            "must be between 0 and 1"
        )

    # ========================================================
    # CREATE QUERY EMBEDDING
    # ========================================================

    print(
        "\n✓ Generating query embedding..."
    )

    query_vector = embeddings.embed_query(
        query
    )

    print(
        f"✓ Embedding dimensions: "
        f"{len(query_vector)}"
    )

    # ========================================================
    # BUILD SQL
    # ========================================================

    sql = """

        SELECT

            id,

            content,

            source,

            company,

            document_type,

            year,

            page,

            chunk_index,

            embedding <=> %s::vector
                AS distance

        FROM document_chunks

        WHERE embedding IS NOT NULL

    """

    parameters = []

    # --------------------------------------------------------
    # First query vector
    # Used in SELECT
    # --------------------------------------------------------

    parameters.append(
        query_vector
    )

    # ========================================================
    # OPTIONAL FILTERS
    # ========================================================

    if company is not None:

        sql += """

            AND company = %s

        """

        parameters.append(
            company
        )

    if year is not None:

        sql += """

            AND year = %s

        """

        parameters.append(
            year
        )

    if document_type is not None:

        sql += """

            AND document_type = %s

        """

        parameters.append(
            document_type
        )

    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    candidate_k = max(
        top_k * 4,
        20
    )

    sql += """

        ORDER BY embedding <=> %s::vector

        LIMIT %s

    """

    # --------------------------------------------------------
    # Query vector for ORDER BY
    # --------------------------------------------------------

    parameters.append(
        query_vector
    )

    # --------------------------------------------------------
    # Candidate count
    # --------------------------------------------------------

    parameters.append(
        candidate_k
    )

    # ========================================================
    # CONNECT TO SUPABASE
    # ========================================================

    print(
        "\nConnecting to PostgreSQL..."
    )

    connection = psycopg.connect(
        DATABASE_URL
    )

    print(
        "✓ Connected"
    )

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                sql,
                parameters
            )

            rows = cursor.fetchall()

    finally:

        connection.close()

    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    results = []

    for row in rows:

        (

            chunk_id,

            content,

            source,

            company_name,

            doc_type,

            doc_year,

            page,

            chunk_index,

            distance,

        ) = row

        # ----------------------------------------------------
        # pgvector cosine distance
        #
        # similarity = 1 - distance
        # ----------------------------------------------------

        similarity = (
            1 - float(distance)
        )

        # ----------------------------------------------------
        # Similarity threshold
        # ----------------------------------------------------

        if similarity < similarity_threshold:

            continue

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(

            {

                "id": chunk_id,

                "content": content,

                "source": source,

                "company": company_name,

                "document_type": doc_type,

                "year": doc_year,

                "page": page,

                "chunk_index": chunk_index,

                "distance": float(distance),

                "similarity": similarity,

            }

        )

    # ========================================================
    # TOP K
    # ========================================================

    results = results[:top_k]

    return results


# ============================================================
# TEST RETRIEVER
# ============================================================

if __name__ == "__main__":

    query = (
        "What products and services "
        "does Apple offer?"
    )

    print("\n" + "=" * 80)

    print(
        "SEMANTIC RETRIEVAL TEST"
    )

    print("=" * 80)

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------------
    # Run retrieval
    # --------------------------------------------------------

    results = retrieve_documents(

        query=query,

        top_k=5,

        similarity_threshold=0.60,

        company="Apple",

        year=2024,

    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 80)

    print(
        f"TOP {len(results)} RESULTS"
    )

    print("=" * 80)

    if not results:

        print(
            "\n⚠ No documents matched "
            "the query."
        )

    else:

        for rank, result in enumerate(

            results,

            start=1

        ):

            print(
                "\n" + "-" * 80
            )

            print(
                f"Rank       : {rank}"
            )

            print(
                f"Chunk ID   : "
                f"{result['id']}"
            )

            print(
                f"Company    : "
                f"{result['company']}"
            )

            print(
                f"Year       : "
                f"{result['year']}"
            )

            print(
                f"Document   : "
                f"{result['document_type']}"
            )

            print(
                f"Page       : "
                f"{result['page']}"
            )

            print(
                f"Chunk      : "
                f"{result['chunk_index']}"
            )

            print(
                f"Distance   : "
                f"{result['distance']:.4f}"
            )

            print(
                f"Similarity : "
                f"{result['similarity']:.4f}"
            )

            print(
                "\nSource:"
            )

            print(
                result["source"]
            )

            print(
                "\nContent:"
            )

            print(
                result["content"][:1000]
            )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "RETRIEVAL COMPLETE"
    )

    print("=" * 80)