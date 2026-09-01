import psycopg
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}

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
    Retrieve the most semantically relevant document chunks.

    Args:
        query:
            User's natural-language question.

        top_k:
            Maximum number of results to return.

        similarity_threshold:
            Minimum cosine similarity required.

        company:
            Optional company filter.

        year:
            Optional year filter.

        document_type:
            Optional document type filter.

    Returns:
        List of dictionaries containing retrieved chunks.
    """

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    # ========================================================
    # CREATE QUERY EMBEDDING
    # ========================================================

    print("\n✓ Generating query embedding...")

    query_vector = embeddings.embed_query(query)

    print(
        f"✓ Embedding dimensions: {len(query_vector)}"
    )

    # ========================================================
    # BUILD SQL QUERY
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

            embedding <=> %s::vector AS distance

        FROM document_chunks

        WHERE embedding IS NOT NULL
    """

    # --------------------------------------------------------
    # SQL PARAMETERS
    # --------------------------------------------------------

    parameters = []

    # IMPORTANT:
    # The first %s in the SELECT statement is the query vector.
    parameters.append(query_vector)

    # ========================================================
    # METADATA FILTERS
    # ========================================================

    if company is not None:
        sql += """
            AND company = %s
        """
        parameters.append(company)

    if year is not None:
        sql += """
            AND year = %s
        """
        parameters.append(year)

    if document_type is not None:
        sql += """
            AND document_type = %s
        """
        parameters.append(document_type)

    # ========================================================
    # RETRIEVE MORE CANDIDATES THAN REQUIRED
    # ========================================================

    candidate_k = max(top_k * 4, 20)

    sql += """
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    # Query vector for ORDER BY
    parameters.append(query_vector)

    # Number of candidates
    parameters.append(candidate_k)

    # ========================================================
    # EXECUTE QUERY
    # ========================================================

    print("\nConnecting to PostgreSQL...")

    connection = psycopg.connect(**DB_CONFIG)

    print("✓ Connected")

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
        # pgvector <=> returns cosine distance.
        #
        # Because embeddings are normalized:
        #
        # cosine similarity = 1 - cosine distance
        # ----------------------------------------------------

        similarity = 1 - float(distance)

        # ----------------------------------------------------
        # APPLY SIMILARITY THRESHOLD
        # ----------------------------------------------------

        if similarity < similarity_threshold:
            continue

        # ----------------------------------------------------
        # STORE RESULT
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
    # KEEP ONLY TOP_K RESULTS
    # ========================================================

    results = results[:top_k]

    return results


# ============================================================
# TEST RETRIEVER
# ============================================================

if __name__ == "__main__":

    query = "What products and services does Apple offer?"

    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL TEST")
    print("=" * 80)

    print(f"\nQuery: {query}")

    # --------------------------------------------------------
    # RUN RETRIEVER
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
    print(f"TOP {len(results)} RESULTS")
    print("=" * 80)

    if not results:

        print("\n⚠ No documents matched the query.")

    else:

        for rank, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "-" * 80)

            print(f"Rank       : {rank}")
            print(f"Chunk ID   : {result['id']}")
            print(f"Company    : {result['company']}")
            print(f"Year       : {result['year']}")
            print(f"Document   : {result['document_type']}")
            print(f"Page       : {result['page']}")
            print(f"Chunk      : {result['chunk_index']}")
            print(
                f"Distance   : {result['distance']:.4f}"
            )
            print(
                f"Similarity : {result['similarity']:.4f}"
            )

            print("\nSource:")
            print(result["source"])

            print("\nContent:")
            print(result["content"][:1000])

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 80)
    print("RETRIEVAL COMPLETE")
    print("=" * 80)