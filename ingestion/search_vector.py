import psycopg

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"

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
# USER QUERY
# ============================================================

query = "What are Tesla's major risk factors?"

print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)

print(query)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

query_vector = embeddings.embed_query(query)

print("\n✓ Query embedding generated")
print("Dimensions:", len(query_vector))


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

print("\nConnecting to PostgreSQL...")

connection = psycopg.connect(**DB_CONFIG)

print("✓ Connected")


# ============================================================
# VECTOR SEARCH
# ============================================================

with connection.cursor() as cursor:

    cursor.execute(
        """
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

        ORDER BY embedding <=> %s::vector

        LIMIT 5;
        """,
        (
            query_vector,
            query_vector,
        )
    )

    results = cursor.fetchall()


connection.close()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TOP 5 SEMANTIC SEARCH RESULTS")
print("=" * 80)


if not results:

    print("\nNo results found.")

else:

    for rank, result in enumerate(results, start=1):

        (
            chunk_id,
            content,
            source,
            company,
            document_type,
            year,
            page,
            chunk_index,
            distance,
        ) = result

        print("\n" + "-" * 80)

        print(f"Rank       : {rank}")
        print(f"Chunk ID    : {chunk_id}")
        print(f"Company    : {company}")
        print(f"Year       : {year}")
        print(f"Source     : {source}")
        print(f"Chunk      : {chunk_index}")
        print(f"Distance   : {distance:.4f}")

        print("\nContent:")
        print(content[:1200])

print("\n" + "=" * 80)