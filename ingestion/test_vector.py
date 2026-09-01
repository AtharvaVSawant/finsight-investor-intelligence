import psycopg

from langchain_huggingface import HuggingFaceEmbeddings


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


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


# Test text
text = "Apple reported strong revenue growth in 2024."


print("Generating embedding...")

vector = embeddings.embed_query(text)

print("✓ Embedding generated")
print("Dimensions:", len(vector))


print("Connecting to PostgreSQL...")

connection = psycopg.connect(**DB_CONFIG)

with connection.cursor() as cursor:

    cursor.execute(
        """
        INSERT INTO document_chunks (
            content,
            source,
            company,
            document_type,
            year,
            chunk_index,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        """,
        (
            text,
            "test_document",
            "Apple",
            "Annual Report",
            2024,
            0,
            vector,
        )
    )

    connection.commit()


connection.close()

print("✓ Vector successfully stored in PostgreSQL")