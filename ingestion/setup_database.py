import psycopg


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


def setup_database():

    print("Connecting to PostgreSQL...")

    connection = psycopg.connect(**DB_CONFIG)

    print("Connected successfully!")

    with connection.cursor() as cursor:

        # Enable pgvector
        cursor.execute(
            "CREATE EXTENSION IF NOT EXISTS vector;"
        )

        # Create document chunks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,

                content TEXT NOT NULL,

                source TEXT,

                company TEXT,

                document_type TEXT,

                year INTEGER,

                page INTEGER,

                chunk_index INTEGER,

                embedding vector(768)
            );
            """
        )

        connection.commit()

    connection.close()

    print("✓ pgvector enabled")
    print("✓ document_chunks table created")
    print("✓ Database setup completed")


if __name__ == "__main__":
    setup_database()