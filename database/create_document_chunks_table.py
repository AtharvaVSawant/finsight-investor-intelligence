import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file.")


def create_document_chunks_table():

    print("Connecting to PostgreSQL...")

    connection = psycopg.connect(DATABASE_URL)

    try:
        with connection.cursor() as cursor:

            # Enable pgvector
            cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            # Create document chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (

                    id SERIAL PRIMARY KEY,

                    content TEXT NOT NULL,

                    source VARCHAR(255),

                    company VARCHAR(100),

                    document_type VARCHAR(100),

                    year INTEGER,

                    page INTEGER,

                    chunk_index INTEGER NOT NULL,

                    embedding VECTOR(768),

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                );
            """)

            # Useful index for metadata filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_company
                ON document_chunks(company);
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_year
                ON document_chunks(year);
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_company_year
                ON document_chunks(company, year);
            """)

        connection.commit()

        print("✓ pgvector extension enabled")
        print("✓ document_chunks table created/verified")
        print("✓ Embedding dimension: 768")
        print("✓ Metadata indexes created")

    except Exception as e:

        connection.rollback()

        print("❌ Failed to create document_chunks table")
        print(f"Error: {e}")

        raise

    finally:
        connection.close()


if __name__ == "__main__":
    create_document_chunks_table()