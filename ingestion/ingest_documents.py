from pathlib import Path
import psycopg

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

MAX_CHUNK_SIZE = 6000

DATA_DIR = Path(
    r"C:\RAG Projects\AI-Powered-Investor-Intelligence-Platform\data\markdown"
)


# ============================================================
# DOCUMENT CONFIGURATION
# ============================================================

DOCUMENTS = [
    {
        "file": DATA_DIR / "2024_Apple.md",
        "company": "Apple",
        "year": 2024,
        "document_type": "Annual Report",
    },
    {
        "file": DATA_DIR / "2024_Tesla.md",
        "company": "Tesla",
        "year": 2024,
        "document_type": "Annual Report",
    },
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embeddings():

    print("\n" + "=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("✓ BGE model loaded")

    return embeddings


# ============================================================
# READ MARKDOWN
# ============================================================

def read_markdown(file_path: Path) -> str:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Markdown file not found:\n{file_path}"
        )

    return file_path.read_text(
        encoding="utf-8"
    )


# ============================================================
# CREATE SEMANTIC CHUNKS
# ============================================================

def create_semantic_chunks(
    text: str,
    embeddings
):

    print("\nCreating semantic chunks...")

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
    )

    chunks = splitter.create_documents(
        [text]
    )

    print(
        f"✓ Semantic chunks generated: {len(chunks)}"
    )

    return chunks


# ============================================================
# SPLIT OVERSIZED CHUNKS
# ============================================================

def enforce_chunk_limit(
    chunks: list[Document]
) -> list[Document]:

    final_chunks = []

    oversized_count = 0

    # Recursive splitter acts as a safety layer.
    # It tries to preserve paragraphs and sentences
    # while keeping chunks below MAX_CHUNK_SIZE.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    for chunk in chunks:

        content = chunk.page_content.strip()

        if not content:
            continue

        if len(content) <= MAX_CHUNK_SIZE:

            final_chunks.append(
                Document(
                    page_content=content,
                    metadata=chunk.metadata,
                )
            )

        else:

            oversized_count += 1

            print(
                f"Splitting oversized chunk: "
                f"{len(content)} characters"
            )

            smaller_chunks = splitter.create_documents(
                [content]
            )

            final_chunks.extend(
                smaller_chunks
            )

    print(
        f"Oversized semantic chunks split: "
        f"{oversized_count}"
    )

    return final_chunks


# ============================================================
# CHUNK STATISTICS
# ============================================================

def print_chunk_statistics(
    chunks: list[Document]
):

    if not chunks:

        print("No chunks generated.")

        return

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print("\n" + "=" * 80)
    print("FINAL CHUNK STATISTICS")
    print("=" * 80)

    print(
        f"Total chunks : {len(chunks)}"
    )

    print(
        f"Smallest     : {min(lengths)} characters"
    )

    print(
        f"Largest      : {max(lengths)} characters"
    )

    print(
        f"Average      : "
        f"{sum(lengths) / len(lengths):.0f} characters"
    )

    if max(lengths) > MAX_CHUNK_SIZE:

        raise ValueError(
            f"Chunk limit violated! "
            f"Largest chunk = {max(lengths)}"
        )

    print(
        f"✓ Maximum chunk size <= "
        f"{MAX_CHUNK_SIZE} characters"
    )


# ============================================================
# CHECK EXISTING DOCUMENT
# ============================================================

def delete_existing_document(
    cursor,
    company: str,
    year: int
):

    cursor.execute(
        """
        DELETE FROM document_chunks
        WHERE company = %s
        AND year = %s;
        """,
        (
            company,
            year,
        ),
    )

    deleted = cursor.rowcount

    if deleted > 0:

        print(
            f"Removed {deleted} existing chunks "
            f"for {company} {year}"
        )


# ============================================================
# STORE CHUNKS + EMBEDDINGS
# ============================================================

def store_chunks(
    chunks: list[Document],
    embeddings,
    document: dict
):

    company = document["company"]
    year = document["year"]
    document_type = document["document_type"]
    file_path = document["file"]

    print("\nConnecting to PostgreSQL...")

    connection = psycopg.connect(
        **DB_CONFIG
    )

    print("✓ Connected to PostgreSQL")

    try:

        with connection.cursor() as cursor:

            # ------------------------------------------------
            # Remove previous chunks for this document
            # ------------------------------------------------

            delete_existing_document(
                cursor,
                company,
                year,
            )

            # ------------------------------------------------
            # Insert new chunks
            # ------------------------------------------------

            total = len(chunks)

            for index, chunk in enumerate(
                chunks
            ):

                content = (
                    chunk.page_content.strip()
                )

                if not content:
                    continue

                # Safety check
                if len(content) > MAX_CHUNK_SIZE:

                    raise ValueError(
                        f"Chunk {index} exceeds "
                        f"maximum size: "
                        f"{len(content)} characters"
                    )

                # Generate BGE embedding
                vector = embeddings.embed_query(
                    content
                )

                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        content,
                        source,
                        company,
                        document_type,
                        year,
                        page,
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
                        %s,
                        %s
                    )
                    """,
                    (
                        content,
                        file_path.name,
                        company,
                        document_type,
                        year,
                        None,
                        index,
                        vector,
                    ),
                )

                if (
                    (index + 1) % 5 == 0
                    or index == total - 1
                ):

                    print(
                        f"Inserted "
                        f"{index + 1}/{total} chunks"
                    )

            connection.commit()

            print(
                f"\n✓ {company} {year} "
                f"successfully stored"
            )

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    document: dict,
    embeddings
):

    company = document["company"]
    year = document["year"]
    file_path = document["file"]

    print("\n" + "#" * 80)
    print(
        f"PROCESSING: {company} {year}"
    )
    print("#" * 80)

    print(
        f"\nDocument: {file_path}"
    )

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    markdown_content = read_markdown(
        file_path
    )

    print(
        f"✓ Document loaded "
        f"({len(markdown_content):,} characters)"
    )

    # --------------------------------------------------------
    # Semantic chunking
    # --------------------------------------------------------

    semantic_chunks = create_semantic_chunks(
        markdown_content,
        embeddings,
    )

    # --------------------------------------------------------
    # Enforce maximum chunk size
    # --------------------------------------------------------

    final_chunks = enforce_chunk_limit(
        semantic_chunks
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_chunk_statistics(
        final_chunks
    )

    # --------------------------------------------------------
    # Store in PostgreSQL
    # --------------------------------------------------------

    store_chunks(
        final_chunks,
        embeddings,
        document,
    )

    return len(final_chunks)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("FINANCIAL DOCUMENT INGESTION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load BGE once
    # --------------------------------------------------------

    embeddings = load_embeddings()

    total_chunks = 0

    # --------------------------------------------------------
    # Process Apple + Tesla
    # --------------------------------------------------------

    for document in DOCUMENTS:

        count = process_document(
            document,
            embeddings,
        )

        total_chunks += count

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(
        f"Total chunks inserted: "
        f"{total_chunks}"
    )

    print("\nDocuments processed:")

    for document in DOCUMENTS:

        print(
            f"  ✓ {document['company']} "
            f"{document['year']}"
        )

    print("=" * 80)