from pathlib import Path

import psycopg
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_markdown_converter import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown


router = APIRouter()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_DIR = BASE_DIR / "data" / "raw_pdfs"
MARKDOWN_DIR = BASE_DIR / "data" / "markdown"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"

print("Loading BGE model for ingestion...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)

print("✓ Ingestion BGE model loaded")


# ============================================================
# PDF UPLOAD + INGESTION
# ============================================================

@router.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected."
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported."
            )


        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        pdf_path = UPLOAD_DIR / Path(file.filename).name

        with open(pdf_path, "wb") as buffer:
            shutil_data = await file.read()
            buffer.write(shutil_data)

        print(f"\n✓ PDF uploaded: {pdf_path.name}")


        # ----------------------------------------------------
        # Convert PDF → Markdown
        # ----------------------------------------------------

        print("Converting PDF to Markdown...")

        converter = PDFToMarkdownConverter()

        markdown_path = converter.convert_pdf(
            pdf_path=str(pdf_path),
            output_dir=str(MARKDOWN_DIR)
        )

        print(f"✓ Markdown created: {markdown_path}")


        # ----------------------------------------------------
        # Semantic chunking
        # ----------------------------------------------------

        print("Creating semantic chunks...")

        chunks = chunk_markdown(
            markdown_file=markdown_path,
            embeddings=embeddings
        )

        print(f"✓ Generated {len(chunks)} chunks")


        # ----------------------------------------------------
        # Extract company/year from filename
        #
        # Example:
        # 2024_Apple.pdf
        # 2024_Microsoft.pdf
        # ----------------------------------------------------

        filename = Path(file.filename).stem

        parts = filename.split("_", 1)

        if len(parts) == 2:

            year = int(parts[0]) if parts[0].isdigit() else None
            company = parts[1]

        else:

            year = None
            company = filename


        # ----------------------------------------------------
        # Store chunks in PostgreSQL
        # ----------------------------------------------------

        print("Connecting to PostgreSQL...")

        connection = psycopg.connect(DATABASE_URL)

        print("✓ Connected to PostgreSQL")

        inserted = 0

        try:

            with connection.cursor() as cursor:

                for index, chunk in enumerate(chunks):

                    content = chunk.page_content.strip()

                    if not content:
                        continue


                    # Generate embedding
                    vector = embeddings.embed_query(content)


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
                            content,connection = psycopg.connect(DATABASE_URL)
                            Path(markdown_path).name,
                            company,
                            "Annual Report",
                            year,
                            None,
                            index,
                            vector,
                        )
                    )

                    inserted += 1


                    if (
                        inserted % 10 == 0
                        or index == len(chunks) - 1
                    ):
                        print(
                            f"Inserted "
                            f"{inserted}/{len(chunks)} chunks"
                        )


                connection.commit()

        finally:

            connection.close()


        print("\n✓ INGESTION COMPLETE")


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "success": True,
            "message": "Document uploaded and ingested successfully.",
            "filename": file.filename,
            "company": company,
            "year": year,
            "chunks_created": len(chunks),
            "chunks_inserted": inserted,
        }


    except HTTPException:
        raise


    except Exception as e:

        print("\n❌ INGESTION FAILED")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Error processing report: {str(e)}"
        )