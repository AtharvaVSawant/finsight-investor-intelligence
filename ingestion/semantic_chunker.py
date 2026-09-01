from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Maximum size of a final chunk in characters.
# This prevents huge semantic chunks such as 24,000+ characters.
MAX_CHUNK_SIZE = 6000

# Small overlap only when an oversized semantic chunk
# needs to be split further.
CHUNK_OVERLAP = 500


# ============================================================
# READ MARKDOWN
# ============================================================

def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content from a file.
    """

    return Path(markdown_file).read_text(
        encoding="utf-8"
    )


# ============================================================
# SPLIT OVERSIZED SEMANTIC CHUNKS
# ============================================================

def split_oversized_chunk(
    document: Document,
) -> list[Document]:
    """
    Split a semantic chunk if it exceeds MAX_CHUNK_SIZE.

    SemanticChunker determines meaningful semantic boundaries,
    while RecursiveCharacterTextSplitter acts as a safety
    mechanism to prevent excessively large chunks.
    """

    if len(document.page_content) <= MAX_CHUNK_SIZE:
        return [document]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    smaller_chunks = splitter.split_documents(
        [document]
    )

    return smaller_chunks


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def chunk_markdown(
    markdown_file: str,
    embeddings,
) -> list[Document]:
    """
    Generate semantic chunks and enforce a maximum chunk size.

    First:
        SemanticChunker creates semantically meaningful chunks.

    Then:
        Oversized chunks are recursively split so that no final
        chunk exceeds MAX_CHUNK_SIZE.
    """

    markdown_content = read_markdown(
        markdown_file
    )

    # --------------------------------------------------------
    # STEP 1: Semantic chunking
    # --------------------------------------------------------

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
    )

    semantic_chunks = semantic_splitter.create_documents(
        [markdown_content]
    )

    print(
        f"Semantic chunks generated: "
        f"{len(semantic_chunks)}"
    )

    # --------------------------------------------------------
    # STEP 2: Enforce maximum chunk size
    # --------------------------------------------------------

    final_chunks = []

    oversized_count = 0

    for chunk in semantic_chunks:

        if len(chunk.page_content) > MAX_CHUNK_SIZE:

            oversized_count += 1

            print(
                f"Splitting oversized chunk: "
                f"{len(chunk.page_content)} characters"
            )

            smaller_chunks = split_oversized_chunk(
                chunk
            )

            final_chunks.extend(
                smaller_chunks
            )

        else:

            final_chunks.append(
                chunk
            )

    # --------------------------------------------------------
    # STEP 3: Final safety check
    # --------------------------------------------------------

    oversized_final_chunks = [
        chunk
        for chunk in final_chunks
        if len(chunk.page_content) > MAX_CHUNK_SIZE
    ]

    if oversized_final_chunks:

        raise ValueError(
            "Final chunking produced chunks larger "
            f"than MAX_CHUNK_SIZE ({MAX_CHUNK_SIZE})."
        )

    print(
        f"Oversized semantic chunks split: "
        f"{oversized_count}"
    )

    print(
        f"Final chunks: "
        f"{len(final_chunks)}"
    )

    return final_chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Loading BGE model...")

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

    markdown_file = (
        "data/markdown/2024_Tesla.md"
    )

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings,
    )

    print("\n" + "=" * 80)
    print("FINAL CHUNK STATISTICS")
    print("=" * 80)

    if chunks:

        sizes = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        print(
            f"Total chunks : {len(chunks)}"
        )

        print(
            f"Smallest     : {min(sizes)} characters"
        )

        print(
            f"Largest      : {max(sizes)} characters"
        )

        print(
            f"Average      : "
            f"{sum(sizes) / len(sizes):.0f} characters"
        )

    print("\nFirst 3 chunks:")

    for index, chunk in enumerate(chunks[:3]):

        print("\n" + "-" * 80)
        print(f"Chunk {index}")
        print("-" * 80)

        print(
            f"Characters: "
            f"{len(chunk.page_content)}"
        )

        print(
            chunk.page_content[:1000]
        )