from ingestion.retriever import retrieve_documents
from generation.generation import generate_answer


def ask_question(
    query: str,
    company: str | None = None,
    year: int | None = None,
    top_k: int = 5,
):

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    documents = retrieve_documents(
        query=query,
        top_k=top_k,
        company=company,
        year=year,
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    answer = generate_answer(
        query=query,
        retrieved_documents=documents,
    )

    return {
        "answer": answer,
        "sources": documents,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What products and services does Apple offer?"
    )

    result = ask_question(
        query=question,
        company="Apple",
        year=2024,
    )

    print("\n" + "=" * 80)
    print("RAG ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"""
Source : {source['source']}
Page   : {source['page']}
Chunk  : {source['chunk_index']}
"""
        )