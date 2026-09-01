from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ingestion.retriever import retrieve_documents
from generation.generation import generate_answer


router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/api/chat")
def chat(request: ChatRequest):

    try:

        # ====================================================
        # COMPARISON QUERY
        # ====================================================

        question_lower = request.question.lower()

        comparison_words = [
            "compare",
            "comparison",
            "versus",
            "vs",
            "difference between",
        ]

        is_comparison = any(
            word in question_lower
            for word in comparison_words
        )

        # ====================================================
        # BALANCED RETRIEVAL FOR COMPARISON
        # ====================================================

        if is_comparison:

            print("\n" + "=" * 80)
            print("COMPARISON QUERY DETECTED")
            print("=" * 80)

            # ------------------------------------------------
            # Retrieve separately for each company
            # ------------------------------------------------

            apple_documents = retrieve_documents(
                query=request.question,
                top_k=3,
                company="Apple",
                year=request.year,
            )

            tesla_documents = retrieve_documents(
                query=request.question,
                top_k=3,
                company="Tesla",
                year=request.year,
            )

            # ------------------------------------------------
            # Combine both company results
            # ------------------------------------------------

            documents = (
                apple_documents +
                tesla_documents
            )

            print(
                f"\n✓ Apple documents: "
                f"{len(apple_documents)}"
            )

            print(
                f"✓ Tesla documents: "
                f"{len(tesla_documents)}"
            )

            print(
                f"✓ Total documents: "
                f"{len(documents)}"
            )

        # ====================================================
        # NORMAL QUERY
        # ====================================================

        else:

            documents = retrieve_documents(
                query=request.question,
                top_k=8,
                company=request.company,
                year=request.year,
            )

        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        answer = generate_answer(
            query=request.question,
            retrieved_documents=documents,
        )

        return {
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )