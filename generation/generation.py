import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# CONTEXT LIMIT
# ============================================================

# Maximum characters from each retrieved chunk sent to the LLM.
# This prevents large RAG contexts from exceeding Groq's TPM limit.
MAX_CHARS_PER_DOCUMENT = 1800


# ============================================================
# RAG GENERATION
# ============================================================

def generate_answer(
    query: str,
    retrieved_documents: list[dict],
) -> str:

    if not retrieved_documents:
        return (
            "I could not find relevant information "
            "in the available financial documents."
        )

    # --------------------------------------------------------
    # BUILD COMPACT CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        content = document.get("content", "")

        # ----------------------------------------------------
        # Limit large chunks
        # ----------------------------------------------------

        if len(content) > MAX_CHARS_PER_DOCUMENT:
            content = content[:MAX_CHARS_PER_DOCUMENT]

            # Avoid cutting a word in half
            last_space = content.rfind(" ")

            if last_space > 0:
                content = content[:last_space]

            content += "\n[Content truncated]"

        source = document.get("source", "Unknown")
        page = document.get("page", "Unknown")
        company = document.get("company", "Unknown")
        year = document.get("year", "Unknown")

        context_parts.append(
            f"""SOURCE {index}
Company: {company}
Year: {year}
Document: {source}
Page: {page}

Content:
{content}
"""
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are an AI financial research assistant.

Answer the user's question using ONLY the supplied financial
document context.

Rules:

- Do not invent information.
- Do not use outside knowledge.
- Preserve financial numbers exactly as provided.
- If the answer is not supported by the context, say so.
- Do not perform calculations unless explicitly requested.
- Use clean Markdown.
- Use a clear title.
- Group related information under headings.
- Use one bullet point per financial fact.
- Leave a blank line between sections.
- Add [Source N] to factual statements.
- End with a short "Overall Assessment".
- Do not use tables unless explicitly requested.
- Do not mention the RAG system, retrieved chunks, prompts,
  or internal processing.
- Do not mention that you are an AI.

Preferred structure:

## Company — Year Financial Analysis

### Section Name

- Financial fact [Source 1]
- Financial fact [Source 2]

### Another Section

- Financial fact [Source 1]

### Overall Assessment

Short conclusion based only on the supplied documents.
"""

    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""USER QUESTION:
{query}

FINANCIAL DOCUMENT CONTEXT:
{context}

Answer the question using only the financial document context.

Requirements:
- Start with a clear Markdown title.
- Organize related information into meaningful sections.
- Use separate bullet points for separate facts.
- Keep the answer concise.
- Preserve financial values exactly.
- Add [Source N] after factual claims.
- End with "### Overall Assessment".
"""

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    total_context_chars = len(context)

    print(
        f"✓ RAG context size: "
        f"{total_context_chars:,} characters"
    )

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    answer = response.choices[0].message.content

    return answer.strip()