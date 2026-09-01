import os
import json
import psycopg

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


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

GROQ_MODEL = "openai/gpt-oss-120b"

MAX_CONTEXT_CHARS = 20000


# ============================================================
# BGE EMBEDDINGS
# ============================================================

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
# GROQ CLIENT
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# FINANCIAL METRICS SCHEMA
# ============================================================

class FinancialMetrics(BaseModel):

    revenue: str | int | None = Field(
        None,
        alias="Revenue"
    )

    net_income: str | int | None = Field(
        None,
        alias="Net Income"
    )

    operating_income: str | int | None = Field(
        None,
        alias="Operating Income"
    )

    cash_flow: str | int | None = Field(
        None,
        alias="Cash Flow from Operating Activities"
    )

    total_assets: str | int | None = Field(
        None,
        alias="Total Assets"
    )

    total_liabilities: str | int | None = Field(
        None,
        alias="Total Liabilities"
    )

    risk_factors: str | list | None = Field(
        None,
        alias="Top Risk Factors"
    )

    growth_drivers: str | list | None = Field(
        None,
        alias="Top Growth Drivers"
    )


# ============================================================
# RETRIEVE FINANCIAL CONTEXT
# ============================================================

def retrieve_context(
    company: str,
    year: int,
    top_k: int = 10
) -> str:

    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year}
    """

    query_vector = embeddings.embed_query(query)

    sql = """
        SELECT
            content,
            source,
            page,
            chunk_index,
            embedding <=> %s::vector AS distance

        FROM document_chunks

        WHERE embedding IS NOT NULL
          AND company = %s
          AND year = %s

        ORDER BY embedding <=> %s::vector

        LIMIT %s
    """

    print("\nRetrieving financial documents...")

    connection = psycopg.connect(**DB_CONFIG)

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    query_vector,
                    company,
                    year,
                    query_vector,
                    top_k
                )
            )

            rows = cursor.fetchall()

    finally:

        connection.close()

    print(f"✓ Retrieved {len(rows)} documents")

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    current_length = 0

    for index, row in enumerate(rows, start=1):

        (
            content,
            source,
            page,
            chunk_index,
            distance
        ) = row

        chunk_text = f"""
    SOURCE {index}

    Document: {source}
    Page: {page}
    Chunk: {chunk_index}

    Content:
    {content}
    """

        chunk_length = len(chunk_text)

        # If this single chunk is too large,
        # truncate it instead of breaking completely.
        if chunk_length > MAX_CONTEXT_CHARS:

            remaining = MAX_CONTEXT_CHARS - current_length

            if remaining <= 0:
                break

            chunk_text = chunk_text[:remaining]
            chunk_length = len(chunk_text)

        elif current_length + chunk_length > MAX_CONTEXT_CHARS:

            remaining = MAX_CONTEXT_CHARS - current_length

            if remaining <= 0:
                break

            chunk_text = chunk_text[:remaining]
            chunk_length = len(chunk_text)

        context_parts.append(chunk_text)

        current_length += chunk_length

        if current_length >= MAX_CONTEXT_CHARS:
            break

    context = "\n".join(context_parts)

    return context


# ============================================================
# EXTRACTION PROMPT
# ============================================================

def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:

    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use ONLY the provided context.
- Return null only when the requested information is genuinely not present in the context.
- Financial values must match the report exactly.
- Do not calculate or infer financial values.
- For Top Risk Factors, identify the most important risks, uncertainties, threats, or adverse factors explicitly discussed in the provided context.
- Risk factors may come from sections discussing risks, risk factors, uncertainties, competition, regulation, supply chain, demand, operations, technology, legal matters, or financial risks.
- Return Top Risk Factors as a concise list of 3 to 5 items when the context contains relevant risk information.
- For Top Growth Drivers, identify the most important growth opportunities or drivers explicitly discussed in the context.
- Return Top Growth Drivers as a concise list of 3 to 5 items when the context contains relevant information.
- Do not invent risk factors or growth drivers.
- Return valid JSON only.

Use exactly these JSON keys:

{{
    "Revenue": null,
    "Net Income": null,
    "Operating Income": null,
    "Cash Flow from Operating Activities": null,
    "Total Assets": null,
    "Total Liabilities": null,
    "Top Risk Factors": null,
    "Top Growth Drivers": null
}}
"""


# ============================================================
# EXTRACT FINANCIAL METRICS
# ============================================================

def extract_financial_metrics(
    company: str,
    year: int
) -> dict:

    context = retrieve_context(
        company=company,
        year=year,
        top_k=15
    )

    print(
        f"✓ LLM context size: "
        f"{len(context):,} characters"
    )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    print("\nExtracting KPIs using Groq...")

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert financial "
                    "analyst. Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    metrics = json.loads(content)

    return metrics


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":

    from database.save_metrics import save_metrics

    company = "Tesla"
    year = 2024

    metrics = extract_financial_metrics(
        company=company,
        year=year
    )

    print("\n" + "=" * 80)
    print("EXTRACTED FINANCIAL KPIs")
    print("=" * 80)

    for key, value in metrics.items():
        print(
            f"{key:45}: {value}"
        )

    save_metrics(
        company=company,
        year=year,
        metrics=metrics
    )