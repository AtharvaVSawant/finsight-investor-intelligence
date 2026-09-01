import psycopg

from fastapi import APIRouter


router = APIRouter()


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


@router.get("/metrics")

def format_money(value):
    if value is None:
        return None

    try:
        # Remove existing formatting if value is already a string
        cleaned = str(value).replace("$", "").replace(",", "").strip()

        number = float(cleaned)

        if number < 0:
            return f"(${abs(number):,.0f})"

        return f"${number:,.0f}"

    except (ValueError, TypeError):
        return str(value)

def get_metrics():

    connection = psycopg.connect(**DB_CONFIG)

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT DISTINCT ON (company, year)
                    company,
                    year,
                    revenue,
                    net_income,
                    operating_income,
                    cash_flow,
                    total_assets,
                    total_liabilities,
                    risk_factors,
                    growth_drivers,
                    created_at
                FROM financial_metrics
                ORDER BY company, year, created_at DESC
                """
            )

            rows = cursor.fetchall()

    finally:

        connection.close()

    metrics = []

    for row in rows:

        metrics.append(
            {
                "company": row[0],
                "year": row[1],
                "revenue": format_money(row[2]),
                "net_income": format_money(row[3]),
                "operating_income": format_money(row[4]),
                "cash_flow": format_money(row[5]),
                "total_assets": format_money(row[6]),
                "total_liabilities": format_money(row[7]),
                "risk_factors": row[8],
                "growth_drivers": row[9],
                "created_at": (
                    row[10].isoformat()
                    if row[10]
                    else None
                ),
            }
        )

    return metrics