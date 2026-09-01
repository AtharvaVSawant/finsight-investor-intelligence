from fastapi import APIRouter
import psycopg

router = APIRouter()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


@router.get("/metrics")
def get_metrics():

    connection = psycopg.connect(**DB_CONFIG)

    try:

        with connection.cursor() as cursor:

            cursor.execute("""
                SELECT
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
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY company, year
                            ORDER BY created_at DESC
                        ) AS rn
                    FROM financial_metrics
                ) t
                WHERE rn = 1
                ORDER BY company;
            """)

            rows = cursor.fetchall()

            columns = [
                "company",
                "year",
                "revenue",
                "net_income",
                "operating_income",
                "cash_flow",
                "total_assets",
                "total_liabilities",
                "risk_factors",
                "growth_drivers",
                "created_at",
            ]

            return [
                dict(zip(columns, row))
                for row in rows
            ]

    finally:
        connection.close()