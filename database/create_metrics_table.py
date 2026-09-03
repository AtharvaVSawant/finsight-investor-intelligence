import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def create_financial_metrics_table():

    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL is not set in the .env file."
        )

    print("Connecting to PostgreSQL...")

    connection = psycopg.connect(DATABASE_URL)

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_metrics (

                    id SERIAL PRIMARY KEY,

                    company VARCHAR(100) NOT NULL,

                    year VARCHAR(10) NOT NULL,

                    revenue NUMERIC,

                    net_income NUMERIC,

                    operating_income NUMERIC,

                    cash_flow NUMERIC,

                    total_assets NUMERIC,

                    total_liabilities NUMERIC,

                    risk_factors TEXT,

                    growth_drivers TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT unique_company_year
                    UNIQUE (company, year)

                );
                """
            )

        connection.commit()

        print("✓ financial_metrics table created/verified")
        print("✓ Unique constraint: company + year")

    except Exception as e:

        connection.rollback()

        print("❌ Failed to create/verify financial_metrics table")
        print(f"Error: {e}")

        raise

    finally:

        connection.close()


if __name__ == "__main__":

    create_financial_metrics_table()