import psycopg


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


def create_financial_metrics_table():

    print("Connecting to PostgreSQL...")

    connection = psycopg.connect(**DB_CONFIG)

    try:

        with connection.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_metrics (

                    id SERIAL PRIMARY KEY,

                    company VARCHAR(100) NOT NULL,

                    year VARCHAR(10) NOT NULL,

                    revenue TEXT,

                    net_income TEXT,

                    operating_income TEXT,

                    cash_flow TEXT,

                    total_assets TEXT,

                    total_liabilities TEXT,

                    risk_factors TEXT,

                    growth_drivers TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT unique_company_year
                    UNIQUE (company, year)

                );
            """)

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