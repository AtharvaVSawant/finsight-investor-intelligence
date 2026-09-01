import psycopg
import re


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
}


def clean_numeric(value):
    """
    Convert financial values such as:
        "$ 97,690"
        "$7,153"
        "97,690"
        "7153 million"

    into numeric values suitable for PostgreSQL.
    """

    if value is None:
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return value

    value = str(value).strip()

    if not value:
        return None

    # Remove currency symbols, commas, spaces and other text
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.replace("₹", "")
    value = value.replace("€", "")
    value = value.replace("£", "")

    # Extract first numeric value
    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if not match:
        return None

    number = match.group()

    try:
        if "." in number:
            return float(number)

        return int(number)

    except ValueError:
        return None


def clean_text_list(value):
    """
    Convert a list into readable text.

    Example:
        ["Risk A", "Risk B"]

    becomes:

        Risk A
        Risk B
    """

    if value is None:
        return None

    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value)

    return str(value).strip()


def save_metrics(
    company: str,
    year: int,
    metrics: dict
) -> None:

    connection = psycopg.connect(**DB_CONFIG)

    try:

        with connection.cursor() as cursor:

            # ---------------------------------------------------------
            # Extract and clean risk / growth information
            # ---------------------------------------------------------

            risk_factors = metrics.get(
                "Top Risk Factors"
            )

            growth_drivers = metrics.get(
                "Top Growth Drivers"
            )

            risk_factors = clean_text_list(
                risk_factors
            )

            growth_drivers = clean_text_list(
                growth_drivers
            )

            # ---------------------------------------------------------
            # Clean financial numbers
            # ---------------------------------------------------------

            revenue = clean_numeric(
                metrics.get("Revenue")
            )

            net_income = clean_numeric(
                metrics.get("Net Income")
            )

            operating_income = clean_numeric(
                metrics.get("Operating Income")
            )

            cash_flow = clean_numeric(
                metrics.get(
                    "Cash Flow from Operating Activities"
                )
            )

            total_assets = clean_numeric(
                metrics.get("Total Assets")
            )

            total_liabilities = clean_numeric(
                metrics.get("Total Liabilities")
            )

            # ---------------------------------------------------------
            # UPSERT
            #
            # If company + year does not exist:
            #     INSERT
            #
            # If company + year already exists:
            #     UPDATE
            # ---------------------------------------------------------

            query = """
                INSERT INTO financial_metrics (
                    company,
                    year,
                    revenue,
                    net_income,
                    operating_income,
                    cash_flow,
                    total_assets,
                    total_liabilities,
                    risk_factors,
                    growth_drivers
                )

                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )

                ON CONFLICT (company, year)

                DO UPDATE SET
                    revenue = EXCLUDED.revenue,
                    net_income = EXCLUDED.net_income,
                    operating_income = EXCLUDED.operating_income,
                    cash_flow = EXCLUDED.cash_flow,
                    total_assets = EXCLUDED.total_assets,
                    total_liabilities = EXCLUDED.total_liabilities,
                    risk_factors = EXCLUDED.risk_factors,
                    growth_drivers = EXCLUDED.growth_drivers
            """

            cursor.execute(
                query,
                (
                    company,
                    int(year),

                    revenue,
                    net_income,
                    operating_income,
                    cash_flow,

                    total_assets,
                    total_liabilities,

                    risk_factors,
                    growth_drivers,
                )
            )

        connection.commit()

        print(
            f"✓ Successfully saved/updated metrics "
            f"for {company} {year}"
        )

    except Exception as e:

        connection.rollback()

        print(
            f"❌ Failed to save metrics "
            f"for {company} {year}"
        )

        print(f"Error: {e}")

        raise

    finally:

        connection.close()