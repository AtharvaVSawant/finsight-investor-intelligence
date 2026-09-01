from kpi.kpi_extractor import extract_financial_metrics


# ============================================================
# TEST KPI EXTRACTION
# ============================================================

if __name__ == "__main__":

    company = "Apple"
    year = 2024

    print("\n" + "=" * 80)
    print("KPI EXTRACTION TEST")
    print("=" * 80)

    print(f"\nCompany: {company}")
    print(f"Year   : {year}")

    results = extract_financial_metrics(
        company=company,
        year=year
    )

    print("\n" + "=" * 80)
    print("EXTRACTED FINANCIAL KPIs")
    print("=" * 80)

    for key, value in results.items():

        print(f"{key:45}: {value}")

    print("=" * 80)