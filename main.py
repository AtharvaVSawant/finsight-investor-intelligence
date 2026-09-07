from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes.metrics import router as metrics_router, get_metrics
from routes.chat import router as chat_router
from routes.ingestion import router as ingestion_router


app = FastAPI(
    title="AI-Powered Investor Intelligence Platform",
    version="1.0.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(metrics_router)
app.include_router(chat_router)
app.include_router(ingestion_router)


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/")
def dashboard(request: Request):

    metrics = get_metrics()

    # --------------------------------------------------------
    # Keep latest metric record for each company
    # --------------------------------------------------------

    companies = {}

    for row in metrics:

        company = row["company"]

        if company not in companies:
            companies[company] = row

    company_list = list(companies.values())

    # --------------------------------------------------------
    # Dashboard statistics
    # --------------------------------------------------------

    total_companies = len(company_list)

    total_reports = len(metrics)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "companies": company_list,
            "total_companies": total_companies,
            "total_reports": total_reports,
        }
    )

@app.get("/health")
def health():
    return {"status": "healthy"}