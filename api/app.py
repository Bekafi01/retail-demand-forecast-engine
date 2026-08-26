"""FastAPI application initialization with CORS and route mounting."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Retail Demand Forecast Engine API",
    description="Production-grade demand forecasting API featuring hierarchical coherence, conformal prediction intervals, and drift monitoring.",
    version="0.1.0",
)

# Enable CORS for Streamlit and web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecast Engine API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
    }
