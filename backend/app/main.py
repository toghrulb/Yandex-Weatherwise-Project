"""
WeatherWise backend FastAPI application entrypoint.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as app_router
from backend.app.api.v1.router import api_router
from backend.app.core.config import CORS_ORIGINS
from backend.app.services.data_loader import DataStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    DataStore.load()
    yield


app = FastAPI(
    title="WeatherWise API",
    version="0.1.0",
    description="Backend service for weather-to-action recommendations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(app_router)
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "WeatherWise API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

