from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    station_id: str | None = Field(default=None, examples=["WS-001"])
    climate_zone: str | None = None
    season: str | None = None
    weather_condition: str | None = None
    precipitation_type: str | None = None
    road_surface: str | None = None
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    is_weekend: int | None = Field(default=None, ge=0, le=1)
    hour_of_day: int | None = Field(default=None, ge=0, le=23)
    month: int | None = Field(default=None, ge=1, le=12)
    timestamp: str | None = Field(default=None, description="Optional ISO timestamp.")
    date: str | None = Field(default=None, description="Optional date string.")

    temperature_c: float | None = None
    feels_like_c: float | None = None
    humidity_pct: float | None = None
    pressure_hpa: float | None = None
    wind_speed_kmh: float | None = None
    wind_gust_kmh: float | None = None
    precipitation_mm: float | None = None
    cloud_cover_pct: float | None = None
    visibility_km: float | None = None
    uv_index: float | None = None


class PredictResponse(BaseModel):
    umbrella_needed: int
    clothing_recommendation: str
    outdoor_suitability_score: float
    recommendation_text: str
    confidence: dict[str, Any]

