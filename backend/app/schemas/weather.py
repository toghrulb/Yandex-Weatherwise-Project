"""
Pydantic schemas for API responses.
"""
from __future__ import annotations
from pydantic import BaseModel


# ── Station ────────────────────────────────────────────────────────────
class StationInfo(BaseModel):
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int
    climate_zone: str


# ── Current weather (raw values for the AI explainability layer) ──────
class RawWeather(BaseModel):
    temperature_c: float
    feels_like_c: float
    dew_point_c: float
    humidity_pct: float
    pressure_hpa: float
    wind_speed_kmh: float
    wind_direction_deg: float
    wind_gust_kmh: float
    precipitation_mm: float
    precipitation_type: str          # rain / snow / none
    cloud_cover_pct: float
    visibility_km: float
    uv_index: float
    weather_condition: str           # clear / rain / snow / …
    is_thunderstorm: bool
    is_day: bool


# ── Hero Card ──────────────────────────────────────────────────────────
class HeroCardResponse(BaseModel):
    station: StationInfo
    timestamp: str
    raw: RawWeather
    # Recommendation outputs
    outdoor_suitability_score: float
    umbrella_needed: bool
    clothing_recommendation: str
    recommendation_headline: str
    recommendation_text: str


# ── Time Strip (morning / afternoon / evening) ────────────────────────
class TimeSlot(BaseModel):
    period: str                       # morning / afternoon / evening
    clothing: str
    tip: str
    go_indicator: str                 # green / orange / red


class DailySummaryResponse(BaseModel):
    station_id: str
    date: str
    temp_min_c: float
    temp_max_c: float
    temp_avg_c: float
    total_precipitation_mm: float
    dominant_condition: str
    avg_outdoor_suitability: float
    best_outdoor_hour: int
    umbrella_recommended: bool
    day_summary_text: str
    time_strip: list[TimeSlot]


# ── Activity Card ──────────────────────────────────────────────────────
class ActivityCardResponse(BaseModel):
    activity_type: str
    general_suitability_score: float
    activity_suitability_score: float
    go_or_no: bool
    umbrella_needed: bool
    clothing_recommendation: str
    activity_advice: str


# ── Forecast accuracy ─────────────────────────────────────────────────
class ForecastAccuracyResponse(BaseModel):
    station_id: str
    total_forecasts: int
    umbrella_accuracy_pct: float
    clothing_accuracy_pct: float
    condition_accuracy_pct: float
    avg_temp_error_c: float
    avg_precip_error_mm: float
    avg_wind_error_kmh: float
    by_lead_time: list[LeadTimeAccuracy]


class LeadTimeAccuracy(BaseModel):
    lead_time_hours: int
    count: int
    umbrella_accuracy_pct: float
    clothing_accuracy_pct: float
    condition_accuracy_pct: float

# Rebuild model to resolve forward ref
ForecastAccuracyResponse.model_rebuild()
