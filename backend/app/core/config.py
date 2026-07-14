"""
Station registry and application configuration.

"""
from __future__ import annotations

import os
from pathlib import Path


from ml.src.utils.paths import (
    ARTIFACTS_DIR,
    CLOTHING_MODEL_PATH,
    SUITABILITY_MODEL_PATH,
    UMBRELLA_MODEL_PATH,
)


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]          # Yandex_Weatherwise_Project/
DATA_RAW = Path(os.environ.get("DATA_RAW_PATH", str(PROJECT_ROOT / "data" / "raw")))
ML_ARTIFACTS = Path(os.environ.get("ML_ARTIFACTS_PATH", str(PROJECT_ROOT / "ml" / "artifacts")))

# ── CORS ───────────────────────────────────────────────────────────────
CORS_ORIGINS = ["*"]   # tighten in production

# ── Station Registry ──────────────────────────────────────────────────
# Matches the 10 stations in the provided dataset
STATIONS = {
    "WS-001": {"name": "Sivas Merkez",  "lat": 39.748, "lon": 37.016, "elevation_m": 1285, "climate_zone": "semi-arid_continental"},
    "WS-002": {"name": "Sivas Kuzey",   "lat": 39.790, "lon": 37.000, "elevation_m": 1310, "climate_zone": "semi-arid_continental"},
    "WS-003": {"name": "Sivas Güney",   "lat": 39.700, "lon": 37.030, "elevation_m": 1260, "climate_zone": "semi-arid_continental"},
    "WS-004": {"name": "Kangal",        "lat": 39.241, "lon": 37.387, "elevation_m": 1520, "climate_zone": "highland"},
    "WS-005": {"name": "Zara",          "lat": 39.894, "lon": 37.749, "elevation_m": 1350, "climate_zone": "semi-arid_continental"},
    "WS-006": {"name": "Suşehri",       "lat": 40.163, "lon": 38.089, "elevation_m": 980,  "climate_zone": "temperate"},
    "WS-007": {"name": "Ankara",        "lat": 39.925, "lon": 32.866, "elevation_m": 890,  "climate_zone": "semi-arid_continental"},
    "WS-008": {"name": "Kayseri",       "lat": 38.735, "lon": 35.478, "elevation_m": 1054, "climate_zone": "semi-arid_continental"},
    "WS-009": {"name": "Erzincan",      "lat": 39.746, "lon": 39.491, "elevation_m": 1185, "climate_zone": "highland"},
    "WS-010": {"name": "Malatya",       "lat": 38.348, "lon": 38.320, "elevation_m": 948,  "climate_zone": "semi-arid"},
}

# ── Open-Meteo ────────────────────────────────────────────────────────
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_PARAMS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "uv_index",
    "is_day",
])
OPEN_METEO_CURRENT_PARAMS = OPEN_METEO_HOURLY_PARAMS   # same set for current

# ── WMO weather code → dataset weather_condition mapping ─────────────
WMO_TO_CONDITION: dict[int, str] = {
    0: "clear",
    1: "partly_cloudy",
    2: "partly_cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    56: "drizzle",
    57: "drizzle",
    61: "rain",
    63: "rain",
    65: "heavy_rain",
    66: "rain",
    67: "heavy_rain",
    71: "snow",
    73: "snow",
    75: "heavy_snow",
    77: "snow",
    80: "rain",
    81: "rain",
    82: "heavy_rain",
    85: "snow",
    86: "heavy_snow",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}

# ── Clothing categories (cold → hot) ─────────────────────────────────
CLOTHING_CATEGORIES = [
    "heavy_winter_coat_gloves_hat",
    "winter_coat_scarf_gloves",
    "warm_jacket_layers",
    "light_jacket_or_sweater",
    "long_sleeves_light_layer",
    "t_shirt_comfortable",
    "light_breathable_clothing",
    "very_light_clothing_stay_hydrated",
]
