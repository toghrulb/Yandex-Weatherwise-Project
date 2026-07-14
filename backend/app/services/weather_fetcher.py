"""
Real-time weather data fetcher using the Open-Meteo API.
Transforms API responses into the same feature schema as the training data.
"""
from __future__ import annotations

import httpx

from backend.app.core.config import (
    OPEN_METEO_BASE,
    OPEN_METEO_CURRENT_PARAMS,
    OPEN_METEO_HOURLY_PARAMS,
    STATIONS,
    WMO_TO_CONDITION,
)


async def fetch_current_weather(station_id: str) -> dict:
    """
    Fetch current + today's hourly forecast from Open-Meteo for a station.
    Returns a dict in the same schema as hourly_observations.csv columns.
    """
    station = STATIONS[station_id]
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "current": OPEN_METEO_CURRENT_PARAMS,
        "hourly": OPEN_METEO_HOURLY_PARAMS,
        "timezone": "auto",
        "forecast_days": 1,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    current = data["current"]
    return _transform_current(current, station_id)


async def fetch_hourly_forecast(station_id: str, days: int = 1) -> list[dict]:
    """
    Fetch hourly forecast for N days.
    Returns a list of dicts, one per hour, in the training-data schema.
    """
    station = STATIONS[station_id]
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": OPEN_METEO_HOURLY_PARAMS,
        "timezone": "auto",
        "forecast_days": days,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    results = []
    for i in range(len(hourly["time"])):
        row = _transform_hourly_row(hourly, i, station_id)
        results.append(row)
    return results


# ── Internal helpers ──────────────────────────────────────────────────

def _derive_precip_type(rain: float, snowfall: float) -> str:
    if snowfall and snowfall > 0:
        return "snow"
    if rain and rain > 0:
        return "rain"
    return "none"


def _derive_road_surface(temp: float, precip: float, precip_type: str) -> str:
    if precip <= 0:
        return "dry"
    if precip_type == "snow":
        return "snow_covered" if temp <= 0 else "wet"
    if temp <= 0:
        return "icy"
    return "wet"


def _wmo_to_condition(code: int | None) -> str:
    if code is None:
        return "clear"
    return WMO_TO_CONDITION.get(code, "clear")


def _transform_current(current: dict, station_id: str) -> dict:
    """Map Open-Meteo current response → training-data schema."""
    wmo = current.get("weather_code")
    condition = _wmo_to_condition(wmo)
    rain_val = current.get("rain", 0) or 0
    snow_val = current.get("snowfall", 0) or 0
    precip_val = current.get("precipitation", 0) or 0
    precip_type = _derive_precip_type(rain_val, snow_val)
    temp = current.get("temperature_2m", 0)

    return {
        "station_id": station_id,
        "timestamp": current.get("time", ""),
        "temperature_c": temp,
        "feels_like_c": current.get("apparent_temperature", temp),
        "dew_point_c": current.get("dew_point_2m", 0),
        "humidity_pct": current.get("relative_humidity_2m", 0),
        "pressure_hpa": current.get("surface_pressure", 0),
        "wind_speed_kmh": current.get("wind_speed_10m", 0),
        "wind_direction_deg": current.get("wind_direction_10m", 0),
        "wind_gust_kmh": current.get("wind_gusts_10m", 0),
        "precipitation_mm": precip_val,
        "precipitation_type": precip_type,
        "cloud_cover_pct": current.get("cloud_cover", 0),
        "visibility_km": (current.get("visibility", 10000) or 10000) / 1000,
        "uv_index": current.get("uv_index", 0),
        "weather_condition": condition,
        "is_thunderstorm": wmo in (95, 96, 99),
        "is_day": bool(current.get("is_day", 1)),
        "road_surface": _derive_road_surface(temp, precip_val, precip_type),
    }


def _transform_hourly_row(hourly: dict, idx: int, station_id: str) -> dict:
    """Map one row of Open-Meteo hourly array → training-data schema."""
    wmo = hourly["weather_code"][idx] if hourly["weather_code"][idx] is not None else 0
    condition = _wmo_to_condition(wmo)
    rain_val = (hourly.get("rain", [0] * (idx + 1))[idx]) or 0
    snow_val = (hourly.get("snowfall", [0] * (idx + 1))[idx]) or 0
    precip_val = (hourly.get("precipitation", [0] * (idx + 1))[idx]) or 0
    precip_type = _derive_precip_type(rain_val, snow_val)
    temp = hourly["temperature_2m"][idx]
    vis_raw = hourly.get("visibility", [10000] * (idx + 1))[idx] or 10000

    return {
        "station_id": station_id,
        "timestamp": hourly["time"][idx],
        "temperature_c": temp,
        "feels_like_c": hourly["apparent_temperature"][idx],
        "dew_point_c": hourly.get("dew_point_2m", [0] * (idx + 1))[idx],
        "humidity_pct": hourly["relative_humidity_2m"][idx],
        "pressure_hpa": hourly.get("surface_pressure", [0] * (idx + 1))[idx],
        "wind_speed_kmh": hourly["wind_speed_10m"][idx],
        "wind_direction_deg": hourly["wind_direction_10m"][idx],
        "wind_gust_kmh": hourly["wind_gusts_10m"][idx],
        "precipitation_mm": precip_val,
        "precipitation_type": precip_type,
        "cloud_cover_pct": hourly["cloud_cover"][idx],
        "visibility_km": vis_raw / 1000,
        "uv_index": hourly.get("uv_index", [0] * (idx + 1))[idx],
        "weather_condition": condition,
        "is_thunderstorm": wmo in (95, 96, 99),
        "is_day": bool(hourly.get("is_day", [1] * (idx + 1))[idx]),
        "road_surface": _derive_road_surface(temp, precip_val, precip_type),
    }
