"""
API routes for the WeatherWise backend.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.app.core.config import STATIONS
from backend.app.schemas.weather import (
    ActivityCardResponse,
    DailySummaryResponse,
    ForecastAccuracyResponse,
    HeroCardResponse,
    RawWeather,
    StationInfo,
    TimeSlot,
)
from backend.app.services.inference import inference_service
from backend.app.services.llm_text import llm_text_service
from backend.app.services.weather_fetcher import fetch_current_weather, fetch_hourly_forecast
from backend.app.services.weather_service import get_forecast_accuracy
from backend.app.core.translations import tl


router = APIRouter(prefix="/api")

ACTIVITY_TYPES = [
    "walking",
    "running",
    "cycling",
    "driving",
    "outdoor_work",
    "picnic",
    "sports",
    "commute",
]


def _headline_from_weather(condition: str, umbrella: bool, feels_like: float, is_day: bool = True, lang: str = "en") -> str:
    label_map = {
        "clear": "Clear skies" if is_day else "Clear night",
        "partly_cloudy": "Partly cloudy" if is_day else "Partly cloudy night",
        "cloudy": "Overcast skies",
        "rain": "Rain expected",
        "heavy_rain": "Heavy rain expected",
        "drizzle": "Light drizzle",
        "snow": "Snow expected",
        "heavy_snow": "Heavy snow expected",
        "fog": "Foggy conditions",
        "thunderstorm": "Thunderstorm risk",
    }
    headline = tl(label_map.get(condition, "Current conditions"), lang)
    if umbrella:
        headline += tl(" - umbrella recommended", lang)
    if feels_like <= -10:
        headline += tl(" - very cold", lang)
    elif feels_like >= 35:
        headline += tl(" - very hot", lang)
    return headline


def _safe_hour(timestamp: str) -> int:
    try:
        return datetime.fromisoformat(timestamp).hour
    except ValueError:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed.hour


def _build_daily_feature_payload(station_id: str, hourly: list[dict]) -> dict:
    station = STATIONS[station_id]
    first_ts = datetime.fromisoformat(hourly[0]["timestamp"].replace("Z", "+00:00"))

    temperatures = [float(h.get("temperature_c", 0.0) or 0.0) for h in hourly]
    feels = [float(h.get("feels_like_c", 0.0) or 0.0) for h in hourly]
    humidity = [float(h.get("humidity_pct", 0.0) or 0.0) for h in hourly]
    winds = [float(h.get("wind_speed_kmh", 0.0) or 0.0) for h in hourly]
    gusts = [float(h.get("wind_gust_kmh", 0.0) or 0.0) for h in hourly]
    uv = [float(h.get("uv_index", 0.0) or 0.0) for h in hourly]
    precip = [float(h.get("precipitation_mm", 0.0) or 0.0) for h in hourly]

    conditions = [str(h.get("weather_condition", "clear")) for h in hourly]
    dominant_condition = max(set(conditions), key=conditions.count)

    rain_hours = sum(
        1 for h in hourly if str(h.get("precipitation_type", "")).lower() == "rain" or float(h.get("precipitation_mm", 0.0) or 0.0) > 0
    )
    snow_hours = sum(1 for h in hourly if str(h.get("precipitation_type", "")).lower() == "snow")

    month = int(first_ts.month)
    day_of_week = int(first_ts.weekday())

    return {
        "station_id": station_id,
        "climate_zone": station["climate_zone"],
        "date": first_ts.date().isoformat(),
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week in (5, 6)),
        "season": inference_service.enrich_weather_payload({"month": month}).get("season", "unknown"),
        "temp_min_c": min(temperatures),
        "temp_max_c": max(temperatures),
        "temp_avg_c": sum(temperatures) / len(temperatures),
        "feels_like_min_c": min(feels),
        "feels_like_max_c": max(feels),
        "total_precipitation_mm": sum(precip),
        "snow_hours": int(snow_hours),
        "rain_hours": int(rain_hours),
        "max_wind_kmh": max(winds),
        "max_gust_kmh": max(gusts),
        "avg_humidity_pct": sum(humidity) / len(humidity),
        "max_uv_index": max(uv),
        "dominant_condition": dominant_condition,
        "thunderstorm_occurred": int(any(bool(h.get("is_thunderstorm")) for h in hourly)),
    }


def _daily_text(
    dominant_condition: str,
    umbrella_recommended: bool,
    best_outdoor_hour: int,
    suitability: float,
    lang: str = "en",
) -> str:
    umbrella_text = tl("Carry an umbrella.", lang) if umbrella_recommended else tl("Umbrella is likely not needed.", lang)
    cond = tl(_headline_from_weather(dominant_condition, False, 15, True, lang), lang) # Best effort localization
    # Condition actually returns "Rain Expected", "Clear Skies" etc. We want "Clear Skies day expected."
    # We will just concatenate cleanly.
    return (
        f"{cond}{tl(' day expected. ', lang)}"
        f"{tl('Best outdoor hour around ', lang)}{best_outdoor_hour}{tl(':00. ', lang)}"
        f"{tl('Expected daily outdoor score ', lang)}{suitability:.1f}{tl('/10. ', lang)}"
        f"{umbrella_text}"
    )


def _best_hour_from_hourly_scores(hourly_scores: list[tuple[dict, float]]) -> int:
    """
    Choose best outdoor hour from hourly continuous suitability.
    Tie-break toward midday for more user-friendly recommendations.
    """
    if not hourly_scores:
        return 12

    daytime = [(row, score) for row, score in hourly_scores if 6 <= _safe_hour(str(row["timestamp"])) <= 21]
    candidates = daytime or hourly_scores
    max_score = max(score for _, score in candidates)
    top = [(row, score) for row, score in candidates if abs(score - max_score) <= 1e-9]
    top.sort(key=lambda rs: abs(_safe_hour(str(rs[0]["timestamp"])) - 14))
    return _safe_hour(str(top[0][0]["timestamp"]))


def _to_station_info(station_id: str) -> StationInfo:
    station = STATIONS[station_id]
    return StationInfo(
        station_id=station_id,
        name=station["name"],
        latitude=station["lat"],
        longitude=station["lon"],
        elevation_m=station["elevation_m"],
        climate_zone=station["climate_zone"],
    )


@router.get("/stations", response_model=list[StationInfo])
async def list_stations():
    return [_to_station_info(sid) for sid in STATIONS]


@router.get("/hero/{station_id}", response_model=HeroCardResponse)
async def get_hero(station_id: str, lang: str = "en"):
    if station_id not in STATIONS:
        raise HTTPException(status_code=404, detail="Station not found")

    weather = await fetch_current_weather(station_id)
    station = STATIONS[station_id]
    payload = inference_service.enrich_weather_payload(weather, climate_zone=station["climate_zone"])
    try:
        pred = inference_service.predict(payload, lang=lang)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    umbrella_needed = bool(pred["umbrella_needed"])
    headline = _headline_from_weather(
        condition=str(payload.get("weather_condition", "clear")),
        umbrella=umbrella_needed,
        feels_like=float(payload.get("feels_like_c", payload.get("temperature_c", 0.0)) or 0.0),
        is_day=bool(payload.get("is_day", True)),
        lang=lang,
    )

    return HeroCardResponse(
        station=_to_station_info(station_id),
        timestamp=str(payload.get("timestamp", "")),
        raw=RawWeather(
            temperature_c=float(payload.get("temperature_c", 0.0) or 0.0),
            feels_like_c=float(payload.get("feels_like_c", 0.0) or 0.0),
            dew_point_c=float(payload.get("dew_point_c", 0.0) or 0.0),
            humidity_pct=float(payload.get("humidity_pct", 0.0) or 0.0),
            pressure_hpa=float(payload.get("pressure_hpa", 0.0) or 0.0),
            wind_speed_kmh=float(payload.get("wind_speed_kmh", 0.0) or 0.0),
            wind_direction_deg=float(payload.get("wind_direction_deg", 0.0) or 0.0),
            wind_gust_kmh=float(payload.get("wind_gust_kmh", 0.0) or 0.0),
            precipitation_mm=float(payload.get("precipitation_mm", 0.0) or 0.0),
            precipitation_type=str(payload.get("precipitation_type", "none")),
            cloud_cover_pct=float(payload.get("cloud_cover_pct", 0.0) or 0.0),
            visibility_km=float(payload.get("visibility_km", 0.0) or 0.0),
            uv_index=float(payload.get("uv_index", 0.0) or 0.0),
            weather_condition=str(payload.get("weather_condition", "clear")),
            is_thunderstorm=bool(payload.get("is_thunderstorm", False)),
            is_day=bool(payload.get("is_day", True)),
        ),
        outdoor_suitability_score=float(pred["outdoor_suitability_score"]),
        umbrella_needed=umbrella_needed,
        clothing_recommendation=str(pred["clothing_recommendation"]),
        recommendation_headline=headline,
        recommendation_text=str(pred["recommendation_text"]),
    )


@router.get("/daily/{station_id}", response_model=DailySummaryResponse)
async def get_daily(station_id: str, date: str | None = None, lang: str = "en"):
    if station_id not in STATIONS:
        raise HTTPException(status_code=404, detail="Station not found")

    hourly = await fetch_hourly_forecast(station_id, days=1)
    if not hourly:
        raise HTTPException(status_code=404, detail="No data available")

    station = STATIONS[station_id]
    enriched_hourly = [
        inference_service.enrich_weather_payload(row, climate_zone=station["climate_zone"])
        for row in hourly
    ]
    if date:
        enriched_hourly = [h for h in enriched_hourly if str(h.get("timestamp", "")).startswith(date)]
    if not enriched_hourly:
        raise HTTPException(status_code=404, detail="No hourly forecast for requested date")

    daily_payload = _build_daily_feature_payload(station_id, enriched_hourly)
    try:
        daily_pred = inference_service.predict_daily(daily_payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    hourly_scores: list[tuple[dict, float]] = []
    try:
        for row in enriched_hourly:
            row_score = inference_service.predict_hourly_suitability_raw(row)
            hourly_scores.append((row, float(row_score)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    best_outdoor_hour = _best_hour_from_hourly_scores(hourly_scores)

    slot_defs = {
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 23),
    }
    clothing_by_slot = {
        "morning": str(daily_pred["clothing_morning"]),
        "afternoon": str(daily_pred["clothing_afternoon"]),
        "evening": str(daily_pred["clothing_evening"]),
    }
    time_strip: list[TimeSlot] = []
    for period, hour_range in slot_defs.items():
        slot_scores = [(r, s) for (r, s) in hourly_scores if _safe_hour(str(r["timestamp"])) in hour_range]
        if not slot_scores:
            time_strip.append(
                TimeSlot(
                    period=period,
                    clothing=clothing_by_slot[period],
                    tip=tl("No hourly data available.", lang),
                    go_indicator="orange",
                )
            )
            continue

        avg_slot_score = sum(score for _, score in slot_scores) / len(slot_scores)
        representative = slot_scores[len(slot_scores) // 2][0]
        condition = str(representative.get("weather_condition", "clear")).replace("_", " ")
        # Basic condition translation
        translated_cond = tl(_headline_from_weather(condition, False, 15, True, lang), lang)
        tip = f"{translated_cond}{tl(' around this period.', lang)}"
        if avg_slot_score >= 6.0:
            go_indicator = "green"
        elif avg_slot_score >= 4.0:
            go_indicator = "orange"
        else:
            go_indicator = "red"

        time_strip.append(
            TimeSlot(
                period=period,
                clothing=clothing_by_slot[period],
                tip=tip,
                go_indicator=go_indicator,
            )
        )

    first_ts = datetime.fromisoformat(str(enriched_hourly[0]["timestamp"]).replace("Z", "+00:00"))
    summary_text = _daily_text(
        dominant_condition=str(daily_payload["dominant_condition"]),
        umbrella_recommended=bool(daily_pred["umbrella_recommended"]),
        best_outdoor_hour=best_outdoor_hour,
        suitability=float(daily_pred["avg_outdoor_suitability"]),
        lang=lang,
    )

    return DailySummaryResponse(
        station_id=station_id,
        date=first_ts.date().isoformat(),
        temp_min_c=round(float(daily_payload["temp_min_c"]), 1),
        temp_max_c=round(float(daily_payload["temp_max_c"]), 1),
        temp_avg_c=round(float(daily_payload["temp_avg_c"]), 1),
        total_precipitation_mm=round(float(daily_payload["total_precipitation_mm"]), 1),
        dominant_condition=str(daily_payload["dominant_condition"]),
        avg_outdoor_suitability=round(float(daily_pred["avg_outdoor_suitability"]), 1),
        best_outdoor_hour=best_outdoor_hour,
        umbrella_recommended=bool(daily_pred["umbrella_recommended"]),
        day_summary_text=summary_text,
        time_strip=time_strip,
    )


@router.get("/activities/{station_id}", response_model=list[ActivityCardResponse])
async def get_activities(station_id: str, lang: str = "en"):
    if station_id not in STATIONS:
        raise HTTPException(status_code=404, detail="Station not found")

    weather = await fetch_current_weather(station_id)
    station = STATIONS[station_id]
    payload = inference_service.enrich_weather_payload(weather, climate_zone=station["climate_zone"])

    try:
        predictions = [
            inference_service.predict_activity(payload, activity, lang=lang)
            for activity in ACTIVITY_TYPES
        ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    llm_advices = llm_text_service.generate_activity_advices(payload=payload, predictions=predictions, lang=lang)
    for pred in predictions:
        activity_name = str(pred.get("activity_type", ""))
        if activity_name in llm_advices:
            pred["activity_advice"] = llm_advices[activity_name]

    results = [ActivityCardResponse(**pred) for pred in predictions]
    return results


@router.get("/forecast-accuracy/{station_id}", response_model=ForecastAccuracyResponse | dict)
async def get_forecast_acc(station_id: str):
    if station_id not in STATIONS:
        raise HTTPException(status_code=404, detail="Station not found")

    result = get_forecast_accuracy(station_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No forecast data available")
    return result
