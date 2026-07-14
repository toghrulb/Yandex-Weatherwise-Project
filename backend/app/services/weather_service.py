"""
Business logic — rule-based recommendation engine.

When ML models are ready, this module will load .pkl/.joblib files
and use them for inference. Until then, uses deterministic rules
derived from the training data patterns.
"""
from __future__ import annotations

from backend.app.core.config import CLOTHING_CATEGORIES, STATIONS
from backend.app.services.data_loader import DataStore


# ═══════════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK PREDICTIONS  (replaced by ML models later)
# ═══════════════════════════════════════════════════════════════════════

def predict_clothing(feels_like: float) -> str:
    """Map feels-like temperature to clothing category."""
    if feels_like <= -15:
        return CLOTHING_CATEGORIES[0]   # heavy_winter_coat_gloves_hat
    elif feels_like <= -5:
        return CLOTHING_CATEGORIES[1]   # winter_coat_scarf_gloves
    elif feels_like <= 3:
        return CLOTHING_CATEGORIES[2]   # warm_jacket_layers
    elif feels_like <= 10:
        return CLOTHING_CATEGORIES[3]   # light_jacket_or_sweater
    elif feels_like <= 17:
        return CLOTHING_CATEGORIES[4]   # long_sleeves_light_layer
    elif feels_like <= 24:
        return CLOTHING_CATEGORIES[5]   # t_shirt_comfortable
    elif feels_like <= 30:
        return CLOTHING_CATEGORIES[6]   # light_breathable_clothing
    else:
        return CLOTHING_CATEGORIES[7]   # very_light_clothing_stay_hydrated


def predict_umbrella(precip_mm: float, condition: str) -> bool:
    """Determine if umbrella is needed."""
    return precip_mm > 0.1 or condition in ("rain", "heavy_rain", "drizzle", "thunderstorm")


def predict_suitability(weather: dict) -> float:
    """Compute outdoor suitability score 0-10."""
    score = 10.0

    # Temperature penalty
    temp = weather["feels_like_c"]
    if temp < -10:
        score -= 4
    elif temp < 0:
        score -= 2.5
    elif temp < 5:
        score -= 1.5
    elif temp > 35:
        score -= 3
    elif temp > 30:
        score -= 1.5

    # Precipitation penalty
    precip = weather["precipitation_mm"]
    if precip > 5:
        score -= 3
    elif precip > 1:
        score -= 2
    elif precip > 0:
        score -= 1

    # Wind penalty
    wind = weather["wind_speed_kmh"]
    if wind > 40:
        score -= 3
    elif wind > 25:
        score -= 1.5
    elif wind > 15:
        score -= 0.5

    # Thunderstorm
    if weather.get("is_thunderstorm"):
        score -= 4

    # UV penalty
    uv = weather.get("uv_index", 0)
    if uv > 8:
        score -= 1

    return max(0.0, min(10.0, round(score, 1)))


def generate_headline(condition: str, umbrella: bool, feels_like: float) -> str:
    """Generate a short condition summary."""
    headlines = {
        "clear": "Clear skies",
        "partly_cloudy": "Partly cloudy",
        "cloudy": "Overcast skies",
        "rain": "Rainy conditions",
        "heavy_rain": "Heavy rainfall",
        "drizzle": "Light drizzle",
        "snow": "Snowfall",
        "heavy_snow": "Heavy snowfall",
        "fog": "Foggy conditions",
        "thunderstorm": "Thunderstorm warning",
        "windy": "Windy conditions",
        "hail": "Hail warning",
    }
    headline = headlines.get(condition, "Current conditions")
    if umbrella:
        headline += " — take an umbrella"
    if feels_like < -10:
        headline += " — dangerously cold"
    elif feels_like > 35:
        headline += " — extreme heat"
    return headline


def generate_recommendation_text(weather: dict, clothing: str, umbrella: bool) -> str:
    """Generate 1-3 sentence actionable advice."""
    parts = []
    condition = weather["weather_condition"]
    feels = weather["feels_like_c"]
    wind = weather["wind_speed_kmh"]
    uv = weather.get("uv_index", 0)
    precip = weather["precipitation_mm"]

    # Main condition advice
    if condition in ("rain", "heavy_rain", "drizzle"):
        if precip > 5:
            parts.append("Heavy rain expected — consider staying indoors or carry waterproof gear.")
        else:
            parts.append("Light rain possible — an umbrella would be handy.")
    elif condition in ("snow", "heavy_snow"):
        parts.append("Roads may be slippery — allow extra travel time.")
    elif condition == "thunderstorm":
        parts.append("Thunderstorm activity — avoid open areas and stay safe.")
    elif condition == "fog":
        parts.append("Reduced visibility — drive carefully.")
    elif condition == "clear":
        parts.append("Beautiful clear conditions outside.")

    # Clothing
    clothing_display = clothing.replace("_", " ")
    parts.append(f"Wear {clothing_display} (feels like {feels:.0f}°C).")

    # Wind chill / heat warnings
    if feels < -10:
        parts.append("Dangerously cold wind chill — limit time outdoors.")
    elif feels > 35:
        parts.append("Heat advisory — stay hydrated and seek shade.")
    elif wind > 30:
        parts.append(f"Strong winds at {wind:.0f} km/h — secure loose items.")

    # UV
    if uv >= 8:
        parts.append(f"Very high UV index ({uv:.0f}) — apply sunscreen.")
    elif uv >= 6:
        parts.append(f"High UV ({uv:.0f}) — wear sunglasses and sunscreen.")

    return " ".join(parts)


def predict_activity_suitability(weather: dict, activity: str) -> dict:
    """
    Generate activity-specific recommendation.
    Returns dict with: activity_suitability_score, go_or_no, activity_advice
    """
    general = predict_suitability(weather)
    score = general
    advice = "Conditions are suitable."
    condition = weather["weather_condition"]
    wind = weather["wind_speed_kmh"]
    precip = weather["precipitation_mm"]
    temp = weather["feels_like_c"]

    if activity == "cycling":
        if wind > 30:
            score -= 3
            advice = "Strong gusts make cycling uncomfortable today."
        if precip > 1:
            score -= 2
            advice = "Wet roads — unsafe for cycling."
        if condition in ("snow", "heavy_snow", "thunderstorm"):
            score -= 4
            advice = "Unsafe for cycling."
    elif activity == "running":
        if temp > 30:
            score -= 2
            advice = "Too hot for running — risk of heat exhaustion."
        if precip > 3:
            score -= 2
            advice = "Heavy rain — consider indoor exercise."
        if condition == "thunderstorm":
            score -= 4
            advice = "Thunderstorm — do not run outdoors."
    elif activity == "walking":
        if precip > 5:
            score -= 2
            advice = "Heavy rain — consider postponing your walk."
        if temp < -10:
            score -= 2
            advice = "Dangerously cold — keep walks very short."
    elif activity == "picnic":
        if precip > 0 or condition in ("rain", "heavy_rain", "drizzle", "snow", "thunderstorm"):
            score -= 4
            advice = "Not ideal for a picnic — precipitation expected."
        elif wind > 25:
            score -= 2
            advice = "Picnic possible but bring a jacket."
    elif activity == "driving":
        if condition in ("heavy_rain", "heavy_snow", "thunderstorm", "fog"):
            score -= 3
            advice = "Poor driving conditions — reduce speed."
        elif weather.get("road_surface") in ("icy", "snow_covered"):
            score -= 3
            advice = "Road surfaces slippery — drive with caution."
    elif activity == "outdoor_work":
        if condition == "thunderstorm":
            score -= 5
            advice = "Stop outdoor work — thunderstorm danger."
        if temp > 35:
            score -= 3
            advice = "Heat advisory — take frequent breaks and hydrate."
        if temp < -10:
            score -= 3
            advice = "Extreme cold — limit outdoor exposure."
    elif activity == "sports":
        if condition in ("thunderstorm", "heavy_rain", "heavy_snow"):
            score -= 5
            advice = "Unsafe for outdoor sports."
        if wind > 35:
            score -= 3
            advice = "High winds — move sports indoors."
    elif activity == "commute":
        if condition in ("heavy_snow", "thunderstorm"):
            score -= 3
            advice = "Expect delays — give yourself extra time."
        elif condition in ("rain", "fog"):
            score -= 1
            advice = "Minor delays possible — carry an umbrella."

    score = max(0.0, min(10.0, round(score, 1)))
    go = score >= 4

    return {
        "activity_suitability_score": score,
        "go_or_no": go,
        "activity_advice": advice,
    }


# ═══════════════════════════════════════════════════════════════════════
#  HISTORICAL DATA QUERIES  (from CSV DataStore)
# ═══════════════════════════════════════════════════════════════════════

def get_daily_summary(station_id: str, date_str: str | None = None) -> dict | None:
    """Fetch daily summary from CSV data store."""
    df = DataStore.daily
    if df is None:
        return None

    mask = df["station_id"] == station_id
    if date_str:
        mask = mask & (df["date"].dt.strftime("%Y-%m-%d") == date_str)

    rows = df[mask].sort_values("date", ascending=False)
    if rows.empty:
        return None

    row = rows.iloc[0]
    return row.to_dict()


def get_forecast_accuracy(station_id: str) -> dict | None:
    """Compute aggregated forecast accuracy stats from CSV data."""
    df = DataStore.forecasts
    if df is None:
        return None

    rows = df[df["station_id"] == station_id]
    if rows.empty:
        return None

    result = {
        "station_id": station_id,
        "total_forecasts": len(rows),
        "umbrella_accuracy_pct": round(rows["umbrella_correct"].mean() * 100, 1),
        "clothing_accuracy_pct": round(rows["clothing_correct"].mean() * 100, 1),
        "condition_accuracy_pct": round(rows["condition_correct"].mean() * 100, 1),
        "avg_temp_error_c": round(rows["temp_error_c"].abs().mean(), 2),
        "avg_precip_error_mm": round(rows["precip_error_mm"].abs().mean(), 2),
        "avg_wind_error_kmh": round(rows["wind_error_kmh"].abs().mean(), 2),
        "by_lead_time": [],
    }

    for lt in sorted(rows["lead_time_hours"].unique()):
        grp = rows[rows["lead_time_hours"] == lt]
        result["by_lead_time"].append({
            "lead_time_hours": int(lt),
            "count": len(grp),
            "umbrella_accuracy_pct": round(grp["umbrella_correct"].mean() * 100, 1),
            "clothing_accuracy_pct": round(grp["clothing_correct"].mean() * 100, 1),
            "condition_accuracy_pct": round(grp["condition_correct"].mean() * 100, 1),
        })

    return result
