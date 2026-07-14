from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.src.features.build_activity_features import build_activity_feature_frame
from ml.src.features.build_daily_features import build_daily_feature_frame
from ml.src.features.build_features import build_feature_frame
from ml.src.utils.paths import (
    ACTIVITY_GO_NO_MODEL_PATH,
    ACTIVITY_SUITABILITY_MODEL_PATH,
    CLOTHING_MODEL_PATH,
    DAILY_CLOTHING_AFTERNOON_MODEL_PATH,
    DAILY_CLOTHING_EVENING_MODEL_PATH,
    DAILY_CLOTHING_MORNING_MODEL_PATH,
    DAILY_SUITABILITY_MODEL_PATH,
    DAILY_UMBRELLA_MODEL_PATH,
    SUITABILITY_MODEL_PATH,
    UMBRELLA_MODEL_PATH,
)
from backend.app.services.llm_text import llm_text_service
from backend.app.core.translations import tl


def _season_from_month(month: int | None) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "unknown"


ACTIVITY_GO_THRESHOLD = 6.0


class InferenceService:
    def __init__(self) -> None:
        self._umbrella_model = None
        self._clothing_model = None
        self._suitability_model = None
        self._activity_go_no_model = None
        self._activity_suitability_model = None
        self._daily_umbrella_model = None
        self._daily_clothing_morning_model = None
        self._daily_clothing_afternoon_model = None
        self._daily_clothing_evening_model = None
        self._daily_suitability_model = None

    @staticmethod
    def enrich_weather_payload(payload: dict[str, Any], climate_zone: str | None = None) -> dict[str, Any]:
        enriched = dict(payload)

        ts_value = enriched.get("timestamp")
        dt: datetime | None = None
        if ts_value:
            parsed = pd.to_datetime(ts_value, errors="coerce", utc=False)
            if not pd.isna(parsed):
                dt = parsed.to_pydatetime()

        if dt is not None:
            enriched.setdefault("hour_of_day", int(dt.hour))
            enriched.setdefault("month", int(dt.month))
            enriched.setdefault("day_of_week", int(dt.weekday()))
        else:
            month_value = pd.to_numeric(enriched.get("month"), errors="coerce")
            if not pd.isna(month_value):
                enriched["month"] = int(month_value)

        day_of_week = pd.to_numeric(enriched.get("day_of_week"), errors="coerce")
        if not pd.isna(day_of_week):
            day_int = int(day_of_week)
            enriched["day_of_week"] = day_int
            enriched.setdefault("is_weekend", int(day_int in (5, 6)))

        month_int = enriched.get("month")
        if enriched.get("season") in (None, "", "unknown"):
            enriched["season"] = _season_from_month(month_int if isinstance(month_int, int) else None)

        if climate_zone and not enriched.get("climate_zone"):
            enriched["climate_zone"] = climate_zone

        if not enriched.get("precipitation_type"):
            condition = str(enriched.get("weather_condition", "")).lower()
            precip_mm = float(enriched.get("precipitation_mm", 0.0) or 0.0)
            if "snow" in condition:
                enriched["precipitation_type"] = "snow"
            elif precip_mm > 0:
                enriched["precipitation_type"] = "rain"
            else:
                enriched["precipitation_type"] = "none"

        return enriched

    @staticmethod
    def _load_optional_model(path):
        return joblib.load(path) if path.exists() else None

    @staticmethod
    def _patch_logistic_compat(model: Any) -> Any:
        """
        Patch cross-version logistic artifacts loaded through joblib.
        Older/newer sklearn versions can disagree on expected attributes.
        """
        candidate = model
        if hasattr(model, "named_steps"):
            candidate = model.named_steps.get("model", model)

        if isinstance(candidate, LogisticRegression) and not hasattr(candidate, "multi_class"):
            candidate.multi_class = "auto"
        return model

    def load_models(self) -> None:
        missing = [
            str(path)
            for path in (UMBRELLA_MODEL_PATH, CLOTHING_MODEL_PATH, SUITABILITY_MODEL_PATH)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Model artifacts are missing. Train models first via "
                "`python -m ml.src.models.train_pipeline`. Missing: " + ", ".join(missing)
            )

        self._umbrella_model = self._patch_logistic_compat(joblib.load(UMBRELLA_MODEL_PATH))
        self._clothing_model = self._patch_logistic_compat(joblib.load(CLOTHING_MODEL_PATH))
        self._suitability_model = self._patch_logistic_compat(joblib.load(SUITABILITY_MODEL_PATH))
        self._activity_go_no_model = self._patch_logistic_compat(self._load_optional_model(ACTIVITY_GO_NO_MODEL_PATH))
        self._activity_suitability_model = self._patch_logistic_compat(self._load_optional_model(ACTIVITY_SUITABILITY_MODEL_PATH))
        self._daily_umbrella_model = self._patch_logistic_compat(self._load_optional_model(DAILY_UMBRELLA_MODEL_PATH))
        self._daily_clothing_morning_model = self._patch_logistic_compat(self._load_optional_model(DAILY_CLOTHING_MORNING_MODEL_PATH))
        self._daily_clothing_afternoon_model = self._patch_logistic_compat(self._load_optional_model(DAILY_CLOTHING_AFTERNOON_MODEL_PATH))
        self._daily_clothing_evening_model = self._patch_logistic_compat(self._load_optional_model(DAILY_CLOTHING_EVENING_MODEL_PATH))
        self._daily_suitability_model = self._patch_logistic_compat(self._load_optional_model(DAILY_SUITABILITY_MODEL_PATH))

    def _ensure_loaded(self) -> None:
        if self._umbrella_model is None or self._clothing_model is None or self._suitability_model is None:
            self.load_models()

    @staticmethod
    def _build_recommendation_text(payload: dict[str, Any], umbrella: int, suitability: float, lang: str = "en") -> str:
        weather_condition = payload.get("weather_condition")
        temp = payload.get("feels_like_c") if payload.get("feels_like_c") is not None else payload.get("temperature_c")
        parts: list[str] = []

        if weather_condition:
            parts.append(f"{tl('Condition: ', lang)}{tl(weather_condition.replace('_', ' ').capitalize(), lang)}.")
        if umbrella == 1:
            parts.append(tl("Take an umbrella.", lang))
        if suitability < 4:
            parts.append(tl("Outdoor plans are risky right now.", lang))
        elif suitability < 7:
            parts.append(tl("Conditions are moderate for outdoor activities.", lang))
        else:
            parts.append(tl("Good time for outdoor activities.", lang))
        if temp is not None:
            parts.append(f"{tl('Feels like around ', lang)}{float(temp):.1f}{tl('C.', lang)}")

        return " ".join(parts)

    @staticmethod
    def _build_activity_advice(activity_type: str, go_or_no: bool, score: float, payload: dict[str, Any], lang: str = "en") -> str:
        condition = str(payload.get("weather_condition", "")).replace("_", " ").strip()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        temp = payload.get("feels_like_c", payload.get("temperature_c"))
        temp_text = f"{float(temp):.0f}C" if temp is not None else tl("current conditions", lang)
        
        # We assume LLM overwrites this, but keep as fallback just in case
        if go_or_no:
            return (
                f"{activity_type.replace('_', ' ').title()} looks good now "
                f"(score {score:.1f}/10, feels like {temp_text})."
            )
        if precip > 0:
            return (
                f"Not ideal for {activity_type.replace('_', ' ')} right now "
                f"(precipitation and {condition or 'unstable weather'})."
            )
        return (
            f"{activity_type.replace('_', ' ').title()} is not recommended now "
            f"(score {score:.1f}/10)."
        )

    @staticmethod
    def _fallback_clothing(feels_like_c: float | None) -> str:
        categories = [
            "heavy_winter_coat_gloves_hat",
            "winter_coat_scarf_gloves",
            "warm_jacket_layers",
            "light_jacket_or_sweater",
            "long_sleeves_light_layer",
            "t_shirt_comfortable",
            "light_breathable_clothing",
            "very_light_clothing_stay_hydrated",
        ]
        if feels_like_c is None:
            return "light_jacket_or_sweater"
        if feels_like_c <= -15:
            return categories[0]
        if feels_like_c <= -5:
            return categories[1]
        if feels_like_c <= 3:
            return categories[2]
        if feels_like_c <= 10:
            return categories[3]
        if feels_like_c <= 17:
            return categories[4]
        if feels_like_c <= 24:
            return categories[5]
        if feels_like_c <= 30:
            return categories[6]
        return categories[7]

    @staticmethod
    def _distance_to_band(value: float, low: float, high: float) -> float:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    @classmethod
    def _activity_temperature_penalty(cls, activity_type: str, temp_c: float) -> float:
        """
        Penalize activity suitability when feels-like temperature is outside
        activity-specific comfort bands.
        """
        profiles: dict[str, dict[str, tuple[float, float] | float]] = {
            "picnic": {"ideal": (18.0, 26.0), "tolerable": (13.0, 30.0), "hard": (8.0, 34.0), "mult": 1.5},
            "running": {"ideal": (11.0, 19.0), "tolerable": (5.0, 25.0), "hard": (0.0, 31.0), "mult": 1.25},
            "cycling": {"ideal": (12.0, 23.0), "tolerable": (6.0, 28.0), "hard": (1.0, 33.0), "mult": 1.2},
            "sports": {"ideal": (10.0, 22.0), "tolerable": (4.0, 28.0), "hard": (-1.0, 33.0), "mult": 1.2},
            "outdoor_work": {"ideal": (8.0, 23.0), "tolerable": (2.0, 29.0), "hard": (-4.0, 34.0), "mult": 1.1},
            "walking": {"ideal": (10.0, 24.0), "tolerable": (3.0, 30.0), "hard": (-4.0, 35.0), "mult": 1.05},
            "commute": {"ideal": (-5.0, 30.0), "tolerable": (-12.0, 35.0), "hard": (-18.0, 40.0), "mult": 0.6},
            "driving": {"ideal": (-8.0, 30.0), "tolerable": (-15.0, 35.0), "hard": (-22.0, 42.0), "mult": 0.5},
        }
        default_profile: dict[str, tuple[float, float] | float] = {
            "ideal": (6.0, 24.0),
            "tolerable": (-2.0, 30.0),
            "hard": (-8.0, 35.0),
            "mult": 1.0,
        }
        profile = profiles.get(activity_type, default_profile)

        ideal_low, ideal_high = cast(tuple[float, float], profile["ideal"])
        tolerable_low, tolerable_high = cast(tuple[float, float], profile["tolerable"])
        hard_low, hard_high = cast(tuple[float, float], profile["hard"])
        multiplier = float(profile["mult"])

        if ideal_low <= temp_c <= ideal_high:
            return 0.0

        if tolerable_low <= temp_c <= tolerable_high:
            ideal_dist = cls._distance_to_band(temp_c, ideal_low, ideal_high)
            base_penalty = 0.45 + 0.12 * ideal_dist
            return base_penalty * multiplier

        if hard_low <= temp_c <= hard_high:
            tolerable_dist = cls._distance_to_band(temp_c, tolerable_low, tolerable_high)
            base_penalty = 1.2 + 0.16 * tolerable_dist
            return base_penalty * multiplier

        hard_dist = cls._distance_to_band(temp_c, hard_low, hard_high)
        base_penalty = 2.4 + 0.18 * hard_dist
        return base_penalty * multiplier

    @classmethod
    def _activity_score_cap(
        cls,
        activity_type: str,
        payload: dict[str, Any],
        base_score: float,
        activity_penalty: float,
    ) -> float:
        """
        Keep activity score aligned with overall conditions.
        Prevent unrealistically high activity scores when global outdoor comfort is moderate/low.
        """
        condition = str(payload.get("weather_condition", "")).lower()
        precip_type = str(payload.get("precipitation_type", "")).lower()
        road_surface = str(payload.get("road_surface", "")).lower()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        feels_like = payload.get("feels_like_c", payload.get("temperature_c", 15.0))
        temp = float(feels_like if feels_like is not None else 15.0)
        visibility = float(payload.get("visibility_km", 10.0) or 10.0)
        is_thunder = bool(payload.get("is_thunderstorm", False)) or ("thunderstorm" in condition)
        is_snow = ("snow" in condition) or (precip_type == "snow")
        is_rainy = ("rain" in condition) or ("drizzle" in condition) or (precip_type == "rain")

        margin_by_activity = {
            "walking": 0.6,
            "running": 0.4,
            "cycling": 0.3,
            "driving": 1.0,
            "outdoor_work": 0.4,
            "picnic": 0.2,
            "sports": 0.4,
            "commute": 0.8,
        }
        cap = min(10.0, base_score + margin_by_activity.get(activity_type, 0.5))
        cap = min(cap, 10.0 - 0.55 * max(activity_penalty, 0.0))

        if is_thunder:
            cap = min(cap, base_score + 0.2)

        if is_rainy or is_snow:
            if activity_type in {"driving", "commute"}:
                cap = min(cap, base_score + 0.6)
            else:
                cap = min(cap, base_score + 0.3)

        if road_surface in {"icy", "snow_covered"} and activity_type in {"driving", "cycling", "running"}:
            cap = min(cap, base_score + 0.15)

        if visibility < 5.0 and activity_type in {"cycling", "driving", "running"}:
            cap = min(cap, base_score + 0.2)

        if temp <= 8:
            cool_caps = {
                "walking": 8.1,
                "running": 7.8,
                "cycling": 7.5,
                "sports": 7.7,
                "outdoor_work": 7.8,
                "picnic": 7.2,
            }
            if activity_type in cool_caps:
                cap = min(cap, cool_caps[activity_type])

        if temp <= 4:
            cold_caps = {
                "walking": 7.5,
                "running": 7.0,
                "cycling": 6.7,
                "sports": 6.9,
                "outdoor_work": 7.0,
                "picnic": 6.5,
                "commute": 8.0,
                "driving": 8.2,
            }
            if activity_type in cold_caps:
                cap = min(cap, cold_caps[activity_type])

        if temp >= 32:
            heat_caps = {
                "walking": 7.8,
                "running": 6.8,
                "cycling": 7.0,
                "sports": 6.7,
                "outdoor_work": 6.5,
                "picnic": 6.8,
            }
            if activity_type in heat_caps:
                cap = min(cap, heat_caps[activity_type])

        if precip > 2.0 and activity_type in {"walking", "running", "cycling", "sports", "outdoor_work", "picnic"}:
            cap = min(cap, base_score + 0.2)

        return float(np.clip(cap, 0.0, 10.0))

    @staticmethod
    def _activity_weather_penalty(activity_type: str, payload: dict[str, Any]) -> float:
        """
        Rule-based activity sensitivity adjustment on top of ML activity score.
        Higher return value means stronger weather-related penalty.
        This layer encodes practical safety/comfort logic per activity.
        """
        condition = str(payload.get("weather_condition", "")).lower()
        precip_type = str(payload.get("precipitation_type", "")).lower()
        road_surface = str(payload.get("road_surface", "")).lower()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        wind = float(payload.get("wind_speed_kmh", 0.0) or 0.0)
        gust = float(payload.get("wind_gust_kmh", wind) or wind)
        visibility = float(payload.get("visibility_km", 10.0) or 10.0)
        uv = float(payload.get("uv_index", 0.0) or 0.0)
        feels_like = payload.get("feels_like_c", payload.get("temperature_c", 15.0))
        temp = float(feels_like if feels_like is not None else 15.0)
        is_thunder = bool(payload.get("is_thunderstorm", False)) or ("thunderstorm" in condition)
        is_snow = ("snow" in condition) or (precip_type == "snow")
        is_fog = "fog" in condition
        is_rainy = ("rain" in condition) or ("drizzle" in condition) or (precip_type == "rain")
        is_icy = road_surface == "icy" or (temp <= 0.5 and (is_rainy or is_snow))
        is_snow_covered = road_surface == "snow_covered" or (is_snow and precip > 0.0 and temp <= 0.5)

        precip_light = precip > 0.1
        precip_moderate = precip > 1.0
        precip_heavy = precip > 4.0 or ("heavy_rain" in condition)

        wind_breezy = wind > 18
        wind_windy = wind > 28
        wind_severe = wind > 40
        gusty = gust > 45

        vis_moderate = visibility < 8.0
        vis_low = visibility < 5.0
        vis_very_low = visibility < 2.5

        profiles: dict[str, dict[str, float]] = {
            "walking": {
                "rain": 0.9,
                "heavy_rain": 0.8,
                "snow": 1.0,
                "wet_surface": 0.3,
                "icy": 0.7,
                "snow_covered": 0.9,
                "wind": 0.5,
                "severe_wind": 0.7,
                "gust": 0.3,
                "fog": 0.2,
                "low_vis": 0.15,
                "very_low_vis": 0.25,
                "uv": 0.3,
                "uv_extreme": 0.5,
                "thunder": 1.5,
            },
            "running": {
                "rain": 1.2,
                "heavy_rain": 1.0,
                "snow": 1.3,
                "wet_surface": 0.3,
                "icy": 0.8,
                "snow_covered": 1.1,
                "wind": 0.6,
                "severe_wind": 0.9,
                "gust": 0.4,
                "fog": 0.3,
                "low_vis": 0.2,
                "very_low_vis": 0.3,
                "uv": 0.7,
                "uv_extreme": 0.9,
                "thunder": 1.9,
            },
            "cycling": {
                "rain": 1.6,
                "heavy_rain": 1.3,
                "snow": 1.8,
                "wet_surface": 0.6,
                "icy": 1.3,
                "snow_covered": 1.6,
                "wind": 1.0,
                "severe_wind": 1.2,
                "gust": 0.7,
                "fog": 0.6,
                "low_vis": 0.4,
                "very_low_vis": 0.6,
                "uv": 0.4,
                "uv_extreme": 0.6,
                "thunder": 2.2,
            },
            "driving": {
                "rain": 0.6,
                "heavy_rain": 1.0,
                "snow": 1.8,
                "wet_surface": 0.5,
                "icy": 1.7,
                "snow_covered": 1.9,
                "wind": 0.5,
                "severe_wind": 0.9,
                "gust": 0.4,
                "fog": 1.0,
                "low_vis": 0.9,
                "very_low_vis": 1.2,
                "uv": 0.0,
                "uv_extreme": 0.0,
                "thunder": 1.1,
            },
            "outdoor_work": {
                "rain": 1.3,
                "heavy_rain": 1.0,
                "snow": 1.4,
                "wet_surface": 0.5,
                "icy": 1.0,
                "snow_covered": 1.2,
                "wind": 0.9,
                "severe_wind": 1.1,
                "gust": 0.6,
                "fog": 0.3,
                "low_vis": 0.2,
                "very_low_vis": 0.3,
                "uv": 0.9,
                "uv_extreme": 1.2,
                "thunder": 2.3,
            },
            "picnic": {
                "rain": 1.9,
                "heavy_rain": 1.5,
                "snow": 2.2,
                "wet_surface": 0.8,
                "icy": 1.6,
                "snow_covered": 2.0,
                "wind": 0.8,
                "severe_wind": 1.0,
                "gust": 0.7,
                "fog": 0.4,
                "low_vis": 0.2,
                "very_low_vis": 0.4,
                "uv": 0.6,
                "uv_extreme": 0.8,
                "thunder": 2.4,
            },
            "sports": {
                "rain": 1.5,
                "heavy_rain": 1.2,
                "snow": 1.5,
                "wet_surface": 0.4,
                "icy": 0.9,
                "snow_covered": 1.3,
                "wind": 0.8,
                "severe_wind": 1.1,
                "gust": 0.6,
                "fog": 0.3,
                "low_vis": 0.2,
                "very_low_vis": 0.3,
                "uv": 0.8,
                "uv_extreme": 1.0,
                "thunder": 2.1,
            },
            "commute": {
                "rain": 0.7,
                "heavy_rain": 0.6,
                "snow": 1.2,
                "wet_surface": 0.4,
                "icy": 0.9,
                "snow_covered": 1.0,
                "wind": 0.4,
                "severe_wind": 0.6,
                "gust": 0.3,
                "fog": 0.4,
                "low_vis": 0.3,
                "very_low_vis": 0.5,
                "uv": 0.2,
                "uv_extreme": 0.3,
                "thunder": 1.0,
            },
        }
        default_profile = profiles["walking"]
        profile = profiles.get(activity_type, default_profile)

        penalty = 0.0
        penalty += InferenceService._activity_temperature_penalty(activity_type=activity_type, temp_c=temp)

        if is_rainy and precip_light:
            penalty += profile["rain"]
            if precip_moderate:
                penalty += 0.55 * profile["rain"]
            if precip_heavy:
                penalty += profile["heavy_rain"]

        if is_snow:
            penalty += profile["snow"]
            if precip_moderate:
                penalty += 0.6 * profile["snow"]

        if road_surface == "wet":
            penalty += profile["wet_surface"]
        if is_icy:
            penalty += profile["icy"]
        if is_snow_covered:
            penalty += profile["snow_covered"]

        if wind_breezy:
            penalty += profile["wind"]
        if wind_windy:
            penalty += 0.6 * profile["wind"]
        if wind_severe:
            penalty += profile["severe_wind"]
        if gusty:
            penalty += profile["gust"]

        if is_fog:
            penalty += profile["fog"]
        if vis_moderate:
            penalty += profile["low_vis"]
        if vis_low:
            penalty += 0.7 * profile["low_vis"]
        if vis_very_low:
            penalty += profile["very_low_vis"]

        if uv > 7:
            penalty += profile["uv"]
        if uv > 9:
            penalty += profile["uv_extreme"]

        if is_thunder:
            penalty += profile["thunder"]

        return penalty

    @classmethod
    def _global_temperature_penalty(cls, temp_c: float) -> float:
        """
        Outdoor comfort penalty based on feels-like temperature distance from
        human-comfort ranges for generic outdoor usage.
        """
        ideal_low, ideal_high = 14.0, 24.0
        tolerable_low, tolerable_high = 8.0, 30.0
        hard_low, hard_high = 2.0, 35.0

        if ideal_low <= temp_c <= ideal_high:
            return 0.0

        if tolerable_low <= temp_c <= tolerable_high:
            ideal_dist = cls._distance_to_band(temp_c, ideal_low, ideal_high)
            return 0.25 + 0.10 * ideal_dist

        if hard_low <= temp_c <= hard_high:
            tolerable_dist = cls._distance_to_band(temp_c, tolerable_low, tolerable_high)
            return 0.95 + 0.14 * tolerable_dist

        hard_dist = cls._distance_to_band(temp_c, hard_low, hard_high)
        return 1.9 + 0.18 * hard_dist

    @classmethod
    def _global_weather_penalty(cls, payload: dict[str, Any]) -> float:
        condition = str(payload.get("weather_condition", "")).lower()
        precip_type = str(payload.get("precipitation_type", "")).lower()
        road_surface = str(payload.get("road_surface", "")).lower()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        wind = float(payload.get("wind_speed_kmh", 0.0) or 0.0)
        gust = float(payload.get("wind_gust_kmh", wind) or wind)
        visibility = float(payload.get("visibility_km", 10.0) or 10.0)
        uv = float(payload.get("uv_index", 0.0) or 0.0)
        humidity = float(payload.get("humidity_pct", 0.0) or 0.0)
        feels_like = payload.get("feels_like_c", payload.get("temperature_c", 15.0))
        temp = float(feels_like if feels_like is not None else 15.0)
        is_thunder = bool(payload.get("is_thunderstorm", False)) or ("thunderstorm" in condition)
        is_snow = ("snow" in condition) or (precip_type == "snow")
        is_rainy = ("rain" in condition) or ("drizzle" in condition) or (precip_type == "rain")
        is_fog = "fog" in condition
        is_heavy_rain = ("heavy_rain" in condition) or precip > 4.0
        is_heavy_snow = ("heavy_snow" in condition) or (is_snow and precip > 2.0)

        penalty = 0.0
        penalty += cls._global_temperature_penalty(temp)

        if is_rainy:
            penalty += 0.8
        if is_rainy and precip > 0.1:
            penalty += 0.6
            if precip > 1.0:
                penalty += 0.8
            if is_heavy_rain:
                penalty += 1.0

        if is_snow:
            penalty += 1.6
            if precip > 1.0:
                penalty += 0.8
            if is_heavy_snow:
                penalty += 0.9

        if road_surface == "wet":
            penalty += 0.2
        if road_surface == "icy":
            penalty += 1.1
        if road_surface == "snow_covered":
            penalty += 0.9

        if wind > 22:
            penalty += 0.4
        if wind > 32:
            penalty += 0.7
        if wind > 45:
            penalty += 1.0
        if gust > 45:
            penalty += 0.4

        if is_fog:
            penalty += 0.6
        if visibility < 8.0:
            penalty += 0.2
        if visibility < 5.0:
            penalty += 0.6
        if visibility < 2.5:
            penalty += 0.9

        if uv > 7:
            penalty += 0.25
        if uv > 9:
            penalty += 0.45

        if temp > 28 and humidity > 90:
            penalty += 0.4

        if is_thunder:
            penalty += 2.6

        return penalty

    @staticmethod
    def _global_weather_bonus(payload: dict[str, Any]) -> float:
        condition = str(payload.get("weather_condition", "")).lower()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        wind = float(payload.get("wind_speed_kmh", 0.0) or 0.0)
        visibility = float(payload.get("visibility_km", 10.0) or 10.0)
        uv = float(payload.get("uv_index", 0.0) or 0.0)
        feels_like = payload.get("feels_like_c", payload.get("temperature_c", 15.0))
        temp = float(feels_like if feels_like is not None else 15.0)
        is_thunder = bool(payload.get("is_thunderstorm", False)) or ("thunderstorm" in condition)

        if (
            condition in {"clear", "partly_cloudy"}
            and precip <= 0.05
            and wind < 15
            and visibility >= 10
            and 14 <= temp <= 24
            and not is_thunder
        ):
            return 0.6 if condition == "clear" else 0.4

        if (
            condition in {"clear", "partly_cloudy", "cloudy"}
            and precip <= 0.05
            and wind < 20
            and visibility >= 9
            and 10 <= temp <= 28
            and uv <= 7
            and not is_thunder
        ):
            return 0.2

        return 0.0

    @classmethod
    def _global_score_cap(cls, payload: dict[str, Any]) -> float:
        condition = str(payload.get("weather_condition", "")).lower()
        precip_type = str(payload.get("precipitation_type", "")).lower()
        road_surface = str(payload.get("road_surface", "")).lower()
        precip = float(payload.get("precipitation_mm", 0.0) or 0.0)
        visibility = float(payload.get("visibility_km", 10.0) or 10.0)
        feels_like = payload.get("feels_like_c", payload.get("temperature_c", 15.0))
        temp = float(feels_like if feels_like is not None else 15.0)
        is_thunder = bool(payload.get("is_thunderstorm", False)) or ("thunderstorm" in condition)
        is_snow = ("snow" in condition) or (precip_type == "snow")
        is_rainy = ("rain" in condition) or ("drizzle" in condition) or (precip_type == "rain")
        is_fog = "fog" in condition

        cap = 10.0
        if condition == "partly_cloudy":
            cap = min(cap, 9.4)
        if condition == "cloudy":
            cap = min(cap, 8.2)

        if is_rainy:
            cap = min(cap, 5.5)
            if precip > 1.0:
                cap = min(cap, 4.8)
            if precip > 4.0 or ("heavy_rain" in condition):
                cap = min(cap, 4.2)

        if is_snow:
            cap = min(cap, 5.8)
            if precip > 1.0 or ("heavy_snow" in condition):
                cap = min(cap, 4.8)

        if road_surface in {"icy", "snow_covered"}:
            cap = min(cap, 5.0)

        if is_fog or visibility < 5.0:
            cap = min(cap, 6.2)
        if visibility < 2.5:
            cap = min(cap, 4.7)

        if temp < 5 or temp > 32:
            cap = min(cap, 6.5)
        if temp < -5 or temp > 37:
            cap = min(cap, 5.8)

        if is_thunder:
            cap = min(cap, 3.8)

        return cap

    @classmethod
    def _calibrate_outdoor_suitability(cls, raw_score: float, payload: dict[str, Any]) -> float:
        """
        Blend model output with a weather-risk heuristic and enforce a realistic cap
        so generic outdoor comfort does not become unrealistically perfect.
        """
        raw = float(np.clip(raw_score, 0.0, 10.0))
        penalty = cls._global_weather_penalty(payload)
        bonus = cls._global_weather_bonus(payload)
        heuristic = float(np.clip(10.0 - penalty + bonus, 0.0, 10.0))

        blended = (0.58 * raw) + (0.42 * heuristic)
        capped = min(blended, cls._global_score_cap(payload))
        return float(np.clip(capped, 0.0, 10.0))

    def predict(self, payload: dict[str, Any], lang: str = "en") -> dict[str, Any]:
        self._ensure_loaded()

        payload = self.enrich_weather_payload(payload, climate_zone=payload.get("climate_zone"))
        frame = pd.DataFrame([payload])
        features = build_feature_frame(frame)

        prob_umb = self._umbrella_model.predict_proba(features)[0][1]
        umbrella = 1 if prob_umb > 0.4 else 0

        pred_val = self._clothing_model.predict(features)[0]
        clothes_categories = [
            "heavy_winter_coat_gloves_hat",
            "winter_coat_scarf_gloves",
            "warm_jacket_layers",
            "light_jacket_or_sweater",
            "long_sleeves_light_layer",
            "t_shirt_comfortable",
            "light_breathable_clothing",
            "very_light_clothing_stay_hydrated",
        ]
        if isinstance(pred_val, str) and pred_val in clothes_categories:
            clothing_rec = pred_val
        else:
            try:
                clothing_code = int(pred_val)
                if 0 <= clothing_code < len(clothes_categories):
                    clothing_rec = clothes_categories[clothing_code]
                else:
                    clothing_rec = InferenceService._fallback_clothing(payload.get("feels_like_c"))
            except (ValueError, TypeError):
                clothing_rec = InferenceService._fallback_clothing(payload.get("feels_like_c"))

        suit_raw = float(self._suitability_model.predict(features)[0])
        suitability = self._calibrate_outdoor_suitability(suit_raw, payload)

        recommendation_text = InferenceService._build_recommendation_text(payload, umbrella, suitability, lang=lang)
        recommendation_text = llm_text_service.generate_hero_text(
            payload=payload,
            score=suitability,
            umbrella_needed=bool(umbrella),
            clothing=clothing_rec,
            fallback_text=recommendation_text,
            lang=lang,
        )

        return {
            "umbrella_needed": bool(umbrella),
            "clothing_recommendation": clothing_rec,
            "outdoor_suitability_score": suitability,
            "recommendation_text": recommendation_text,
        }

    def predict_hourly_suitability_raw(self, payload: dict[str, Any]) -> float:
        """Continuous suitability prediction used for hourly ranking/argmax."""
        self._ensure_loaded()
        payload = self.enrich_weather_payload(payload, climate_zone=payload.get("climate_zone"))
        frame = pd.DataFrame([payload])
        features = build_feature_frame(frame)
        score_raw = float(self._suitability_model.predict(features)[0])
        return self._calibrate_outdoor_suitability(score_raw, payload)

    def predict_activity(self, payload: dict[str, Any], activity_type: str, lang: str = "en") -> dict[str, Any]:
        self._ensure_loaded()

        payload = self.enrich_weather_payload(payload, climate_zone=payload.get("climate_zone"))
        payload["activity_type"] = activity_type
        frame = pd.DataFrame([payload])
        features = build_activity_feature_frame(frame)

        base_weather_feats = build_feature_frame(frame)
        prob_umb = self._umbrella_model.predict_proba(base_weather_feats)[0][1]
        umbrella = 1 if prob_umb > 0.4 else 0

        pred_val = self._clothing_model.predict(base_weather_feats)[0]
        clothes_categories = [
            "heavy_winter_coat_gloves_hat",
            "winter_coat_scarf_gloves",
            "warm_jacket_layers",
            "light_jacket_or_sweater",
            "long_sleeves_light_layer",
            "t_shirt_comfortable",
            "light_breathable_clothing",
            "very_light_clothing_stay_hydrated",
        ]
        if isinstance(pred_val, str) and pred_val in clothes_categories:
            clothing_rec = pred_val
        else:
            try:
                clothing_code = int(pred_val)
                if 0 <= clothing_code < len(clothes_categories):
                    clothing_rec = clothes_categories[clothing_code]
                else:
                    clothing_rec = InferenceService._fallback_clothing(payload.get("feels_like_c"))
            except (ValueError, TypeError):
                clothing_rec = InferenceService._fallback_clothing(payload.get("feels_like_c"))

        gen_suit_raw = float(self._suitability_model.predict(base_weather_feats)[0])
        gen_suitability = self._calibrate_outdoor_suitability(gen_suit_raw, payload)

        act_suit_raw = float(self._activity_suitability_model.predict(features)[0])
        activity_penalty = self._activity_weather_penalty(activity_type, payload)
        capped_act_suit_raw = gen_suitability + (act_suit_raw - 5.0) * 0.4
        act_suitability = self._activity_score_cap(
            activity_type=activity_type,
            payload=payload,
            base_score=capped_act_suit_raw,
            activity_penalty=activity_penalty,
        )

        prob_go = self._activity_go_no_model.predict_proba(features)[0][1]

        auto_veto = False
        if umbrella == 1 and activity_type in ["picnic", "cycling"]:
            auto_veto = True
        elif act_suitability < ACTIVITY_GO_THRESHOLD:
            auto_veto = True

        go_or_no = False if auto_veto else bool(prob_go > 0.4)

        return {
            "activity_type": activity_type,
            "general_suitability_score": gen_suitability,
            "activity_suitability_score": act_suitability,
            "go_or_no": go_or_no,
            "umbrella_needed": bool(umbrella),
            "clothing_recommendation": clothing_rec,
            "activity_advice": InferenceService._build_activity_advice(
                activity_type, go_or_no, act_suitability, payload, lang=lang
            ),
        }

    def predict_daily(self, daily_payload: dict[str, Any], lang: str = "en") -> dict[str, Any]:
        self._ensure_loaded()
        frame = pd.DataFrame([daily_payload])
        try:
            features = build_daily_feature_frame(frame)
        except ZeroDivisionError:
            return self._predict_daily_fallback(daily_payload)

        try:
            prob_umb = self._daily_umbrella_model.predict_proba(features)[0][1]
            clothing_morn = self._daily_clothing_morning_model.predict(features)[0]
            clothing_aft = self._daily_clothing_afternoon_model.predict(features)[0]
            clothing_eve = self._daily_clothing_evening_model.predict(features)[0]
            suitability_raw = float(self._daily_suitability_model.predict(features)[0])
            suitability = float(np.clip(suitability_raw, 0.0, 10.0))
        except Exception as e:
            print(f"Fallback due to model prediction error: {e}")
            return self._predict_daily_fallback(daily_payload)

        clothes_categories = [
            "heavy_winter_coat_gloves_hat",
            "winter_coat_scarf_gloves",
            "warm_jacket_layers",
            "light_jacket_or_sweater",
            "long_sleeves_light_layer",
            "t_shirt_comfortable",
            "light_breathable_clothing",
            "very_light_clothing_stay_hydrated",
        ]

        def map_clothing(code: Any) -> str:
            if isinstance(code, str) and code in clothes_categories:
                return code
            try:
                val = int(code)
                if 0 <= val < len(clothes_categories):
                    return clothes_categories[val]
            except (ValueError, TypeError):
                pass
            return "light_jacket_or_sweater"

        return {
            "umbrella_recommended": bool(prob_umb > 0.35),
            "clothing_morning": map_clothing(clothing_morn),
            "clothing_afternoon": map_clothing(clothing_aft),
            "clothing_evening": map_clothing(clothing_eve),
            "avg_outdoor_suitability": suitability,
        }


inference_service = InferenceService()
