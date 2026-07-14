from __future__ import annotations

import math

from backend.app.services.inference import InferenceService


class _UmbrellaModelBrokenProba:
    def predict(self, x):
        return [1]

    def predict_proba(self, x):
        raise AttributeError("broken proba for compatibility test")


class _ClothingModelStub:
    def predict(self, x):
        return ["light_jacket_or_sweater"]


class _SuitabilityModelStub:
    def predict(self, x):
        return [7.2]


class _SuitabilityModelConstantStub:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x):
        return [self.value]


class _ActivitySuitabilityModelStub:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x):
        return [self.value]


class _ActivityGoNoModelStub:
    def __init__(self, value: int) -> None:
        self.value = value

    def predict(self, x):
        return [self.value]


def test_predict_handles_predict_proba_compat_errors() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()

    out = svc.predict(
        {
            "station_id": "WS-001",
            "climate_zone": "semi-arid_continental",
            "season": "spring",
            "weather_condition": "rain",
            "precipitation_type": "rain",
            "road_surface": "wet",
            "timestamp": "2026-04-19T10:00:00",
            "temperature_c": 12.0,
            "feels_like_c": 10.0,
            "humidity_pct": 70.0,
            "pressure_hpa": 1012.0,
            "wind_speed_kmh": 14.0,
            "wind_gust_kmh": 20.0,
            "precipitation_mm": 0.8,
            "cloud_cover_pct": 65.0,
            "visibility_km": 8.0,
            "uv_index": 3.0,
        }
    )

    assert out["umbrella_needed"] == 1
    assert out["clothing_recommendation"] == "light_jacket_or_sweater"
    assert 0.0 <= out["outdoor_suitability_score"] <= 10.0
    assert out["outdoor_suitability_score"] < 7.2
    assert out["confidence"]["umbrella_probability_max"] is None


def test_predict_activity_uses_continuous_score_and_threshold_go_logic() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(5.04)
    # This should be ignored now; go/no must come from score threshold.
    svc._activity_go_no_model = _ActivityGoNoModelStub(1)

    out = svc.predict_activity(
        {
            "station_id": "WS-001",
            "climate_zone": "semi-arid_continental",
            "season": "spring",
            "weather_condition": "clear",
            "precipitation_type": "none",
            "road_surface": "dry",
            "timestamp": "2026-04-19T10:00:00",
            "temperature_c": 12.0,
            "feels_like_c": 10.0,
            "humidity_pct": 70.0,
            "pressure_hpa": 1012.0,
            "wind_speed_kmh": 14.0,
            "wind_gust_kmh": 20.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 65.0,
            "visibility_km": 8.0,
            "uv_index": 3.0,
        },
        activity_type="cycling",
    )

    assert 0.0 <= out["activity_suitability_score"] <= 5.1
    assert out["go_or_no"] is False


def test_predict_activity_applies_activity_specific_rain_penalties() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(6.2)

    payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "spring",
        "weather_condition": "rain",
        "precipitation_type": "rain",
        "road_surface": "wet",
        "timestamp": "2026-04-19T10:00:00",
        "temperature_c": 9.0,
        "feels_like_c": 7.0,
        "humidity_pct": 85.0,
        "pressure_hpa": 1009.0,
        "wind_speed_kmh": 16.0,
        "wind_gust_kmh": 24.0,
        "precipitation_mm": 1.8,
        "cloud_cover_pct": 88.0,
        "visibility_km": 7.0,
        "uv_index": 1.0,
    }

    cycling = svc.predict_activity(payload, activity_type="cycling")
    driving = svc.predict_activity(payload, activity_type="driving")

    assert cycling["activity_suitability_score"] < driving["activity_suitability_score"]


def test_predict_activity_applies_activity_specific_temperature_penalties() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(8.8)

    cold_payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "spring",
        "weather_condition": "clear",
        "precipitation_type": "none",
        "road_surface": "dry",
        "timestamp": "2026-04-19T10:00:00",
        "temperature_c": 8.0,
        "feels_like_c": 8.0,
        "humidity_pct": 55.0,
        "pressure_hpa": 1014.0,
        "wind_speed_kmh": 8.0,
        "wind_gust_kmh": 14.0,
        "precipitation_mm": 0.0,
        "cloud_cover_pct": 15.0,
        "visibility_km": 10.0,
        "uv_index": 4.0,
    }

    picnic = svc.predict_activity(cold_payload, activity_type="picnic")
    driving = svc.predict_activity(cold_payload, activity_type="driving")

    assert picnic["activity_suitability_score"] < driving["activity_suitability_score"]
    assert picnic["activity_suitability_score"] <= 7.0


def test_predict_activity_low_temperature_hits_running_more_than_driving() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(8.5)

    cold_payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "winter",
        "weather_condition": "clear",
        "precipitation_type": "none",
        "road_surface": "dry",
        "timestamp": "2026-01-19T10:00:00",
        "temperature_c": -7.0,
        "feels_like_c": -8.0,
        "humidity_pct": 60.0,
        "pressure_hpa": 1015.0,
        "wind_speed_kmh": 6.0,
        "wind_gust_kmh": 10.0,
        "precipitation_mm": 0.0,
        "cloud_cover_pct": 20.0,
        "visibility_km": 10.0,
        "uv_index": 2.0,
    }

    running = svc.predict_activity(cold_payload, activity_type="running")
    driving = svc.predict_activity(cold_payload, activity_type="driving")

    assert running["activity_suitability_score"] < driving["activity_suitability_score"]


def test_predict_activity_snowy_icy_hits_driving_more_than_walking() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelStub()
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(8.4)

    risky_payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "winter",
        "weather_condition": "snow",
        "precipitation_type": "snow",
        "road_surface": "icy",
        "timestamp": "2026-01-19T19:00:00",
        "temperature_c": -2.0,
        "feels_like_c": -4.0,
        "humidity_pct": 88.0,
        "pressure_hpa": 1008.0,
        "wind_speed_kmh": 22.0,
        "wind_gust_kmh": 34.0,
        "precipitation_mm": 1.6,
        "cloud_cover_pct": 95.0,
        "visibility_km": 2.0,
        "uv_index": 0.0,
        "is_thunderstorm": False,
    }

    driving = svc.predict_activity(risky_payload, activity_type="driving")
    walking = svc.predict_activity(risky_payload, activity_type="walking")

    assert driving["activity_suitability_score"] < walking["activity_suitability_score"]


def test_predict_calibrates_rainy_conditions_down_from_overconfident_raw() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelConstantStub(9.6)

    payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "spring",
        "weather_condition": "rain",
        "precipitation_type": "rain",
        "road_surface": "wet",
        "timestamp": "2026-04-19T10:00:00",
        "temperature_c": 4.0,
        "feels_like_c": 3.0,
        "humidity_pct": 85.0,
        "pressure_hpa": 1009.0,
        "wind_speed_kmh": 16.0,
        "wind_gust_kmh": 24.0,
        "precipitation_mm": 1.2,
        "cloud_cover_pct": 88.0,
        "visibility_km": 7.0,
        "uv_index": 1.0,
    }

    out = svc.predict(payload)
    assert out["outdoor_suitability_score"] < 7.5


def test_predict_avoids_perfect_score_for_cool_cloudy_conditions() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelConstantStub(10.0)

    payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "spring",
        "weather_condition": "cloudy",
        "precipitation_type": "none",
        "road_surface": "dry",
        "timestamp": "2026-04-19T10:00:00",
        "temperature_c": 9.0,
        "feels_like_c": 8.0,
        "humidity_pct": 70.0,
        "pressure_hpa": 1016.0,
        "wind_speed_kmh": 11.0,
        "wind_gust_kmh": 18.0,
        "precipitation_mm": 0.0,
        "cloud_cover_pct": 90.0,
        "visibility_km": 10.0,
        "uv_index": 2.0,
    }

    out = svc.predict(payload)
    assert out["outdoor_suitability_score"] < 9.0


def test_predict_activity_caps_overconfident_scores_in_cool_conditions() -> None:
    svc = InferenceService()
    svc._umbrella_model = _UmbrellaModelBrokenProba()
    svc._clothing_model = _ClothingModelStub()
    svc._suitability_model = _SuitabilityModelConstantStub(10.0)
    svc._activity_suitability_model = _ActivitySuitabilityModelStub(10.0)

    payload = {
        "station_id": "WS-001",
        "climate_zone": "semi-arid_continental",
        "season": "spring",
        "weather_condition": "cloudy",
        "precipitation_type": "none",
        "road_surface": "dry",
        "timestamp": "2026-04-19T10:00:00",
        "temperature_c": 9.0,
        "feels_like_c": 8.0,
        "humidity_pct": 70.0,
        "pressure_hpa": 1016.0,
        "wind_speed_kmh": 11.0,
        "wind_gust_kmh": 18.0,
        "precipitation_mm": 0.0,
        "cloud_cover_pct": 90.0,
        "visibility_km": 10.0,
        "uv_index": 2.0,
    }

    walking = svc.predict_activity(payload, activity_type="walking")
    running = svc.predict_activity(payload, activity_type="running")
    cycling = svc.predict_activity(payload, activity_type="cycling")

    assert walking["activity_suitability_score"] <= 8.1
    assert running["activity_suitability_score"] <= 7.8
    assert cycling["activity_suitability_score"] <= 7.5
