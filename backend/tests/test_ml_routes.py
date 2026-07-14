from __future__ import annotations

import asyncio

import backend.app.api.routes as routes


def _sample_weather(station_id: str = "WS-001", timestamp: str = "2026-04-19T10:00:00") -> dict:
    return {
        "station_id": station_id,
        "timestamp": timestamp,
        "temperature_c": 12.0,
        "feels_like_c": 10.5,
        "dew_point_c": 5.0,
        "humidity_pct": 60.0,
        "pressure_hpa": 1015.0,
        "wind_speed_kmh": 10.0,
        "wind_direction_deg": 220.0,
        "wind_gust_kmh": 18.0,
        "precipitation_mm": 0.3,
        "precipitation_type": "rain",
        "cloud_cover_pct": 55.0,
        "visibility_km": 12.0,
        "uv_index": 4.0,
        "weather_condition": "rain",
        "is_thunderstorm": False,
        "road_surface": "wet",
    }


def test_hero_route_uses_ml_prediction(monkeypatch) -> None:
    async def fake_fetch_current_weather(station_id: str) -> dict:
        return _sample_weather(station_id=station_id)

    def fake_predict(payload: dict) -> dict:
        return {
            "umbrella_needed": 1,
            "clothing_recommendation": "warm_jacket_layers",
            "outdoor_suitability_score": 6.0,
            "recommendation_text": "Model says moderate conditions.",
            "confidence": {"umbrella_probability_max": 0.91},
        }

    monkeypatch.setattr(routes, "fetch_current_weather", fake_fetch_current_weather)
    monkeypatch.setattr(routes.inference_service, "predict", fake_predict)

    response = asyncio.run(routes.get_hero("WS-001"))
    assert response.umbrella_needed is True
    assert response.clothing_recommendation == "warm_jacket_layers"
    assert response.outdoor_suitability_score == 6.0
    assert response.recommendation_text == "Model says moderate conditions."


def test_activities_route_uses_ml_activity_models(monkeypatch) -> None:
    async def fake_fetch_current_weather(station_id: str) -> dict:
        return _sample_weather(station_id=station_id)

    def fake_predict_activity(payload: dict, activity: str) -> dict:
        return {
            "activity_type": activity,
            "general_suitability_score": 7.0,
            "activity_suitability_score": 8.0,
            "go_or_no": True,
            "umbrella_needed": True,
            "clothing_recommendation": "light_jacket_or_sweater",
            "activity_advice": f"{activity} is fine now.",
        }

    monkeypatch.setattr(routes, "fetch_current_weather", fake_fetch_current_weather)
    monkeypatch.setattr(routes.inference_service, "predict_activity", fake_predict_activity)

    response = asyncio.run(routes.get_activities("WS-001"))
    assert len(response) == len(routes.ACTIVITY_TYPES)
    assert response[0].activity_type == routes.ACTIVITY_TYPES[0]
    assert response[0].activity_suitability_score == 8.0
    assert response[0].go_or_no is True


def test_daily_route_uses_daily_models(monkeypatch) -> None:
    async def fake_fetch_hourly_forecast(station_id: str, days: int = 1) -> list[dict]:
        return [
            _sample_weather(station_id=station_id, timestamp="2026-04-19T07:00:00"),
            _sample_weather(station_id=station_id, timestamp="2026-04-19T13:00:00"),
            _sample_weather(station_id=station_id, timestamp="2026-04-19T19:00:00"),
        ]

    def fake_predict_daily(payload: dict) -> dict:
        return {
            "umbrella_recommended": True,
            "avg_outdoor_suitability": 7.5,
            "clothing_morning": "warm_jacket_layers",
            "clothing_afternoon": "light_jacket_or_sweater",
            "clothing_evening": "warm_jacket_layers",
        }

    def fake_predict_hourly_suitability_raw(payload: dict) -> float:
        hour = int(str(payload["timestamp"])[11:13])
        score_by_hour = {7: 5.0, 13: 8.0, 19: 6.0}
        return score_by_hour.get(hour, 5.0)

    monkeypatch.setattr(routes, "fetch_hourly_forecast", fake_fetch_hourly_forecast)
    monkeypatch.setattr(routes.inference_service, "predict_daily", fake_predict_daily)
    monkeypatch.setattr(routes.inference_service, "predict_hourly_suitability_raw", fake_predict_hourly_suitability_raw)

    response = asyncio.run(routes.get_daily("WS-001"))
    assert response.umbrella_recommended is True
    assert response.best_outdoor_hour == 13
    assert response.avg_outdoor_suitability == 7.5
    assert response.time_strip[0].clothing == "warm_jacket_layers"
    assert response.time_strip[1].clothing == "light_jacket_or_sweater"
