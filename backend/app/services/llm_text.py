from __future__ import annotations

import json
import os
import re
from hashlib import sha1
from typing import Any

import httpx


class LLMTextService:
    """
    Lightweight OpenAI-backed text helper for short UX comments.
    Falls back safely to deterministic text when unavailable.
    """

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("tutor_openai_api")
            or os.getenv("TUTOR_OPENAI_API")
            or os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("WEATHERWISE_OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = float(os.getenv("WEATHERWISE_OPENAI_TIMEOUT_S", "6.0"))
        enabled_flag = os.getenv("WEATHERWISE_ENABLE_LLM_TEXT", "1").strip().lower() not in {"0", "false", "no"}
        self._base_enabled = bool(self.api_key) and enabled_flag
        self._cache: dict[str, str] = {}
        self._cache_limit = 256

    def _is_enabled(self) -> bool:
        return self._base_enabled and ("PYTEST_CURRENT_TEST" not in os.environ)

    @staticmethod
    def _normalize_text(text: str, max_words: int) -> str:
        compact = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
        if not compact:
            return ""
        words = compact.split(" ")
        if len(words) > max_words:
            compact = " ".join(words[:max_words]).rstrip(",;:")
        if compact and compact[-1] not in ".!?":
            compact += "."
        return compact

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _cache_key(self, kind: str, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return f"{kind}:{sha1(raw.encode('utf-8')).hexdigest()}"

    def _cache_get(self, key: str) -> str | None:
        return self._cache.get(key)

    def _cache_set(self, key: str, value: str) -> None:
        if len(self._cache) >= self._cache_limit:
            first_key = next(iter(self._cache.keys()))
            self._cache.pop(first_key, None)
        self._cache[key] = value

    def _chat(self, messages: list[dict[str, str]], max_tokens: int = 120, temperature: float = 0.2) -> str | None:
        if not self._is_enabled():
            return None

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception:
            return None

        try:
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return None

    def generate_hero_text(
        self,
        payload: dict[str, Any],
        score: float,
        umbrella_needed: bool,
        clothing: str,
        fallback_text: str,
        lang: str = "en",
    ) -> str:
        if not self._is_enabled():
            return fallback_text

        request_payload = {
            "weather_condition": str(payload.get("weather_condition", "clear")),
            "temperature_c": float(payload.get("temperature_c", 0.0) or 0.0),
            "feels_like_c": float(payload.get("feels_like_c", payload.get("temperature_c", 0.0)) or 0.0),
            "precipitation_mm": float(payload.get("precipitation_mm", 0.0) or 0.0),
            "precipitation_type": str(payload.get("precipitation_type", "none")),
            "wind_speed_kmh": float(payload.get("wind_speed_kmh", 0.0) or 0.0),
            "uv_index": float(payload.get("uv_index", 0.0) or 0.0),
            "outdoor_suitability_score": float(score),
            "umbrella_needed": bool(umbrella_needed),
            "clothing_recommendation": str(clothing),
            "lang": lang,
        }
        key = self._cache_key("hero", request_payload)
        cached = self._cache_get(key)
        if cached:
            return cached

        messages = [
            {
                "role": "system",
                "content": (
                    "You write WeatherWise micro-advice. "
                    "Follow rubric: clear plain English, actionable, grounded only in provided facts. "
                    "No jargon, no hallucinations, no extra risks beyond given weather. "
                    "Keep it understandable in ~1 second.\n\n"
                    f"CRITICAL RULE: Always respond in the requested language code (e.g., 'en' for English, 'tr' for Turkish, 'ru' for Russian).\n"
                    f"Requested language code: {lang.upper()}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create recommendation_text for hero card.\n"
                    "Rules:\n"
                    "1) 1-2 short sentences.\n"
                    "2) Max 24 words total.\n"
                    "3) Include one action (umbrella or clothing or timing).\n"
                    "4) Do not invent data.\n"
                    "Return JSON only: {\"recommendation_text\":\"...\"}\n\n"
                    f"Facts:\n{json.dumps(request_payload, ensure_ascii=True)}"
                ),
            },
        ]
        raw = self._chat(messages=messages, max_tokens=110, temperature=0.2)
        parsed = self._extract_json(raw or "")
        candidate = str(parsed.get("recommendation_text", "")) if parsed else ""
        candidate = self._normalize_text(candidate, max_words=24)
        if not candidate:
            return fallback_text

        self._cache_set(key, candidate)
        return candidate

    def generate_activity_advices(
        self,
        payload: dict[str, Any],
        predictions: list[dict[str, Any]],
        lang: str = "en",
    ) -> dict[str, str]:
        if (not self._is_enabled()) or (not predictions):
            return {}

        weather_facts = {
            "weather_condition": str(payload.get("weather_condition", "clear")),
            "temperature_c": float(payload.get("temperature_c", 0.0) or 0.0),
            "feels_like_c": float(payload.get("feels_like_c", payload.get("temperature_c", 0.0)) or 0.0),
            "precipitation_mm": float(payload.get("precipitation_mm", 0.0) or 0.0),
            "precipitation_type": str(payload.get("precipitation_type", "none")),
            "wind_speed_kmh": float(payload.get("wind_speed_kmh", 0.0) or 0.0),
            "wind_gust_kmh": float(payload.get("wind_gust_kmh", 0.0) or 0.0),
            "visibility_km": float(payload.get("visibility_km", 0.0) or 0.0),
            "uv_index": float(payload.get("uv_index", 0.0) or 0.0),
            "road_surface": str(payload.get("road_surface", "dry")),
            "is_thunderstorm": bool(payload.get("is_thunderstorm", False)),
        }
        compact_predictions = [
            {
                "activity_type": str(item.get("activity_type", "")),
                "activity_suitability_score": float(item.get("activity_suitability_score", 0.0) or 0.0),
                "go_or_no": bool(item.get("go_or_no", False)),
                "umbrella_needed": bool(item.get("umbrella_needed", False)),
                "clothing_recommendation": str(item.get("clothing_recommendation", "")),
            }
            for item in predictions
        ]
        key = self._cache_key(
            "activities",
            {"weather": weather_facts, "predictions": compact_predictions, "lang": lang},
        )
        cached = self._cache_get(key)
        if cached:
            parsed_cached = self._extract_json(cached)
            if parsed_cached:
                return {k: self._normalize_text(str(v), max_words=14) for k, v in parsed_cached.items()}

        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise activity weather tips for WeatherWise. "
                    "Use only provided facts. Keep each tip actionable, plain, and fast to understand.\n\n"
                    f"CRITICAL RULE: Always respond in the requested language code (e.g., 'en' for English, 'tr' for Turkish, 'ru' for Russian).\n"
                    f"Requested language code: {lang.upper()}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate one short comment per activity.\n"
                    "Rules:\n"
                    "1) Exactly one sentence per activity.\n"
                    "2) Max 14 words per sentence.\n"
                    "3) Respect score/go_or_no: high score positive, mid score cautious, low score discourage.\n"
                    "4) Mention strongest relevant reason (rain, wind, snow, ice, fog, temperature, UV).\n"
                    "5) If umbrella_needed=true, you may mention umbrella.\n"
                    "Return JSON object only with activity_type keys.\n\n"
                    f"Weather facts:\n{json.dumps(weather_facts, ensure_ascii=True)}\n\n"
                    f"Activity predictions:\n{json.dumps(compact_predictions, ensure_ascii=True)}"
                ),
            },
        ]
        raw = self._chat(messages=messages, max_tokens=260, temperature=0.2)
        parsed = self._extract_json(raw or "")
        if not parsed:
            return {}

        advice_map: dict[str, str] = {}
        for activity in [str(item.get("activity_type", "")) for item in compact_predictions]:
            if activity not in parsed:
                continue
            advice = self._normalize_text(str(parsed[activity]), max_words=14)
            if advice:
                advice_map[activity] = advice

        if advice_map:
            self._cache_set(key, json.dumps(advice_map, ensure_ascii=True))
        return advice_map


llm_text_service = LLMTextService()
