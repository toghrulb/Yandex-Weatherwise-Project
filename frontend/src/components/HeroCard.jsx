import { WEATHER_ICONS, WEATHER_ICONS_NIGHT, CLOTHING_ICONS, formatClothing, getSuitabilityColor } from "../lib/icons";
import { useState } from "react";
import { tl } from "../lib/translations";
import { useLanguage } from "../contexts/LanguageContext";

export default function HeroCard({ data }) {
  const [showAI, setShowAI] = useState(false);
  const { lang } = useLanguage();
  if (!data) return null;

  const { raw, outdoor_suitability_score: score, umbrella_needed, clothing_recommendation, recommendation_headline, recommendation_text } = data;
  const isDay = raw.is_day !== false; // defaults to true
  const iconSet = isDay ? WEATHER_ICONS : WEATHER_ICONS_NIGHT;
  const weatherIcon = iconSet[raw.weather_condition] || "🌡️";
  const clothingIcon = CLOTHING_ICONS[clothing_recommendation] || "👕";
  const suitColor = getSuitabilityColor(score);

  // SVG ring
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 10) * circumference;

  return (
    <div className="card hero-card section">
      {/* Suitability ring */}
      <div className="suitability-ring">
        <svg viewBox="0 0 90 90">
          <circle className="ring-bg" cx="45" cy="45" r={radius} />
          <circle
            className="ring-fill"
            cx="45" cy="45" r={radius}
            stroke={suitColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <span className="ring-label" style={{ color: suitColor }}>
          {score.toFixed(1)}
        </span>
      </div>

      {/* Weather icon */}
      <div className="weather-icon">{weatherIcon}</div>

      {/* Headline */}
      <h1 className="headline">{recommendation_headline}</h1>

      {/* Recommendation text */}
      <p className="recommendation-text">{recommendation_text}</p>

      {/* Indicators */}
      <div className="hero-indicators">
        <div className="indicator">
          <span className={`icon ${umbrella_needed ? "umbrella-yes" : "umbrella-no"}`}>☂️</span>
          <span className="label">{umbrella_needed ? tl("Take it", lang) : tl("Not needed", lang)}</span>
        </div>
        <div className="indicator">
          <span className="icon">{clothingIcon}</span>
          <span className="label" style={{ whiteSpace: "nowrap" }}>{formatClothing(clothing_recommendation, lang).substring(0, 20)}</span>
        </div>
        <div className="indicator">
          <span className="icon">🌡️</span>
          <span className="label">{tl("Feels like", lang)} {raw.feels_like_c.toFixed(0)}°C</span>
        </div>
      </div>

      {/* AI explainability toggle */}
      <div className="ai-explain" onClick={() => setShowAI(!showAI)}>
        <div className="ai-header">
          <span>💡</span>
          <span>{showAI ? tl("Hide", lang) : tl("How we decided this", lang)}</span>
        </div>
        {showAI && (
          <div className="ai-body">
            Based on {raw.temperature_c.toFixed(1)}°C (feels like {raw.feels_like_c.toFixed(1)}°C),{" "}
            {raw.humidity_pct}% humidity, {raw.wind_speed_kmh.toFixed(0)} km/h wind,{" "}
            {raw.precipitation_mm > 0 ? `${raw.precipitation_mm}mm ${raw.precipitation_type}` : "no precipitation"},{" "}
            UV index {raw.uv_index.toFixed(0)}, {raw.weather_condition.replace(/_/g, " ")} skies.
            Our model scored outdoor comfort at {score.toFixed(1)}/10.
          </div>
        )}
      </div>
    </div>
  );
}
