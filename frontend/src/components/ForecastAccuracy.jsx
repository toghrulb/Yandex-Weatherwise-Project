import { useLanguage } from "../contexts/LanguageContext";
import { tl } from "../lib/translations";

export default function ForecastAccuracy({ data }) {
  const { lang } = useLanguage();
  if (!data) return null;

  return (
    <div className="section">
      <div className="section-header">
        <span>📊</span>
        <span className="text-caption">{tl("Forecast Confidence", lang)}</span>
      </div>

      <div className="card">
        <div className="accuracy-stats">
          <div className="accuracy-stat">
            <div className="value" style={{ color: data.umbrella_accuracy_pct >= 80 ? "#22c55e" : "#f59e0b" }}>
              {data.umbrella_accuracy_pct.toFixed(0)}%
            </div>
            <div className="stat-label">{tl("Umbrella", lang)}</div>
          </div>
          <div className="accuracy-stat">
            <div className="value" style={{ color: data.clothing_accuracy_pct >= 70 ? "#22c55e" : "#f59e0b" }}>
              {data.clothing_accuracy_pct.toFixed(0)}%
            </div>
            <div className="stat-label">{tl("Clothing", lang)}</div>
          </div>
          <div className="accuracy-stat">
            <div className="value" style={{ color: data.condition_accuracy_pct >= 70 ? "#22c55e" : "#f59e0b" }}>
              {data.condition_accuracy_pct.toFixed(0)}%
            </div>
            <div className="stat-label">{tl("Condition", lang)}</div>
          </div>
        </div>

        <div style={{ marginTop: "1rem", fontSize: "0.78rem", color: "var(--text-muted)", textAlign: "center" }}>
          {tl("Based on ", lang)}{data.total_forecasts.toLocaleString()}{tl(" historical forecasts ·", lang)}
          <br/>
          {tl("Avg temp error: ±", lang)}{data.avg_temp_error_c.toFixed(1)}°C
        </div>
      </div>
    </div>
  );
}
