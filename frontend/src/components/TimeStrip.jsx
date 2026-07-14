import { CLOTHING_ICONS, formatClothing } from "../lib/icons";
import { tl } from "../lib/translations";
import { useLanguage } from "../contexts/LanguageContext";

export default function TimeStrip({ data }) {
  const { lang } = useLanguage();
  if (!data || !data.time_strip) return null;

  const getComfortLabel = (indicator) => {
    if (indicator === "green") return tl("Great", lang);
    if (indicator === "red") return tl("Poor", lang);
    return tl("Fair", lang);
  };

  return (
    <div className="section">
      <div className="section-header">
        <span>🕐</span>
        <span className="text-caption">{tl("Today's Timeline", lang)}</span>
      </div>
      <div className="timeline-helper">{tl("Outdoor comfort by period", lang)}</div>

      <div className="time-strip">
        {data.time_strip.map((slot) => (
          <div className="card time-slot" key={slot.period}>
            <div className="period-label">{slot.period}</div>
            <div className="clothing-icon">
              {CLOTHING_ICONS[slot.clothing] || "👕"}
            </div>
            <div className="clothing-name">{formatClothing(slot.clothing, lang)}</div>
            <div className="tip">{slot.tip}</div>
            <div className="comfort-indicator" title="Outdoor comfort for this period">
              <div className={`go-dot ${slot.go_indicator}`} />
              <span className={`comfort-label-text ${slot.go_indicator}`}>
                {getComfortLabel(slot.go_indicator)}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Best outdoor hour highlight */}
      {data.best_outdoor_hour != null && (
        <div style={{
          textAlign: "center",
          marginTop: "0.75rem",
          fontSize: "0.8rem",
          color: "var(--text-secondary)"
        }}>
          ⭐ {tl("Best time outdoors:", lang)} <strong>{data.best_outdoor_hour}:00</strong>
        </div>
      )}
    </div>
  );
}
