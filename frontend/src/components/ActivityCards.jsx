import { useEffect, useRef, useState } from "react";
import { ACTIVITY_ICONS, formatClothing, getScoreBarColor } from "../lib/icons";
import { tl } from "../lib/translations";
import { useLanguage } from "../contexts/LanguageContext";

export default function ActivityCards({ activities }) {
  const [selected, setSelected] = useState(null);
  const scrollRef = useRef(null);
  const { lang } = useLanguage();

  if (!activities || activities.length === 0) return null;

  const active = activities.find((a) => a.activity_type === selected);
  const getActivityState = (score) => {
    if (score >= 6) {
      return { key: "great", dot: "green", title: tl("Ready To Go", lang) };
    }
    if (score >= 4) {
      return { key: "fair", dot: "orange", title: tl("Proceed With Care", lang) };
    }
    return { key: "poor", dot: "red", title: tl("Hold Off For Now", lang) };
  };
  const activeState = active ? getActivityState(active.activity_suitability_score) : null;

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return undefined;

    const onWheel = (event) => {
      const canScrollHorizontally = container.scrollWidth > container.clientWidth;
      if (!canScrollHorizontally) return;

      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;

      const maxScroll = container.scrollWidth - container.clientWidth;
      const nextScroll = Math.max(0, Math.min(maxScroll, container.scrollLeft + delta));

      container.scrollLeft = nextScroll;
      event.preventDefault();
      event.stopPropagation();
    };

    container.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", onWheel);
    };
  }, [activities.length]);

  return (
    <div className="section">
      <div className="section-header">
        <span>🎯</span>
        <span className="text-caption">{tl("Activity Planner", lang)}</span>
      </div>

      {/* Scrollable bubbles */}
      <div
        className="activity-scroll"
        ref={scrollRef}
        role="list"
        aria-label="Activity options"
      >
        {activities.map((a) => {
          const state = getActivityState(a.activity_suitability_score);
          return (
          <div
            key={a.activity_type}
            className={`activity-bubble ${selected === a.activity_type ? "active" : ""}`}
            onClick={() => setSelected(selected === a.activity_type ? null : a.activity_type)}
            role="listitem"
          >
            <span className="a-icon">{ACTIVITY_ICONS[a.activity_type] || "🏃"}</span>
            <span className="a-label">{tl(a.activity_type, lang)}</span>
            <div
              className={`go-dot ${state.dot}`}
              style={{ width: 6, height: 6, borderRadius: "50%" }}
            />
          </div>
          );
        })}
      </div>

      {/* Expanded detail */}
      {active && (
        <div className="card activity-detail">
          <div className={`go-label ${activeState.key}`}>
            {activeState.title}
          </div>

          {/* Score bar */}
          <div style={{ marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: "0.25rem" }}>
              <span className="text-secondary">{tl("Activity score", lang)}</span>
              <span style={{ fontWeight: 700 }}>{active.activity_suitability_score.toFixed(1)}/10</span>
            </div>
            <div className="score-bar">
              <div
                className="score-bar-fill"
                style={{
                  width: `${(active.activity_suitability_score / 10) * 100}%`,
                  background: getScoreBarColor(active.activity_suitability_score),
                }}
              />
            </div>
          </div>

          <p className="advice-text">{active.activity_advice}</p>

          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            {active.umbrella_needed && `☂️ ${tl("Bring an umbrella · ", lang)}`}
            👕 {formatClothing(active.clothing_recommendation, lang)}
          </div>
        </div>
      )}
    </div>
  );
}
