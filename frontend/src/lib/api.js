/**
 * WeatherWise API client
 */
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getStations:         ()   => fetchJSON("/api/stations"),
  getHero:             (id, lang="en") => fetchJSON(`/api/hero/${id}?lang=${lang}`),
  getDaily:            (id, date, lang="en") => fetchJSON(`/api/daily/${id}?lang=${lang}${date ? `&date=${date}` : ""}`),
  getActivities:       (id, lang="en") => fetchJSON(`/api/activities/${id}?lang=${lang}`),
  getForecastAccuracy: (id) => fetchJSON(`/api/forecast-accuracy/${id}`),
};
