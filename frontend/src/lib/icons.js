import { tl } from "./translations";

/**
 * Weather icon and clothing icon mappings
 */
export const WEATHER_ICONS = {
  clear: "☀️",
  partly_cloudy: "⛅",
  cloudy: "☁️",
  rain: "🌧️",
  heavy_rain: "⛈️",
  drizzle: "🌦️",
  snow: "🌨️",
  heavy_snow: "❄️",
  fog: "🌫️",
  thunderstorm: "⚡",
  windy: "💨",
  hail: "🧊",
};

export const WEATHER_ICONS_NIGHT = {
  ...WEATHER_ICONS,
  clear: "🌙",
  partly_cloudy: "☁️",
};

export const CLOTHING_ICONS = {
  heavy_winter_coat_gloves_hat: "🧥",
  winter_coat_scarf_gloves: "🧣",
  warm_jacket_layers: "🧥",
  light_jacket_or_sweater: "👔",
  long_sleeves_light_layer: "👕",
  t_shirt_comfortable: "👕",
  light_breathable_clothing: "👙",
  very_light_clothing_stay_hydrated: "🩳",
};

export const ACTIVITY_ICONS = {
  walking: "🚶",
  running: "🏃",
  cycling: "🚴",
  driving: "🚗",
  outdoor_work: "🔧",
  picnic: "🧺",
  sports: "⚽",
  commute: "🚌",
};

export function formatClothing(str, lang = "en") {
  if (!str) return "";
  const baseKey = str.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  return tl(baseKey, lang);
}

export function getSuitabilityColor(score) {
  if (score >= 7) return "#22c55e";
  if (score >= 6) return "#84cc16";
  if (score >= 3) return "#f59e0b";
  return "#ef4444";
}

export function getScoreBarColor(score) {
  if (score >= 6) return "linear-gradient(90deg, #22c55e, #4ade80)";
  if (score >= 4) return "linear-gradient(90deg, #f59e0b, #fbbf24)";
  return "linear-gradient(90deg, #ef4444, #f87171)";
}
