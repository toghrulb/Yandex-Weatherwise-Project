import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "./lib/api";
import HeroCard from "./components/HeroCard";
import TimeStrip from "./components/TimeStrip";
import ActivityCards from "./components/ActivityCards";
import ForecastAccuracy from "./components/ForecastAccuracy";
import WeatherBackground from "./components/WeatherBackground";
import { tl } from "./lib/translations";
import { useLanguage } from "./contexts/LanguageContext";
import "./index.css";

function haversineDistanceKm(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const earthRadiusKm = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return 2 * earthRadiusKm * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function findClosestStationId(stations, userLat, userLon) {
  if (!stations.length) return "";
  const closest = stations.reduce((best, current) => {
    const bestDist = haversineDistanceKm(
      userLat,
      userLon,
      Number(best.latitude),
      Number(best.longitude)
    );
    const currentDist = haversineDistanceKm(
      userLat,
      userLon,
      Number(current.latitude),
      Number(current.longitude)
    );
    return currentDist < bestDist ? current : best;
  });
  return closest.station_id;
}

function LanguageSelector({ lang, setLang }) {
  return (
    <div className="lang-selector">
      {["en", "tr", "ru"].map((l) => (
        <span 
          key={l}
          className={`lang-option ${lang === l ? "active" : ""}`}
          onClick={() => setLang(l)}
        >
          {l.toUpperCase()}
        </span>
      ))}
    </div>
  );
}

function CustomDropdown({ stations, value, onChange, closestStationId, lang }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedStation = stations.find(s => s.station_id === value) || stations[0];

  return (
    <div className="custom-dropdown-wrapper" ref={dropdownRef}>
      <button 
        className="station-selector" 
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        disabled={!stations.length}
      >
        <span className="icon">📍</span>
        <span className="selected-text">
          {selectedStation ? selectedStation.name : "Loading..."}
          {selectedStation && selectedStation.station_id === closestStationId && " (closest)"}
        </span>
        <span className="icon chevron" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>▼</span>
      </button>

      {isOpen && stations.length > 0 && (
        <ul className="custom-dropdown-menu" role="listbox">
          {stations.map((s) => (
            <li 
              key={s.station_id} 
              className={`dropdown-item ${s.station_id === value ? 'active' : ''}`}
              role="option"
              aria-selected={s.station_id === value}
              onClick={() => {
                onChange(s.station_id);
                setIsOpen(false);
              }}
            >
              <span className="dropdown-item-name">{s.name}</span>
              {s.station_id === closestStationId && <span className="dropdown-item-closest badge">{tl("Closest", lang)}</span>}
              {s.station_id === value && <span className="dropdown-item-check">✓</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function App() {
  const [stations, setStations] = useState([]);
  const [stationId, setStationId] = useState("");
  const [closestStationId, setClosestStationId] = useState("");
  const [hero, setHero] = useState(null);
  const [daily, setDaily] = useState(null);
  const [activities, setActivities] = useState([]);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const stationSelectRef = useRef(null);
  const userSelectedStationRef = useRef(false);
  const { lang, setLang } = useLanguage();

  // Load stations once
  useEffect(() => {
    let cancelled = false;

    const resolveInitialStation = async () => {
      try {
        const loadedStations = await api.getStations();
        if (cancelled) return;

        setStations(loadedStations);
        if (!loadedStations.length) {
          setError("No stations available.");
          setLoading(false);
          return;
        }

        const fallbackStationId = loadedStations[0].station_id;
        const useFallback = () => {
          setClosestStationId("");
          setStationId((currentId) => {
            if (userSelectedStationRef.current) return currentId;
            return currentId || fallbackStationId;
          });
        };

        if (!("geolocation" in navigator)) {
          useFallback();
          return;
        }

        navigator.geolocation.getCurrentPosition(
          (position) => {
            if (cancelled) return;
            const nearestStationId = findClosestStationId(
              loadedStations,
              position.coords.latitude,
              position.coords.longitude
            );
            setClosestStationId(nearestStationId || "");
            setStationId((currentId) => {
              if (userSelectedStationRef.current) return currentId;
              return nearestStationId || currentId || fallbackStationId;
            });
          },
          () => {
            if (cancelled) return;
            useFallback();
          },
          {
            enableHighAccuracy: false,
            timeout: 7000,
            maximumAge: 300000,
          }
        );
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setError("Failed to load stations.");
        setLoading(false);
      }
    };

    resolveInitialStation();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load all data when station changes
  const loadData = useCallback(async (sid) => {
    setLoading(true);
    setError(null);
    try {
      const [heroData, dailyData, actData, accData] = await Promise.allSettled([
        api.getHero(sid, lang),
        api.getDaily(sid, null, lang),
        api.getActivities(sid, lang),
        api.getForecastAccuracy(sid),
      ]);

      if (heroData.status === "fulfilled") setHero(heroData.value);
      else throw new Error("Failed to load weather data");

      if (dailyData.status === "fulfilled") setDaily(dailyData.value);
      if (actData.status === "fulfilled") setActivities(actData.value);
      if (accData.status === "fulfilled") setAccuracy(accData.value);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [lang]);

  useEffect(() => {
    if (!stationId) return;
    loadData(stationId);
  }, [stationId, loadData]);

  // Weather condition for reactive background
  const urlParams = new URLSearchParams(window.location.search);
  const forceWeather = urlParams.get('weather');
  const baseCondition = forceWeather || hero?.raw?.weather_condition || "clear";
  const isNight = hero?.raw?.is_day === false;
  const weatherCondition = `${baseCondition}${isNight ? "-night" : ""}`;

  // Crucial fix: apply weather condition to body so CSS variables inherit globally
  useEffect(() => {
    document.body.setAttribute('data-weather', weatherCondition);
  }, [weatherCondition]);

  return (
    <div className="app" data-weather={weatherCondition}>
      <WeatherBackground condition={weatherCondition} />
      
      {/* Top Header Row for Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
        <CustomDropdown 
          stations={stations} 
          value={stationId} 
          onChange={(val) => {
            userSelectedStationRef.current = true;
            setStationId(val);
          }}
          closestStationId={closestStationId} 
          lang={lang}
        />
        <LanguageSelector lang={lang} setLang={setLang} />
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          <span className="text-secondary">{tl("Fetching weather data...", lang)}</span>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="card error-card">
          <p>⚠️ {error}</p>
          <button
            onClick={() => loadData(stationId)}
            style={{
              marginTop: "1rem",
              padding: "0.5rem 1.5rem",
              border: "1px solid var(--accent-cyan)",
              borderRadius: "var(--radius-full)",
              background: "transparent",
              color: "var(--accent-cyan)",
              cursor: "pointer",
              fontFamily: "var(--font-family)",
              fontWeight: 600,
            }}
          >
            {tl("Retry", lang)}
          </button>
        </div>
      )}

      {/* Content */}
      {!loading && !error && (
        <>
          <HeroCard data={hero} />
          <TimeStrip data={daily} />
          <ActivityCards activities={activities} />
          <ForecastAccuracy data={accuracy} />

          <footer className="footer">
            <span className="brand">WeatherWise</span> · Anadolu Hackathon 2026
            <br />
            Powered by Open-Meteo · Updated {new Date().toLocaleTimeString()}
          </footer>
        </>
      )}
    </div>
  );
}
