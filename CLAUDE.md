# WeatherWise — Development Progress Log

## Project Overview
**Hackathon:** Anadolu Hackathon 2026 | Case 3 — WEATHERWISE
**Team:** 2 people (Person A: Frontend/Backend, Person B: ML)
**Stack:** FastAPI + Vite/React + scikit-learn + Docker
**Weather API:** Open-Meteo (free, no key)

## Roadmap Status

| Phase | Status | Notes |
|-------|--------|-------|
| Planning & Analysis | ✅ Done | See `docs/` artifacts |
| Backend API (FastAPI) | 🔄 In Progress | Open-Meteo integration, CSV data loading |
| Frontend (Vite+React) | ⬜ Not Started | Hero card, time strip, activity cards |
| ML Model Integration | ⬜ Waiting on teammate | Placeholder predictions for now |
| Docker Setup | ⬜ Not Started | docker-compose with backend + frontend |
| Polish & Verification | ⬜ Not Started | — |

## Key Decisions
- **Open-Meteo** chosen over Visual Crossing (free, no API key, sufficient parameters)
- **Vite+React** chosen over Next.js (faster setup, no SSR needed)
- Backend loads CSV data at startup for historical lookups, fetches live data from Open-Meteo
- ML models will be `.pkl`/`.joblib` files loaded at startup — using rule-based fallback until models are ready

## How to Run

### Development (without Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Production (Docker)
```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

## Log

### 2026-04-15 ~05:00
- Completed planning: project analysis, API research, implementation plan
- User approved plan — starting backend + frontend development
- Creating FastAPI backend with Open-Meteo integration
- Creating Vite+React frontend with weather dashboard UI
