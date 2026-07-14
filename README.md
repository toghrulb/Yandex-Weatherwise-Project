# WeatherWise 🌤️🧠

**WeatherWise** is an AI-powered smart weather application developed for the Anadolu Hackathon. It bridges modern UI experiences, live weather aggregation, and localized AI insights through specialized Scikit-Learn pipelines and Language Models to provide precise clothing and activity recommendations.

## 🔥 Key Features

- **Responsive & Dynamic UI**: Built with React and Vite. The dashboard uses CSS variables to adapt dynamically to the weather, with changing background gradients and weather cards.
- **Multilingual Support**: Real-time toggling between English (EN), Turkish (TR), and Russian (RU) for static assets and AI-generated dynamic text.
- **AI-Powered Recommendations**: Uses multiple local Sklearn classifiers alongside OpenAI integrations to score activity suitability (e.g., picnic, cycling, running) and give localized equipment/clothing advice based on climate factors, temperature extremes, and condition modifiers.
- **Full Docker Integration**: Completely containerized development pipeline making deployment a 1-click process across frontend and backend clusters.

## 🏗️ Project Architecture

```text
.
├── backend/              # FastAPI Python service hosting ML models + OpenAI LLM layer
├── frontend/             # Vite/React App with custom CSS-driven weather reactive variables 
├── ml/                   # Model Training Scripts (sklearn LogisticRegression pipelines)
├── data/                 # Raw and processed CSV feature sets used by the backend
├── docs/                 # Documentation (Presentation Outline, Architecture Diagram)
├── docker-compose.yml    # Root cluster deployment
└── README.md
```

## 🚀 Getting Started

Ensure you have Docker and `docker-compose` installed natively.

### 1. Build and Run the Stack
Because the latest UI logic heavily relies on container variables, always construct fresh builds using the following flags to bypass cache locks:

```bash
# Pull down any stray containers
docker-compose down

# Force fresh build without legacy cache mechanisms
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker-compose up -d --build
```

### 2. Access the Application
- **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 🌎 Supported Cities

Current hardcoded supported cities from our dataset include:
*Sivas Merkez, Sivas Kuzey, Sivas Güney, Kangal, Zara, Suşehri, Ankara, Kayseri, Erzincan, and Malatya.*

## ⚙️ Development Highlights

- **Frontend**: Context providers handle language distribution, propagating user language selection down through every individual weather component, triggering backend localized API lookups.
- **Models**: Our Scikit-Learn models enforce logic like strict umbrella requirements based on fractional rain probability and strict clothing boundaries using customized temperature threshold variables.
- **LLM Wrapper**: Our `llm_text` pipeline wraps strict parameter constraints on an OpenAI model instance to generate the short, snappy "Hero Advice" text directly related to real conditions in the users chosen language.
