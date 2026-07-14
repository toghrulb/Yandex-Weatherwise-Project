## Backend

FastAPI service layer for:
- inference APIs
- business logic
- request/response schemas
- model loading and serving

Planned app package location: `backend/app/`

### Quick Start

1. Install dependencies:
   - `pip install -r backend/requirements.txt`
2. Ensure model artifacts exist:
   - `python -m ml.src.models.train_pipeline`
3. Run API:
   - `uvicorn backend.app.main:app --reload`
4. Predict endpoint:
   - `POST /api/v1/predict`

### Optional LLM Text (Hero + Activity Advice)

The backend can generate short LLM comments for:
- hero `recommendation_text`
- activity `activity_advice` (batched in one call)

Set environment variables before starting API:
- `tutor_openai_api=<your_openai_api_key>` (primary key var)
- optional `WEATHERWISE_OPENAI_MODEL=gpt-4o-mini`
- optional `WEATHERWISE_ENABLE_LLM_TEXT=1` (`0` to disable)
- optional `WEATHERWISE_OPENAI_TIMEOUT_S=6.0`

If key/model is unavailable or request fails, the service falls back to deterministic rule-based text.
