# Gati Infrastructure Monitoring – AI/ML Backend

AI/ML backend for infrastructure-project monitoring, prediction, risk analysis, early warning, benchmarking, recommendations, and natural-language project intelligence.

## Overview

This Django/DRF backend combines project data, historical project similarity, ML predictions, and an LLM-powered Project Intelligence Assistant.

### Core capabilities

- Cost Overrun Prediction
- Time / Schedule Delay Prediction
- Project Risk Scoring
- Early Warning System
- Project Benchmarking / Comparative Analytics
- Cost Escalation Driver Analysis
- Prescriptive Recommendations
- LLM Project Intelligence Assistant

## Technology Stack

| Layer         | Technology                            |
| ------------- | ------------------------------------- |
| Backend       | Python, Django, Django REST Framework |
| Database      | Supabase / PostgreSQL                 |
| Vector Search | PGVector / embedding-based retrieval  |
| ML            | Python prediction pipeline            |
| LLM Provider  | OpenRouter                            |
| LLM Model     | `google/gemini-3.6-flash`             |
| Deployment    | Render                                |
| API Testing   | Postman                               |

## Architecture

```text
Project Data
     |
     v
Supabase / PostgreSQL
     |
     v
Django REST API
     |
     v
Historical Similarity / Embeddings
     |
     v
Prediction Engine
   /    |    \
 Cost  Time  Risk
   \    |    /
    +---+---+
        |
        +--> Early Warning
        +--> Recommendations
        +--> Benchmarking
        |
        v
Project Intelligence Assistant
        |
        v
OpenRouter
        |
        v
Gemini 3.6 Flash
        |
        v
AI Project Answer
```

## Workflow

### Existing Project

```text
project_id
   ↓
Database lookup
   ↓
Project preprocessing
   ↓
Historical similar-project retrieval
   ↓
Cost prediction
   ↓
Time-delay prediction
   ↓
Risk scoring
   ↓
Cost escalation analysis
   ↓
Early warning / recommendations / benchmarking
   ↓
Project Assistant context
```

### New Project

Required fields:

```text
project_name
agency
ministry
sector
state
start_date
original_completion_date
original_cost
physical_progress
```

Optional:

```text
project_id
progress_status
revised_cost
revised_completion_date
```

Workflow:

```text
Project Input
     ↓
Validation
     ↓
Project Text
     ↓
Embedding
     ↓
Historical Similar Projects
     ↓
Cost Prediction
     ↓
Time Prediction
     ↓
Risk Assessment
     ↓
Prediction Response
```

## Project Intelligence Assistant

The Project Assistant is LLM-driven and answers natural-language questions using the project record, ML analysis, historical evidence, and optional comparison context.

It does **not** depend on keyword-based intent routing or hardcoded answers.

```text
User Question
     ↓
POST /project-assistant/
     ↓
Load Project
     ↓
Load ML Analysis / Prediction
     ↓
Build AI Context
     ↓
OpenRouter
     ↓
Gemini 3.6 Flash
     ↓
Question-specific Answer
```

Current LLM configuration:

```text
Provider: OpenRouter
Model: google/gemini-3.6-flash
Endpoint: https://openrouter.ai/api/v1/chat/completions
```

## API

### Base URL

```text
https://aiml-77m5.onrender.com/api/aiml/
```

Protected endpoints use:

```http
X-AI-API-KEY: YOUR_AI_API_KEY
```

### Endpoint list

| Method | Endpoint                                   | Purpose                               |
| ------ | ------------------------------------------ | ------------------------------------- |
| GET    | `/health/`                                 | Health check                          |
| GET    | `/predict-project/?project_id=1700`        | Existing project prediction           |
| GET    | `/predict-new-project/`                    | New project prediction                |
| GET    | `/early-warning/?project_id=1700`          | Early warning                         |
| GET    | `/project-recommendation/?project_id=1700` | Recommendations                       |
| GET    | `/project-benchmarking/?project_id=1700`   | Benchmarking                          |
| POST   | `/project-assistant/`                      | Natural-language project intelligence |

### Health

```http
GET /api/aiml/health/
```

Response:

```json
{
  "status": "ok",
  "service": "Django AI/ML Engine"
}
```

### Existing project prediction

```bash
curl -X GET   "https://aiml-77m5.onrender.com/api/aiml/predict-project/?project_id=1700"   -H "X-AI-API-KEY: YOUR_AI_API_KEY"
```

Main response sections:

```json
{
  "project_id": 1700,
  "cost_prediction": {},
  "time_prediction": {},
  "risk": {}
}
```

Cost prediction may contain final cost, overrun percentage, expected range, confidence, historical evidence, similarity, uncertainty warning, and cost-escalation analysis.

Time prediction may contain predicted delay, expected range, planned duration, ML/historical delay signals, historical evidence, similarity, confidence, and warning.

Risk may contain risk level, risk score, issue ID, reason, recommendation, and detected issues.

### New project prediction

```http
GET /api/aiml/predict-new-project/
```

Required query parameters:

```text
project_name
agency
ministry
sector
state
start_date
original_completion_date
original_cost
physical_progress
```

Optional:

```text
project_id
progress_status
revised_cost
revised_completion_date
```

### Early warning

```http
GET /api/aiml/early-warning/?project_id=1700
```

### Recommendation

```http
GET /api/aiml/project-recommendation/?project_id=1700
```

### Benchmarking

```http
GET /api/aiml/project-benchmarking/?project_id=1700
```

### Project Assistant

```http
POST /api/aiml/project-assistant/
Content-Type: application/json
X-AI-API-KEY: YOUR_AI_API_KEY
```

Request:

```json
{
  "project_id": 1700,
  "message": "What is the project risk and why?"
}
```

Response:

```json
{
  "question": "What is the project risk and why?",
  "answer": "...",
  "project_id": 1700
}
```

The public Project Assistant response does not expose provider/model metadata.

## AI/ML Capability Mapping

| Capability                           | Status      |
| ------------------------------------ | ----------- |
| Cost Overrun Prediction              | Implemented |
| Time / Schedule Overrun Prediction   | Implemented |
| Project Risk Scoring                 | Implemented |
| Early Warning System                 | Implemented |
| Benchmarking / Comparative Analytics | Implemented |
| Cost Escalation Driver Analysis      | Implemented |
| Prescriptive / Recommended Action    | Implemented |
| LLM Project Intelligence Assistant   | Implemented |

## Environment Variables

For the Project Assistant:

```env
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_MODEL=google/gemini-3.6-flash
```

`OPENROUTER_MODEL` can be omitted when the application default is used.

Never hardcode API keys or commit them to GitHub.

## Error Handling

|  Status | Meaning                                    |
| ------: | ------------------------------------------ |
|     200 | Successful request                         |
|     400 | Invalid or missing input                   |
| 401/403 | Authentication failure                     |
|     404 | Project not found                          |
|     500 | Internal processing error                  |
|     503 | LLM/provider unavailable or not configured |

Example:

```json
{
  "error": "Project not found.",
  "project_id": 1700
}
```

## End-to-End Flow

```text
User selects project
      ↓
Prediction API
      ↓
Cost + Time + Risk
      ↓
Historical Evidence
      ↓
Early Warning / Benchmarking / Recommendations
      ↓
User asks a question
      ↓
Project Assistant
      ↓
Project + ML Context
      ↓
OpenRouter → Gemini 3.6 Flash
      ↓
Natural-language answer
```

## Deployment Checklist

1. Configure `OPENROUTER_API_KEY` in Render.
2. Optionally configure `OPENROUTER_MODEL`.
3. Keep `X-AI-API-KEY` configured for protected APIs.
4. Deploy the latest backend commit.
5. Test `/health/`.
6. Test `/predict-project/`.
7. Test `/early-warning/`.
8. Test `/project-recommendation/`.
9. Test `/project-benchmarking/`.
10. Test `/project-assistant/`.

## Testing Examples

```json
{
  "project_id": 1700,
  "message": "What is the project risk?"
}
```

```json
{
  "project_id": 1700,
  "message": "Why is the project delayed?"
}
```

```json
{
  "project_id": 1700,
  "message": "What is causing the cost pressure?"
}
```

```json
{
  "project_id": 1700,
  "message": "Give me the overall summary of the project."
}
```

```json
{
  "project_id": 1700,
  "message": "In which state is the project located?"
}
```

## Security

- Never commit API keys to GitHub.
- Keep OpenRouter credentials in Render environment variables.
- Keep application API authentication separate from LLM provider credentials.
- Rotate any provider key that has been exposed.
- Avoid logging full API keys.
- Do not commit secrets in Postman collections.

## Project Structure

```text
aiml_engine/
├── manage.py
├── aiml_engine/
│   ├── settings.py
│   └── ...
└── ai/
    ├── views.py
    ├── services/
    │   └── project_assistant.py
    └── ...
```

## Summary

```text
Project Data
    ↓
Historical Similarity
    ↓
ML Predictions
    ↓
Cost + Time + Risk
    ↓
Early Warning + Benchmarking + Recommendations
    ↓
LLM Project Intelligence
    ↓
OpenRouter
    ↓
Gemini 3.6 Flash
```

### Project Assistant contract

```text
POST /api/aiml/project-assistant/
```

Request:

```json
{
  "project_id": 1700,
  "message": "Your project question"
}
```

Response:

```json
{
  "question": "Your project question",
  "answer": "AI-generated answer",
  "project_id": 1700
}
```

## License

Add the project's applicable license here if one has been defined.
