# Async - Investment Risk Prediction System

ML-based portfolio analysis: risk classification (Low/Medium/High), explainable allocation, historical
simulation, scenarios, live tracking, multilingual UI. Educational tool, not financial advice.

## Layout
| Path | Owner | What |
|---|---|---|
| `engine/` | P1 | market data + cache, metrics, optimizer, simulation, scenarios, tracking, Supabase (see `engine/README.md`) |
| `ml/` | P2 | dataset, training, classifier, explanations, LLM / voice / news / i18n (see `ml/README.md`) |
| `app/` | P3 | Streamlit UI. `app/api.py` is the adapter to `engine/` + `ml/` (`USE_MOCKS=true` runs the UI on `data/mocks`) |
| `data/` | | `universe.json`, `fallback/*.parquet` (offline prices), `mocks/*.json` |
| `sql/schema.sql` | | run once in the Supabase SQL editor |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in the keys
python -m pytest tests -q                  # engine + ML + app contract + headless whole-app smoke tests (offline)
python scripts/check_supabase.py           # live Supabase round trip (needs secrets + schema)
python scripts/download_fallback.py        # refresh offline price files before a demo
python -m ml.train                         # retrain + write ml/reports (uploads the model to Supabase Storage)
python scripts/prewarm.py                  # fill the LLM cache for the demo (needs API keys), then --offline-check
python -m streamlit run app/main.py        # the app (real backend; add USE_MOCKS = true to secrets for UI-only dev)
```

## Run modes
* **Real backend (default)**: prices from Yahoo with caching; if the network is down the committed
  `data/fallback/*.parquet` files serve; explanations are templates unless Gemini/Groq keys are configured.
* **Offline mode** (sidebar toggle): zero network AI calls; cached results (`scripts/prewarm.py`) and templates.
* **`USE_MOCKS = true`** in `.streamlit/secrets.toml`: UI development against `data/mocks/*.json`.

## Checks
| Command | What it proves |
|---|---|
| `python -m pytest tests -q` | engine maths, ML pipeline, adapter contracts and the whole app rendering offline |
| `python scripts/smoke_app.py --real --offline --interact` | cycles every scenario / regime / language / benchmark / risk level |
| `python scripts/profile_engine.py` | real-data scenarios + latency budgets |
| `python scripts/check_supabase.py` | live Supabase round trip |
