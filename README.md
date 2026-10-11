# Async - Investment Risk Prediction System

ML-based portfolio analysis: risk classification (Low/Medium/High), explainable allocation, historical
simulation, scenarios, live tracking, multilingual UI. Educational tool, not financial advice.

## Layout
| Path | Owner | What |
|---|---|---|
| `engine/` | P1 | market data + cache, metrics, optimizer, simulation, scenarios, tracking, Supabase (see `engine/README.md`) |
| `ml/` | P2 | dataset, training, classifier, explanations, LLM / voice / news / i18n (see `ml/README.md`) |
| `app/` | P3 | Streamlit UI (build against `data/mocks`, `engine.serialize.load_mock`) |
| `data/` | | `universe.json`, `fallback/*.parquet` (offline prices), `mocks/*.json` |
| `sql/schema.sql` | | run once in the Supabase SQL editor |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in the keys
python -m pytest tests -q                  # offline engine tests
python scripts/check_supabase.py           # live Supabase round trip (needs secrets + schema)
python scripts/download_fallback.py        # refresh offline price files before a demo
python -m ml.train                         # retrain + write ml/reports (uploads the model to Supabase Storage)
python scripts/prewarm.py                  # fill the LLM cache for the demo (needs API keys), then --offline-check
streamlit run app/main.py                  # once app/ exists
```
