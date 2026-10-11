# P3 — FRONTEND ONLY: YOUR TASK BRIEF (READ FIRST)

> You are **P3**. This brief is followed by the full shared project plan. Everything in the shared plan applies to you (zero-cost stack, contracts, brownie points, efficiency rules). Where this brief is more specific about **your** tasks, this brief wins. You build ONLY the Streamlit UI. **No finance math, ML or LLM logic in `app/`:** call P1's `engine/` and P2's `ml/` functions, and render the results. Start on mocks so you are never blocked.

## 0. What you own

| Area | Files |
|---|---|
| App entry, layout, sidebar | `app/main.py`, `.streamlit/config.toml` |
| Backend adapter (mock/real switch) | `app/api.py` |
| UI helpers (charts, badges, formatting) | `app/ui.py`, `app/charts.py` |
| Tabs | `app/tabs/build.py` (Risk), `allocation.py`, `simulate.py`, `track.py`, `compare.py`, `scenarios.py`, `news.py` |
| PDF report export | `app/report.py`, `assets/fonts/NotoSansDevanagari-*.ttf` |
| README, demo script | `README.md`, `DEMO.md` |

## 1. Phase 0 (0:00-0:30)

1. Set up the Streamlit skeleton and `requirements.txt` (shared plan section 10).
2. Create `app/api.py`: one wrapper per backend function (see contracts in the shared plan, plus the new helpers: `compare_growth`, `risk_return_points`, `validate_tickers`, `classify_stressed`, `explain_allocation`). Each wrapper tries the real function; if `USE_MOCKS=True` (or the import fails) it returns the matching JSON from `data/mocks/`. This lets you switch to real functions **one at a time** in Phase 3.
3. Wait for P1/P2 mocks (they commit them in Phase 0); build against the exact shapes in their briefs.

## 2. Global UI requirements

- **Layout:** sidebar for inputs, 7 tabs in the main area: `Risk`, `Allocation`, `Simulate`, `Track`, `Compare`, `Scenarios`, `Market Intel`.
- **Session state:** central keys `lang`, `offline`, `tickers`, `amounts` (or `weights`), `total_investment`, `risk_level`, `start`, `end`, `benchmark`, `rf`. Set widget defaults from state; for voice autofill update state **before** widgets render (use callbacks).
- **Caching:** `@st.cache_data(ttl=3600)` on every backend call, keyed by `tuple(tickers)`, `tuple(weights)`, dates, risk level. `@st.cache_resource` for the ML model loader. Never recompute on unrelated reruns.
- **Auto-recompute (brownie "dynamic adjustment"):** changing risk level, amount, assets or dates updates every tab; avoid recompute storms (use `on_change` callbacks and sliders, show `st.spinner`).
- **Error handling:** every backend call wrapped in try/except; show friendly `st.warning`/`st.error` with a retry hint, never a traceback. Show notices for `df.attrs["source"] == "fallback"` ("using saved data"), dropped tickers, mixed currencies, and LLM offline/cached state.
- **Footer disclaimer:** "Educational tool, not financial advice. Past performance does not guarantee future results." (translated).
- **Multilingual:** use `t(key, lang)` from P2's `ml/i18n.py` for EVERY label (no hardcoded English strings). Language selector: English / हिन्दी / मराठी. Translated explanations come from P2 functions. Check Devanagari renders correctly in all charts and tables.
- **Theme:** `.streamlit/config.toml` with a clean light theme, colorblind-safe palette, consistent risk colors (Low green, Medium amber, High red).
- **Responsive:** usable at laptop width; use `st.columns`, avoid wide tables without scrolling.

## 3. Sidebar spec

1. Language selector + **Offline mode** toggle (calls P2's `set_offline`).
2. **LLM usage panel** (expander) from P2's `usage_stats()`: per provider calls, cache hits, failures.
3. **Voice input** (expander): `st.audio_input` -> language hint from selector -> `transcribe` -> editable `st.text_area` with the transcript -> "Fill portfolio" button -> `extract_portfolio` -> populates tickers, amounts, risk level. Show `unresolved` names and ask the user to fix them. If transcription fails, show "please type instead".
4. **Assets:** multiselect from `universe.json` plus a free-text ticker box (validated with `validate_tickers`; reject bad tickers with a message).
5. **Amounts/weights:** `st.data_editor` with one row per asset (amount in currency, auto-computed %), total investment, normalize to 100%.
6. **Risk level:** radio Low / Medium / High.
7. **Date range** (start/end), **benchmark** select (NIFTY 50 `^NSEI`, NASDAQ-100 `^NDX`), advanced expander for risk-free rate.
8. **Save / Load portfolio** (Supabase via P1's `db.py`) using a `session_name` text box (no login), plus **Export PDF** button.

## 4. Tab specs

**Risk tab (Phase 2):** large risk badge (Low/Medium/High) with class probabilities as a small bar; metric cards (return, volatility, max drawdown, Sharpe, Sortino, CVaR); correlation heatmap; grouped bar chart of **weight vs risk contribution per asset** (highlight assets whose risk share greatly exceeds weight); top-3 factors list; explanation text from `explain_portfolio`.

**Allocation tab (Phase 2):** original vs recommended weights (side-by-side bars + table with deltas), before/after metrics, `reason` + `tradeoffs` text, `explain_allocation` text (explainable recommendations brownie), amount in currency per asset. Changing risk level updates it instantly.

**Simulate tab (Phase 2):** preset regime buttons (COVID crash, 2021 bull, 2022 bear/volatile, last 1y) + custom date pickers; growth-of-100 line chart (log scale toggle); underwater (drawdown) chart; metrics table; warning for dropped tickers; option to test alternate asset combos (second multiselect) and overlay.

**Compare tab (Phase 2):** table for Original / Recommended / Equal Weight / Benchmark (returns, vol, max drawdown, Sharpe, diversification); growth lines from `compare_growth`; **risk-return scatter** from `risk_return_points` (assets, portfolios, benchmark, labeled); metric bar charts; auto text verdict "outperforms/underperforms benchmark on a risk-adjusted basis" based on Sharpe difference; limitations note.

**Track tab (Phase 3):** holdings editor (ticker, quantity, buy price) with Refresh button; table with current price, value, P&L and P&L % (green/red); allocation pie; totals **per currency** (never mixed); current risk label; "as of" timestamp and note that data may be delayed; cache TTL ~5 min.

**Scenarios tab (Phase 4):** scenario picker (market crash -20%/-30%, recovery +15%, volatility x2, sector downturn with sector dropdown and slider); result cards: value before/after, % change, new risk label (via `classify_stressed`), new volatility; Monte Carlo fan chart (5/25/50/75/95 percentile paths), probability of loss, expected shortfall.

**Market Intel tab (Phase 4):** banner "Supplementary AI insight. It does not change the ML risk score."; "Generate insight" button (never auto-call); per-ticker sentiment chips, sector trends, risks, summary, headline links; show `source` (live/cache/offline).

## 5. PDF export (`app/report.py`)

Use `fpdf2` with a bundled **Noto Sans Devanagari** TTF (free, SIL OFL) so Hindi/Marathi render. Include: portfolio table, risk label, key metrics, explanation text, recommended allocation, disclaimer, language-aware headings. For charts use matplotlib PNGs (avoid kaleido, which is heavy on Streamlit Cloud). Offer download via `st.download_button`.

## 6. Phase timeline (aligned with the shared plan)

- **Phase 0-1 (0:00-2:00):** `api.py` with mock switch, sidebar, tab shells, i18n wiring, theme, validation, session state.
- **Phase 2 (2:00-3:30):** Risk, Allocation, Simulate, Compare tabs with plotly (still mocks where backend isn't ready).
- **Phase 3 (3:30-4:15):** swap to real functions tab by tab (pull first, flip one `USE_MOCKS` per function, test, commit); build Track tab, save/load UI.
- **Phase 4 (4:15-5:15):** Scenarios, Market Intel, risk-return visualization polish, voice widget, language switching checks, PDF export, usage panel + Offline toggle.
- **Phase 5 (5:15-6:00):** full test pass, screenshots, `DEMO.md`, record the demo, README with run steps and secrets setup.

## 7. Demo script (write in `DEMO.md`, about 6 minutes)

1. Say a voice command in Hindi: "Mere paas 1 lakh rupaye hain, TCS, Infosys aur HDFC Bank mein lagana hai, risk kam chahiye" -> form auto-fills.
2. Risk tab: label + why (risk contribution, correlation).
3. Allocation tab: switch Low -> High and show weights change with explanation.
4. Simulate: COVID crash vs 2021 bull. Compare: vs NIFTY 50.
5. Scenarios: market crash -20% and Monte Carlo.
6. Market Intel in Marathi. Show usage panel and Offline mode.
7. Save portfolio, export PDF. Close with the free-tier architecture slide.

## 8. Acceptance checklist

- [ ] Every tab works in mock mode by ~2:30 and in real mode by ~4:15
- [ ] No tab ever shows a traceback (simulate failures by disconnecting internet, bad tickers, empty inputs)
- [ ] Single asset, 10 assets, mixed currencies all handled with clear messages
- [ ] Hindi and Marathi render correctly in UI, charts and PDF
- [ ] Offline mode: full demo flow works
- [ ] Reruns are fast (cached); no LLM call without a button click
- [ ] App runs from a clean clone with `streamlit run app/main.py`

## 9. Do NOT

- Do not compute metrics, optimize weights, train models or call LLMs inside `app/` (only call the wrappers). Do not hardcode English strings. Do not sum values across currencies. Do not commit keys. Do not block on backend delays: use mocks.

---

# SHARED PROJECT PLAN (applies to everyone)

# Investment Risk Prediction System — Build Plan (Zero-Cost, 6-Hour Hackathon)

> Instructions for the coding agent: read this whole file first. Build exactly what is described, in the phase order given. Every service used MUST be free-tier with no credit card required. Never hardcode secrets. Prefer simple, working code over clever code. After each phase, run the app and fix errors before moving on.

---

## 1. Goal

An ML-based investment portfolio analysis and risk prediction system with a Streamlit UI. Users build a portfolio, and the system analyses historical and current data, classifies risk (Low/Medium/High), explains why, recommends an improved allocation, compares against benchmarks, simulates history, and tracks live holdings. Brownie points: multilingual (English/Hindi/Marathi), voice input, scenario analysis, AI market intelligence, explainable recommendations.

## 2. Hard Constraint: Everything Free

| Need | Free choice | Notes |
|---|---|---|
| Hosting | Streamlit Community Cloud (or run locally for demo) | ~1 GB RAM, so NO torch / heavy libs |
| DB + Storage | Supabase free tier | Projects pause after ~1 week idle: open the dashboard before the demo to wake it |
| Market data | `yfinance` (no key) | Rate-limited: cache hard, keep a parquet fallback |
| News | `yfinance` `.news` and Google News RSS via `feedparser` (no key) | |
| LLM | **Provider chain, all free, no card:** Google Gemini (AI Studio) -> Groq (Llama 3.3 70B) -> Mistral (`mistral-small-latest`, free Experiment tier) | Verify current limits and signup rules. Each provider has an independent quota. Cache every output (section 9) |
| Speech-to-text | **Groq Whisper** (free tier) as primary, `st.audio_input` to record. Fallback: `faster-whisper` small model locally | Hindi good, Marathi weaker: let the user correct the transcript text before submit |
| Translation | The LLM itself (answer in selected language) | No separate translation API |
| Embeddings (optional) | Gemini embedding API (free tier), stored in Supabase pgvector | Skip if short on time |
| ML | scikit-learn, LightGBM, scipy | All free, CPU only |
| Charts | plotly | |

Do NOT use: Claude/OpenAI paid APIs, paid NewsAPI, Sarvam paid plans, GPU services, or any service requiring a credit card.

Secrets (in `.streamlit/secrets.toml` locally and Streamlit Cloud secrets; add to `.gitignore`):
```
SUPABASE_URL = ""
SUPABASE_KEY = ""     # anon key
GEMINI_API_KEYS = [""]    # list; a teammate may add their OWN key as backup
GROQ_API_KEYS = [""]
MISTRAL_API_KEYS = [""]
```
Keys needed: Supabase, Gemini, Groq, Mistral (all free). Core ML and finance engine need NO key and make ZERO LLM calls.
Note: several keys from the same account/project share ONE quota, so spread load across providers, not duplicate keys. Never create throwaway accounts to get more quota (violates provider terms, risks a ban before the demo).

## 3. Team Split (3 people)

- **P1: Finance engine (backend):** `engine/` + Supabase data layer.
- **P2: ML + AI (backend):** `ml/` + LLM, STT, translation.
- **P3: Frontend only:** `app/` Streamlit UI. Builds against mock JSON first.

## 4. Repo Structure

```
.
├── PLAN.md
├── requirements.txt
├── scripts/prewarm.py         # fills llm_cache for the demo (section 9.3)
├── .gitignore                 # secrets.toml, .env, __pycache__, data/cache
├── .streamlit/secrets.toml    # NOT committed
├── data/
│   ├── universe.json          # demo tickers (NSE + US)
│   ├── fallback/              # pre-downloaded parquet price files
│   └── mocks/                 # mock_*.json for P3
├── engine/                    # P1
│   ├── data.py                # get_prices, cache, fallback
│   ├── metrics.py             # portfolio_metrics, risk_contribution
│   ├── optimizer.py           # optimize
│   ├── simulate.py            # simulate, regimes, scenarios, monte carlo
│   ├── track.py               # live holdings tracking
│   └── db.py                  # Supabase client + price_cache + portfolios
├── ml/                        # P2
│   ├── dataset.py             # random portfolio generator + labels
│   ├── train.py               # train + evaluate + save model
│   ├── classify.py            # classify_risk
│   ├── explain.py             # feature importance + plain-text reasons
│   ├── llm.py                 # Gemini / Groq wrapper with cache + fallback
│   ├── news.py                # get_news_insight
│   ├── voice.py               # transcribe + extract portfolio JSON
│   └── i18n.py                # labels, glossary, language handling
└── app/                       # P3
    ├── main.py
    └── tabs/ build.py, risk.py, allocation.py, simulate.py, track.py, compare.py, scenarios.py, news.py
```

## 5. Function Contracts (do not change without telling the team)

```python
# engine/data.py
get_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame
# adjusted close, columns=tickers, index=date. Uses Supabase price_cache, fetches only missing days via ONE batched yf.download, falls back to data/fallback parquet on failure.

# engine/metrics.py
portfolio_metrics(prices: pd.DataFrame, weights: dict) -> dict
# keys: ann_return, ann_vol, max_drawdown, sharpe, sortino, calmar, cvar_95, avg_corr, hhi, effective_n
risk_contribution(prices, weights) -> pd.Series   # % contribution to portfolio vol per asset
return_contribution(prices, weights) -> pd.Series

# engine/optimizer.py
optimize(prices, risk_level: str, bounds=(0.05, 0.40)) -> dict
# {"weights": {...}, "metrics": {...}, "reason": str, "tradeoffs": str}

# engine/simulate.py
simulate(prices, weights, start, end) -> {"growth": pd.Series, "metrics": dict}
regime_presets() -> dict       # named historical windows
apply_scenario(prices, weights, scenario: dict) -> dict   # impact on value, return, risk
monte_carlo(prices, weights, n=1000, horizon=252) -> dict
compare_strategies(prices, original_w, recommended_w, benchmark: str) -> pd.DataFrame

# engine/track.py
track_holdings(holdings: list[dict]) -> {"table": pd.DataFrame, "total_value": float, "total_pnl": float}

# ml/classify.py
classify_risk(prices, weights) -> {"label": "Low|Medium|High", "probs": dict, "top_factors": list}

# ml/explain.py
explain_portfolio(prices, weights, label, lang="en") -> str

# ml/news.py
get_news_insight(tickers: list[str], lang="en") -> {"sentiment": str, "summary": str, "risks": list, "headlines": list}

# ml/voice.py
transcribe(audio_bytes, lang_hint="hi") -> str
extract_portfolio(text: str) -> {"tickers": [...], "amounts": {...}, "risk_level": str}
```

## 6. Supabase Schema (run in SQL editor)

```sql
create extension if not exists vector;

create table portfolios (
  id uuid primary key default gen_random_uuid(),
  session_name text, name text, risk_level text, language text default 'en',
  created_at timestamptz default now()
);
create table holdings (
  id bigserial primary key,
  portfolio_id uuid references portfolios on delete cascade,
  ticker text, amount numeric, quantity numeric, weight numeric
);
create table price_cache (
  ticker text, date date, adj_close double precision,
  primary key (ticker, date)
);
create table analysis_runs (
  id bigserial primary key,
  portfolio_id uuid references portfolios on delete cascade,
  risk_label text, metrics jsonb, weights jsonb,
  created_at timestamptz default now()
);
create table news_insights (
  id bigserial primary key,
  ticker text, lang text, sentiment text, summary text, risks jsonb,
  created_at timestamptz default now()
);
create table llm_cache (
  key text primary key,          -- sha256(task + prompt + lang)
  task text, lang text, response text,
  provider text, created_at timestamptz default now()
);
-- optional (pgvector brownie): dimension MUST match the embedding model used
create table news_embeddings (
  id bigserial primary key, ticker text, headline text,
  embedding vector(768)
);
-- Hackathon simplicity: no auth. Use permissive RLS policies for the anon key and a session_name field instead of login.
```

Supabase Storage bucket `models` (private): `risk_model.joblib`, `feature_meta.json`. Bucket `reports`: exported PDFs. Free tier is ~500 MB DB and ~1 GB storage, which is plenty.

pgvector decision: NOT needed for core features. Only build it (Phase 5, optional) for semantic search/dedup over news headlines. If skipped, no harm.

## 7. Compulsory Requirements (detailed)

### 7.1 Inputs
- Select or type multiple tickers (NSE `.NS`, BSE `.BO`, US, indices). Validate each ticker via a quick price fetch; show a clear error for bad tickers.
- Amount per asset OR percentage weights (normalize to 100%). Total investment amount.
- Risk preference: Low / Medium / High.
- Date range selector.

### 7.2 Historical performance and risk
Use log returns on adjusted close, 252 trading days. Compute: annualized return, volatility, max drawdown, Sharpe, Sortino, Calmar, CVaR 95%, correlation matrix, HHI, effective N. Use **Ledoit-Wolf shrinkage covariance** (`sklearn.covariance.LedoitWolf`) everywhere covariance is needed. Risk-free rate configurable (default 6.5% for INR, 4% for USD).

### 7.3 ML risk classification (Low/Medium/High)
1. Generate ~5,000-10,000 random portfolios (3-10 assets, Dirichlet weights) from a 30-60 stock universe over rolling windows.
2. **Features (from the lookback window only):** annualized return, vol, max drawdown, avg pairwise correlation, HHI, effective N, top-asset weight, skew, kurtosis.
3. **Labels from FUTURE data:** next 6 months realized volatility and drawdown, converted to Low/Medium/High using tercile thresholds. This avoids leakage and a model that just memorizes a formula.
4. **Time-based train/test split** (train earlier dates, test later). Models: RandomForest baseline, LightGBM main. Report macro-F1 and confusion matrix; save to `ml/reports/`.
5. Save model to Supabase Storage and a local copy; load once with `@st.cache_resource`.
6. If the model file is missing, retrain automatically from cached data.

### 7.4 Explainability
- Per-asset risk contribution: `RC_i = w_i * (Σw)_i / σ_p`. Show as bar chart + table, flag any asset with contribution far above its weight.
- Model-level factors: LightGBM feature importance, or SHAP TreeExplainer on the single portfolio (SHAP is optional; fall back to importance).
- Plain-text reason, e.g. "Rated High because volatility is 28%, 62% of risk comes from X, and average correlation is 0.71 (low diversification)." Template-based first; LLM rewrite only for translation.

### 7.5 Allocation optimizer
`scipy.optimize` SLSQP with per-asset bounds (default 5%-40%), weights sum to 1, long-only.
- Low: minimum variance.
- Medium: maximum Sharpe.
- High: maximize return subject to a volatility cap (e.g. 1.3x equal-weight vol).
Return weights, metrics, and a written trade-off explanation (what return you give up for lower risk, which assets were reduced/increased and why).

### 7.6 Historical simulation
Date range + asset combination test. Show growth of ₹100 chart, total return, vol, max drawdown, Sharpe. Provide regime presets: 2020 COVID crash, 2021 bull run, 2022 bear/volatile, plus custom range.

### 7.7 Live tracking
Input holdings (ticker, quantity, buy price). Show current price (latest available, note data may be delayed), current value, gain/loss per asset and total, allocation pie, current portfolio risk label.

### 7.8 Benchmark comparison
NIFTY 50 (`^NSEI`) or NASDAQ-100 (`^NDX`), same period. Compare returns, vol, max drawdown, Sharpe, growth chart. State clearly whether the portfolio outperforms on a risk-adjusted basis.

### 7.9 Strategy comparison
Original vs system-recommended vs equal-weight vs benchmark. One table + one growth chart + risk-return scatter. Add a "limitations" note (past performance is not a guarantee).

## 8. Brownie Points (all included)

1. **Dynamic adjustment:** all inputs in the sidebar; Streamlit reruns automatically; use `@st.cache_data` so reruns are fast.
2. **Risk-return visualization:** scatter of individual assets, original portfolio, recommended portfolio, benchmark; period selector across bull/bear/volatile presets.
3. **Market scenario analysis:** crash (-20%/-30%), recovery (+15%), volatility x2, sector-specific drop (-30% on chosen sector). Show new value, return, and risk label. Add Monte Carlo (1,000 paths) with 5th/50th/95th percentile fan chart.
4. **AI market intelligence (supplementary):** pull headlines from yfinance news + Google News RSS, send to Gemini for sentiment (Positive/Neutral/Negative), sector trends, risks. Label clearly "supplementary, not part of ML risk prediction". Cache in `news_insights`.
5. **Explainable recommendations:** show why each weight changed (volatility, correlation, risk contribution, diversification), as text + before/after bar chart.
6. **Multilingual (English / Hindi / Marathi):** language selector in the sidebar. All numbers and ML stay in English internally; only UI labels (`i18n.py` dictionary), explanations and news summaries are translated by the LLM with a fixed glossary (volatility, drawdown, Sharpe ratio, diversification, benchmark).
7. **Voice input:** `st.audio_input` -> Groq Whisper -> editable transcript -> Gemini extracts JSON `{tickers, amounts, risk_level}` -> auto-fills the sidebar form. Example: "Mere paas 50,000 rupaye hain, TCS aur Infosys mein lagana hai, risk kam chahiye."
8. **Save/load portfolios** in Supabase and **export PDF report** (use `reportlab` or `fpdf2`; both free).

## 9. LLM Wrapper Rules (`ml/llm.py`)

- Single function `ask_llm(prompt, system, json_mode=False, lang="en")`.
- Provider chain `[gemini, groq, mistral]`, each with its own key list. Prefer the provider with the most headroom (track per-provider 429 counts and a cooldown timestamp). On 429/5xx/timeout: retry once with exponential backoff, mark that provider cooling down for ~60 s, then move to the next provider. Rotate to the next key within a provider only if it belongs to a different account/project.
- Check `llm_cache` BEFORE any network call (also use `st.cache_data` in-session).
- Hash `(prompt, lang)` and cache results (Supabase table or `st.cache_data`) so repeat clicks cost zero requests.
- Call the LLM ONLY on explicit button click, never on every rerun.
- If every provider fails, return the template-based English explanation (or the cached result) and show a small notice. The app must never crash because of the LLM.
- Mistral is weaker on Hindi/Marathi than Gemini: keep it last in the chain for non-English tasks, and test Marathi output quality before trusting it.
- Voice-to-JSON prompt: instruct the model to output ONLY valid JSON, map company names to tickers using `data/universe.json`, default currency INR, and return `null` for missing fields.

### 9.1 Call budget (keep LLM usage minimal)
- LLM is used ONLY for: (1) news sentiment summary, (2) translating explanations when language is not English, (3) voice transcript -> JSON, (4) optional "rewrite nicely". Everything else is local code. Voice transcription itself is a Whisper call (Groq), not an LLM call.
- Batch ALL headlines into ONE news call, never one call per headline.
- Translate ONE combined explanation per analysis run, not each paragraph separately.
- English explanations use templates with no LLM call.
- Limit max output tokens and ask for compact JSON.
- Cache key = `sha256(task + prompt + lang)`, independent of provider, so a hit works whichever provider produced it.

### 9.2 UI safeguards and key-free fallbacks
- Sidebar usage panel: per provider this session show calls, cache hits and failures.
- **Offline mode** toggle: forces templates and cached results only, makes zero network LLM calls. The whole demo flow must work with it on.
- Key-free fallbacks: `faster-whisper` (small model) for voice when running locally; English template explanations everywhere; optional local Ollama only when running on the presenter's laptop (Streamlit Cloud cannot host it).
- Never show a raw error page for quota problems; show a short notice and the cached or template result.

### 9.3 Pre-warm the demo cache (`scripts/prewarm.py`)
- Run every demo portfolio x 3 languages (en/hi/mr) x (explanation, news insight, scenario text) and save the responses in `llm_cache`.
- Include 2-3 pre-recorded voice samples (Hindi, Marathi, English) with their transcripts and extracted JSON, so the voice demo works even if STT or the LLM is down.
- Run it at 5:40 and again just before the demo. Then verify by switching Offline mode ON and clicking through the full demo flow: every screen must render.

## 10. Efficiency Rules

- ONE batched `yf.download(tickers, ...)` call; never loop per ticker.
- `price_cache` in Supabase: fetch only missing dates; retry with backoff; fall back to `data/fallback/*.parquet`.
- Vectorized numpy/pandas only (no per-asset Python loops in metrics).
- `@st.cache_data(ttl=3600)` for data/metrics, `@st.cache_resource` for the model.
- Pre-download the demo universe and pre-train the model BEFORE the demo.
- Keep `requirements.txt` light: streamlit, yfinance, pandas, numpy, scipy, scikit-learn, lightgbm, plotly, supabase, feedparser, google-generativeai (or `google-genai`), groq, mistralai, fpdf2, joblib, shap (optional). No torch/tensorflow.

## 11. Accuracy Rules

- Adjusted close and log returns; align dates across markets (inner join) and warn about mixed currencies (compare in % terms).
- No look-ahead leakage anywhere: features use only past data, labels use only future data, split by date.
- Ledoit-Wolf covariance; bounds on weights; handle singular matrices and single-asset portfolios.
- Minimum history check (e.g. 1 year); warn when a ticker has too little data.
- Show a disclaimer: educational tool, not financial advice.

## 12. Phased Timeline (6 hours)

### Phase 0 (0:00-0:30) — everyone
Create repo + structure, create Supabase project and run schema SQL, get Gemini, Groq and Mistral keys (each teammate may add their own Gemini key to the list), commit `requirements.txt`, agree on the contracts in section 5. P1 and P2 drop `data/mocks/*.json` immediately so P3 can start.

### Phase 1 (0:30-2:00)
- **P1:** `data.py` (batched fetch + Supabase cache + fallback), `metrics.py`, `risk_contribution`, benchmark loader, pre-download universe to parquet.
- **P2:** `dataset.py` (random portfolios, features, future labels), start `llm.py` wrapper (provider chain Gemini/Groq/Mistral + `llm_cache` + usage counters, section 9).
- **P3:** app skeleton: sidebar inputs, language selector, tab layout, wired to mocks.

### Phase 2 (2:00-3:30)
- **P1:** `optimizer.py`, `simulate.py`, `compare_strategies`.
- **P2:** `train.py` (RF + LightGBM, time split, report), `classify.py`, `explain.py`, upload model to Storage.
- **P3:** Risk, Allocation, Simulate, Compare tabs with plotly charts (still on mocks where needed).

### Phase 3 (3:30-4:15) — integration
P3 swaps mocks for real functions tab by tab. P1 builds `track.py` + `db.py` save/load; P3 builds Track tab and save/load UI. P1/P2 fix bugs reported by P3 immediately.

### Phase 4 (4:15-5:15) — brownie points
- **P1:** `apply_scenario`, `monte_carlo`, regime presets.
- **P2:** `news.py`, `voice.py`, `i18n.py` glossary + translation, explainable recommendation text.
- **P3:** Scenarios tab, News tab, risk-return scatter, mic widget, language switching, PDF export button, sidebar LLM usage panel + Offline mode toggle.

### Phase 5 (5:15-6:00)
- 5:15-5:40: full test pass (see section 13). Optional pgvector news search only if everything above is green.
- 5:40-6:00: run `scripts/prewarm.py`, verify the demo in Offline mode, freeze code, record demo, prepare talking points (methodology, leakage avoidance, explanations, free-tier architecture).

**Cut order if behind schedule:** pgvector -> Monte Carlo -> PDF export -> SHAP -> sector scenario -> voice. Never cut a compulsory item.

## 13. Test Checklist

- [ ] Indian-only, US-only, and mixed-currency portfolios
- [ ] Single-asset portfolio, 10-asset portfolio, bad ticker, short-history ticker
- [ ] Weights not summing to 100%
- [ ] Each of Low/Medium/High optimizer outputs differ sensibly
- [ ] Classifier confusion matrix saved; no leakage
- [ ] Benchmark comparison for NIFTY 50 and NASDAQ-100
- [ ] yfinance failure -> fallback parquet works
- [ ] Gemini failure -> Groq -> Mistral -> template fallback (simulate by using a bad key)
- [ ] Second call with the same input is a cache hit (usage panel shows 0 new calls)
- [ ] Offline mode ON: full demo flow works with no network LLM calls
- [ ] `prewarm.py` fills the cache for all demo portfolios in en/hi/mr
- [ ] Mistral output checked for Hindi/Marathi quality
- [ ] Hindi and Marathi UI + explanation render correctly (fonts OK)
- [ ] Voice: Hindi, Marathi, English sample inputs produce valid JSON
- [ ] Supabase save/load round trip; Supabase awake before demo
- [ ] App runs from a clean clone using only README steps and secrets

## 14. Definition of Done

All sections 7.1-7.9 working on real data; all brownie points in section 8 present (or consciously cut per the cut order); app deployed on Streamlit Community Cloud or runnable locally in one command (`streamlit run app/main.py`); zero paid services; no secrets in git.
