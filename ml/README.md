# ml/ - P2 hand-off (ML + AI)

No Streamlit imports. Offline mode, usage counters and the loaded model are module-level state.
Tests: `python -m pytest tests/test_ml.py -q` (hermetic, no API keys needed).

## For P3 - what to call

| Need | Call | Returns |
|---|---|---|
| Risk label | `classify.classify_risk(prices, weights)` | `{"label","probs","top_factors":[{"name","value","impact"}],"features","window_rows","note"}` |
| Re-label a stressed portfolio | `classify.classify_stressed(apply_scenario(...)["stressed_prices"], weights)` | same + `"stressed": True` |
| Why this label | `explain.explain_portfolio(prices, w, label, lang)` / `explain_portfolio_ex(...)` | text / `{"text","lang","translated","fallback"}` |
| Why weights changed | `explain.explain_allocation(prices, orig_w, opt_w, risk_level, lang, tradeoffs=opt["tradeoffs"])` | text (one line per change > 3 pts + trade-off) |
| Scenario text | `explain.explain_scenario(result, label_before, label_after, lang)` | text |
| News | `news.get_news_insight(tickers, lang)` | `{"sentiment","per_ticker","sector_trends","risks","summary","headlines","source","ai_available"}` |
| Voice -> text | `voice.transcribe_ex(audio_bytes, lang_hint)` | `{"text","provider","error"}` (`error`: `empty_audio`/`offline`/`stt_failed`) |
| Text -> portfolio | `voice.extract_portfolio(text)` | `{"tickers","amounts","risk_level","unresolved","confidence","source"}` |
| Saved demo utterances | `voice.demo_samples()` | 20 utterances + expected output (works with STT/LLM down) |
| UI strings | `i18n.t(key, lang)`, `i18n.missing_keys(lang)` | falls back to English, then the key |
| Sidebar usage panel | `llm.usage_stats()` | `{provider: {"calls","cache_hits","failures"}}` |
| Offline toggle | `llm.set_offline(bool)` | zero network LLM calls; cache + templates only |

Rules: call anything that may hit an LLM (`*_ex` with `lang != "en"`, `get_news_insight`, `extract_portfolio`)
**only from a button click**, never on every rerun. Wrap `classify.load_model()` in `@st.cache_resource` if you like
(it is already cached per process). English explanations never call an LLM. Mocks: `data/mocks/mock_{classify,
explain,news,voice,i18n_sample}.json` (exact shapes, enforced by `tests/test_ml.py`).

## What was built

* `features.py` - `build_features` (32 features: 12 base from `engine.metrics.feature_inputs` + short-term,
  structure and market-regime groups; trailing 252 returns; market features are NaN-safe).
* `dataset.py` - rolling windows (lookback 252, forward 126, step 21), 3-10 asset Dirichlet portfolios,
  labels from the future window only (fixed volatility thresholds), purged time split.
* `train.py` - RandomForest baseline + LightGBM, early stopping on the end of the training period,
  honest baselines + window-level bootstrap, report in `ml/reports/`, model in `ml/models/` and Supabase Storage.
* `classify.py` - load order local -> Storage -> quick retrain; factors by occlusion (no SHAP).
* `llm.py` - Gemini -> Groq -> Mistral over plain REST (no vendor SDKs), cache, cooldowns, offline mode.
* `explain.py`, `news.py`, `voice.py`, `i18n.py`, `scripts/prewarm.py`.

## Honest results (v2: 32 features, forward-volatility label, widened universe)

**Label** (what "risk" means): the portfolio's realised annualised volatility over the next 126 trading days,
cut at fixed thresholds **Low < 12% <= Medium < 17% <= High** (rounded terciles of the earliest 40% of history,
then frozen; 15%/25% was tried and rejected because only 9% of portfolios, 4% in the latest period, were High).
**Universe**: 46 stocks + 12 ETFs, including bond / cash / gold ETFs so that all three classes exist.

Test period 2022-10 .. 2026-04 (126-day purge, 32 windows, 2,016 portfolios; class mix 35 / 41 / 25%):

| | macro-F1 | balanced acc. | accuracy | within one class | quadratic kappa |
|---|---|---|---|---|---|
| majority class | 0.13 | 0.33 | 0.25 | 0.66 | 0.00 |
| rule: trailing 1-year volatility | **0.61** | 0.61 | 0.61 | 0.98 | 0.61 |
| rule: EWMA volatility | 0.57 | 0.58 | 0.57 | 0.95 | 0.52 |
| LightGBM (shipped) | 0.60 | 0.63 | 0.60 | 0.95 | 0.58 |
| RandomForest | 0.64 | 0.65 | 0.64 | 0.97 | 0.63 |

Equity-heavy subset (>= 80% stocks, n = 1,176, where bond/cash ETFs cannot make "Low" easy):
LightGBM 0.55 macro-F1 vs 0.56 for the 1-year rule.

**Read this honestly.** Macro-F1 rose from ~0.50 to ~0.60 and 95% of predictions are within one class, but the
gain comes from a better *label* (forward volatility only; drawdown was noise), a richer universe and
interpretable thresholds - **not** from the model: the one-line trailing-volatility rule scores the same
(bootstrap 95% CI of LightGBM minus rule includes 0). The model's value is calibrated probabilities, factor
explanations and re-labelling stressed portfolios. We did not pick RandomForest for scoring higher here: the
model type was fixed from the walk-forward folds before this test period was examined.

### Accuracy experiment (`scripts/experiment_v2.py`, walk-forward, 3 design folds + 1 untouched holdout)

Same regularised LightGBM for every variant; thresholds fitted per fold on training data only.

* **Drop drawdown from the label**: best model macro-F1 0.58 -> 0.62 (design folds). Drawdown is noisy.
* **Short-term features** (21/63/126-day volatility, EWMA, vol-of-vol, structure, market regime) lift the
  classifier from ~0.52 to ~0.62 on the design folds (2019-2023, includes COVID and 2022) but **not on the
  calm 2024-26 holdout**, where the plain 1-year rule is as good. The edge is regime dependent.
* A one-line EWMA-volatility rule matches the learned model on the design folds (0.616 vs 0.616-0.621).
* Predicting log forward volatility and binning it (instead of 3-class) is equivalent (0.62 vs 0.62).
* 63-day vs 126-day horizon: no meaningful difference for the models (0.59-0.61 vs 0.59-0.62).
* Absolute 15%/25% thresholds: classes badly unbalanced (54 / 37 / 8%).
* Realistic ceiling here: macro-F1 about 0.55-0.65, within-one-class 93-98%. "0.8+" would mean leakage.

## Known limits

* Provider model names drift (Gemini 2.5 closed to new users; Groq dropped Llama 3.3 70B). Defaults were verified
  live on 2026-10-11 (Gemini `gemini-flash-lite-latest` -> `gemini-3.5-flash`; Groq `openai/gpt-oss-120b` ->
  `gpt-oss-20b` -> `qwen3.8-27b`; Whisper `whisper-large-v3-turbo`) and are configurable (`GEMINI_MODEL`,
  `GROQ_MODEL`, `MISTRAL_MODEL`, `WHISPER_MODEL` in secrets or env). Mistral was never tested (no key).
* Hindi/Marathi strings and glossary are a careful first pass: have a native speaker review them.
* The 20 voice samples were written alongside the rule-based parser, so they are a regression suite, not an
  independent accuracy estimate. Real recorded audio was not tested.
* Prewarmed cache keys contain numbers from today's prices: re-run `scripts/prewarm.py` on the day.
* Data glitches: `engine.data` blanks transient bad ticks and reports unadjusted splits (e.g. TMPV.NS on
  2025-10-14); `ml.dataset.repair_breaks` treats such breaks as corporate actions for training only.
