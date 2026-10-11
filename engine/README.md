# engine/ - P1 hand-off (finance engine + data layer)

Pure python, no Streamlit. Every function degrades gracefully; nothing here calls an LLM.
Run tests: `python -m pytest tests -q` (offline). Real-data check + latency: `python scripts/profile_engine.py`.

## Quick reference

| Function | Returns |
|---|---|
| `data.get_prices(tickers, start, end, align="inner", min_rows=None)` | DataFrame (date index, ticker columns). **`end` is inclusive.** |
| `data.validate_tickers(list)` | `{"valid": [...], "invalid": [...]}` |
| `data.get_benchmark(name, start, end)` | adjusted-close Series (`"NIFTY 50"`, `"NASDAQ-100"`, `^NSEI`, `^NDX`) |
| `metrics.portfolio_metrics(prices, weights, rf=None)` | 10 keys: ann_return, ann_vol, max_drawdown, sharpe, sortino, calmar, cvar_95, avg_corr, hhi, effective_n |
| `metrics.risk_contribution / return_contribution(prices, weights)` | Series summing to 1 |
| `metrics.feature_inputs(prices, weights)` | **for P2**: ann_return, ann_vol, max_drawdown, avg_corr, hhi, effective_n, top_weight, skew, kurtosis |
| `optimizer.optimize(prices, risk_level, bounds=(0.05,0.40), current_weights=None)` | weights, metrics, reason, tradeoffs, mode (+ before_weights, before_metrics, bounds_used) |
| `simulate.simulate(prices, weights, start, end)` | growth (Series, base 100), metrics, dropped_tickers |
| `simulate.regime_presets()` | `{name: (start, end)}` |
| `simulate.compare_strategies(prices, orig_w, rec_w, benchmark)` | DataFrame, rows Original/Recommended/Equal Weight/Benchmark |
| `simulate.compare_growth(...)` | DataFrame, date index, one column per strategy, base 100 |
| `simulate.risk_return_points(prices, {name: weights}, benchmark)` | `{name: {"ret","vol","kind"}}` |
| `simulate.apply_scenario(prices, weights, scenario)` | before, after, value_change_pct, stressed_prices (+ asset_shocks, betas, warnings) |
| `simulate.monte_carlo(prices, weights, n=1000, horizon=252)` | percentiles p5..p95 (lists, base 100), prob_loss, expected_shortfall, horizon |
| `track.track_holdings(holdings)` | table, total_value / total_cost / total_pnl (dicts **per currency**), allocation, risk_metrics, as_of, warnings |

## Things that go beyond the original brief (please read)

* **`df.attrs`** on `get_prices` output: `source` ("live" | "cache" | "fallback"), `dropped`, `dropped_reasons`,
  `currencies`, `mixed_currency`, `warnings`, `as_of`. Read them right after the call.
* **`optimize(..., current_weights=...)`**: pass the user's portfolio so the explanation compares against it
  (otherwise equal weight). `mode` is one of `min_variance`, `max_sharpe`, `max_return_vol_cap`, `fallback_equal`.
  `reason`/`tradeoffs` are plain-English template strings, ready for P2 to translate.
* **`compare_strategies`** has an extra `total_return` column. `benchmark` may be a name or a price Series (cheaper).
* **`align="outer"`** keeps NaN where an asset has no data (use for regime windows, then pass to `simulate`,
  which drops assets missing in the window and lists them in `dropped_tickers`).
* **`apply_scenario`**: `stressed_prices` = history + one synthetic final-day move (feed to P2's `classify_risk`).
  `vol_multiplier` rebuilds the last 60 rows instead; its `value_change_pct` is the worst point of that window
  (`value_change_basis` says which definition applies). Betas are vs NIFTY 50 for INR assets, NASDAQ-100 for USD.
* **Serialization**: `engine.serialize.load_mock("metrics" | "optimize" | "simulate" | "track" | "compare" |
  "compare_growth" | "risk_return" | "scenario" | "monte_carlo")` returns real pandas objects from `data/mocks`.

## db.py (Supabase + local fallback)

`get_client`, `health` (`{configured, reachable, schema_ready}`), `save_portfolio`, `list_portfolios`, `load_portfolio`, `save_run`, `upload_model`,
`download_model(name, dest_dir=None)`, `llm_cache_get(key) -> {"response","provider","task","lang"} | None`,
`llm_cache_put(key, task, lang, response, provider)`. Credentials: env `SUPABASE_URL` / `SUPABASE_KEY` or
`.streamlit/secrets.toml`. Unreachable Supabase => local JSON under `data/cache/`, never an exception.
`price_cache` is a second-level cache; the first level is a local parquet cache in `data/cache/prices/`.
The holdings table has no buy-price column, so `buy_price` is reconstructed as `amount / quantity` on load.

## Verification

* `python -m pytest tests -q` - 43 offline tests (also green on pandas 2.3 / numpy 2.2).
* `python scripts/profile_engine.py` - real-data scenarios + latency budgets + offline fallback.
* `python scripts/check_supabase.py` - live round trip of every Supabase path (cleans up after itself).
* Weight keys match price columns case-insensitively (`aapl` == `AAPL`).

## Accuracy notes / talking points

* Ledoit-Wolf shrunk covariance everywhere; log returns; 252-day annualisation; rf 6.5% INR / 4% USD (weighted if mixed).
* Optimizer shrinks expected returns 50% toward the cross-sectional mean (raw means are the noisiest input).
  The max-Sharpe objective is exactly the reported Sharpe, and if the 50% solution scores below equal weight on
  that Sharpe the shrinkage is relaxed (25%, then 0%) and the reason says so, so a recommendation is never
  worse than equal weight on the numbers shown. `return_shrinkage` reports the level used.
* Cache re-fetches a week of overlap and does a full refresh if overlapping adjusted closes changed (split/dividend).
* Monte Carlo = moving-block bootstrap (10-day blocks), seeded; history's drift is included, so it is not a forecast.
