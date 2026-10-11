"""Acceptance + regression tests for the ML/AI layer (P2). Hermetic: synthetic data, scripted fake HTTP for
the LLM / Whisper, temporary caches, no Supabase. Run:  python -m pytest tests/test_ml.py -q
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data, db  # noqa: E402
from engine import metrics as M  # noqa: E402
from engine import simulate  # noqa: E402
from ml import classify, dataset, explain, features, i18n, llm, news, train, voice  # noqa: E402

DEFAULT_MODELS = {"gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"], "groq": ["llama-3.3-70b-versatile"],
                  "mistral": ["mistral-small-latest"]}


# --------------------------------------------------------------------------- fixtures / helpers
def synth_prices(n=1500, k=8, seed=0, start="2016-01-04") -> pd.DataFrame:
    """Prices with market-wide volatility regimes, so risk is (somewhat) persistent like real markets."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.uniform(0.6, 2.0, n // 100 + 1), 100)[:n]
    market = rng.normal(0, 0.008, n) * regime
    cols = {}
    for i in range(k):
        r = 0.0003 + rng.uniform(0.5, 1.5) * market + rng.normal(0, rng.uniform(0.004, 0.015), n) * np.sqrt(regime)
        cols[f"S{i}.NS"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range(start, periods=n))


class FakeNet:
    """Scripted replacement for ``llm._http_post``: responses are popped per provider (default: 200 OK)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []                # (provider, model-or-url)
        self.script: dict[str, list] = {"gemini": [], "groq": [], "mistral": []}
        self.text = {"gemini": "G-answer", "groq": "Q-answer", "mistral": "M-answer"}

    @staticmethod
    def provider_of(url: str) -> str:
        return "gemini" if "generativelanguage" in url else "groq" if "groq.com" in url else "mistral"

    def __call__(self, url, headers, body, timeout=15.0):
        p = self.provider_of(url)
        self.calls.append((p, url))
        if self.script[p]:
            status = self.script[p].pop(0)
            if isinstance(status, tuple):                      # (200, "custom text")
                status, text = status
            else:
                text = self.text[p]
            if status != 200:
                return status, {"error": {"message": "x"}}
        else:
            text = self.text[p]
        if p == "gemini":
            return 200, {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return 200, {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "LOCAL_DIR", tmp_path / "local")
    monkeypatch.setattr(db, "_read_secrets", lambda: (None, None))
    monkeypatch.setattr(db, "_client", None)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "prices")
    monkeypatch.setattr(data, "SYNC_TO_SUPABASE", False)
    monkeypatch.setattr(data, "_sleep", lambda s: None)
    monkeypatch.setattr(data, "_download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    data._tail_tried.clear()
    data._head_tried.clear()
    monkeypatch.setattr(llm, "_config", {"keys": {"gemini": ["g1"], "groq": ["q1"], "mistral": ["m1"]},
                                         "models": {k: list(v) for k, v in DEFAULT_MODELS.items()},
                                         "whisper_models": ["whisper-large-v3-turbo", "whisper-large-v3"]})
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    llm.reset_state()
    llm.set_offline(False)
    monkeypatch.setattr(classify, "_loaded", {})
    yield
    llm.set_offline(False)


@pytest.fixture
def net(monkeypatch) -> FakeNet:
    fake = FakeNet()
    monkeypatch.setattr(llm, "_http_post", fake)
    return fake


def synth_market(n=1500, seed=11, start="2016-01-04") -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.009, n - 1) * np.repeat(rng.uniform(0.7, 1.8, n // 100 + 1), 100)[: n - 1],
                     index=idx[1:], name="market")


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    return dataset.build_dataset(target_rows=700, seed=7, prices=synth_prices(), market=synth_market(), save=False)


# --------------------------------------------------------------------------- leakage / features
def test_dataset_rows_never_leak_the_future(small_dataset):
    df = small_dataset
    assert len(df) >= 500
    assert (df["window_start"] < df["window_end"]).all()
    assert (df["window_end"] < df["label_start"]).all()               # label window starts after the features end
    assert (df["label_start"] <= df["label_end"]).all()
    train_df, test_df = dataset.time_split(df)
    assert train_df["label_end"].max() < test_df["window_start"].min()
    assert train_df["window_id"].max() < test_df["window_id"].min()
    assert len(train_df) + len(test_df) < len(df)                      # the purge gap removed rows


def test_labeler_is_fitted_on_training_period_only(small_dataset):
    train_df, test_df = dataset.time_split(small_dataset)
    p = dataset.fit_labeler(train_df)
    y = dataset.apply_labels(train_df, p)
    assert np.bincount(y, minlength=3) / len(y) == pytest.approx([1 / 3] * 3, abs=0.02)   # terciles of TRAIN
    shifted = test_df.assign(fwd_vol=test_df["fwd_vol"] * 3)          # tampering with test cannot move thresholds
    assert dataset.fit_labeler(train_df) == p
    assert dataset.apply_labels(shifted, p).mean() >= dataset.apply_labels(test_df, p).mean()


def test_features_identical_in_dataset_and_inference(small_dataset):
    px = synth_prices()                                                # same prices the fixture used
    rng = np.random.default_rng(1)
    for i in rng.choice(len(small_dataset), 8, replace=False):
        row = small_dataset.iloc[i]
        names = row["tickers"].split("|")
        w = dict(zip(names, map(float, row["weights"].split("|"))))
        window = px.loc[row["window_start"]:row["window_end"], names]
        live = features.build_features(window, w, market=synth_market(), fetch_market=False)
        for f in features.ALL_FEATURES:
            np.testing.assert_allclose(live[f], row[f], rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f)
        # a longer history gives the same answer: only the trailing window is used
        longer = px.loc[:row["window_end"], names]
        again = features.build_features(longer, w, market=synth_market(), fetch_market=False)
        for f in features.ALL_FEATURES:
            np.testing.assert_allclose(again[f], live[f], rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f)


def test_build_features_reuses_engine_and_is_finite():
    px = synth_prices(400)
    w = {"S0.NS": 0.5, "S1.NS": 0.3, "S2.NS": 0.2}
    f = features.build_features(px, w, fetch_market=False)
    raw = M.feature_inputs(features.lookback_slice(px), w)
    assert list(f) == features.ALL_FEATURES == features.FEATURE_NAMES
    assert all(f[k] == pytest.approx(raw[k]) for k in features.BASE_FEATURES)       # reused, not re-implemented
    assert all(np.isfinite(v) for k, v in f.items() if k not in features.MARKET_FEATURES)
    assert all(np.isnan(f[k]) for k in features.MARKET_FEATURES)                    # no market -> NaN (imputed later)


def test_repair_breaks_neutralises_unadjusted_corporate_actions():
    px = synth_prices(300, k=2)
    px.iloc[150:, 0] *= 0.5                                            # a 2:1 demerger-style drop, never reverts
    fixed, report = dataset.repair_breaks(px)
    assert list(report) == ["S0.NS"]
    assert abs(np.log(fixed["S0.NS"]).diff().abs().max()) < 0.4
    assert fixed["S1.NS"].equals(px["S1.NS"])


# --------------------------------------------------------------------------- training + saved model
def test_training_pipeline_split_report_and_files(small_dataset, tmp_path):
    rep = train.train(small_dataset, quick=True, upload=False, model_dir=tmp_path / "m", report_dir=tmp_path / "r",
                      save_labeled=False)
    sp = rep["split"]
    assert sp["train_label_end_max"] < sp["test_window_start"]         # purge gap respected
    assert {"lightgbm", "random_forest"} == set(rep["models"])
    assert {"majority_class", "volatility_rule_1y", "ewma_volatility_rule"} == set(rep["baselines"])
    assert rep["label"]["mode"] == "abs_vol" and rep["label"]["thresholds"] == list(dataset.ABS_THRESHOLDS)
    for k in ("balanced_accuracy", "within_one_class", "macro_f1"):
        assert 0.0 <= rep["models"]["lightgbm"][k] <= 1.0
    assert -1.0 <= rep["models"]["lightgbm"]["quadratic_kappa"] <= 1.0      # kappa can be negative (worse than chance)
    cm = np.array(rep["models"]["lightgbm"]["confusion_matrix"])
    assert cm.shape == (3, 3) and cm.sum() == rep["split"]["rows"]["test"]
    for f in ("risk_model.joblib", "feature_meta.json"):
        assert (tmp_path / "m" / f).is_file()
    assert (tmp_path / "r" / "metrics.json").is_file() and (tmp_path / "r" / "confusion_matrix.png").is_file()
    assert rep["verdict"]


def test_saved_report_beats_majority_baseline():
    rep_path = ROOT / "ml" / "reports" / "metrics.json"
    if not rep_path.is_file():
        pytest.skip("run `python -m ml.train` first")
    rep = json.loads(rep_path.read_text())
    assert rep["models"]["lightgbm"]["macro_f1"] > rep["baselines"]["majority_class"]["macro_f1"]
    assert (ROOT / "ml" / "reports" / "confusion_matrix.png").is_file()
    assert rep["split"]["train_label_end_max"] < rep["split"]["test_window_start"]   # no leakage in the shipped model
    meta = json.loads((ROOT / "ml" / "models" / "feature_meta.json").read_text())
    assert meta["feature_names"] == features.FEATURE_NAMES
    assert meta["labeler"]["mode"] == "abs_vol" and meta["labeler"]["thresholds"] == [0.12, 0.17]
    assert set(meta["train_medians"]) == set(features.FEATURE_NAMES)


# --------------------------------------------------------------------------- classification
def test_classify_risk_contract_and_speed():
    classify.load_model(force=True)
    px = synth_prices(400)
    w = {"S0.NS": 0.4, "S1.NS": 0.3, "S2.NS": 0.2, "S3.NS": 0.1}
    classify.classify_risk(px, w)                                      # warm
    t0 = time.perf_counter()
    r = classify.classify_risk(px, w)
    assert time.perf_counter() - t0 < 0.2
    assert set(r) >= {"label", "probs", "top_factors"}
    assert r["label"] in ("Low", "Medium", "High") and r["label"] == max(r["probs"], key=r["probs"].get)
    assert sum(r["probs"].values()) == pytest.approx(1.0, abs=1e-9)
    assert len(r["top_factors"]) == 3
    assert all(f["impact"] in ("raises risk", "lowers risk") and f["name"] and f["value"] for f in r["top_factors"])


def test_classifier_ranks_volatile_above_calm():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=300)
    calm = pd.DataFrame({f"C{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.004, 300))) for i in range(4)}, index=idx)
    wild = pd.DataFrame({f"W{i}": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.035, 300))) for i in range(4)}, index=idx)
    w_c, w_w = {c: 0.25 for c in calm}, {c: 0.25 for c in wild}
    pc, pw = classify.classify_risk(calm, w_c)["probs"], classify.classify_risk(wild, w_w)["probs"]
    assert pw["High"] > pc["High"] and pc["Low"] > pw["Low"]


def test_classify_needs_enough_history_and_stressed_wrapper():
    px = synth_prices(300)
    with pytest.raises(ValueError):
        classify.classify_risk(px.iloc[:30], {"S0.NS": 1.0})
    short = classify.classify_risk(px.iloc[:100], {"S0.NS": 0.5, "S1.NS": 0.5})
    assert short["note"]
    w = {"S0.NS": 0.5, "S1.NS": 0.5}
    st = classify.classify_stressed(px, w)
    assert st["stressed"] is True and st["label"] in ("Low", "Medium", "High")


def test_load_model_without_any_source_and_no_retrain(tmp_path):
    with pytest.raises(FileNotFoundError):
        classify.load_model(force=True, model_dir=tmp_path / "empty", allow_retrain=False)


# --------------------------------------------------------------------------- LLM wrapper
def test_second_identical_call_is_a_cache_hit(net):
    a = llm.ask_llm("t", "hello", lang="en")
    b = llm.ask_llm("t", "hello", lang="en")
    assert (a["cached"], b["cached"]) == (False, True) and a["text"] == b["text"] == "G-answer"
    assert len(net.calls) == 1
    s = llm.usage_stats()
    assert s["gemini"] == {"calls": 1, "cache_hits": 1, "failures": 0}
    c = llm.ask_llm("t", "hello", lang="hi")                           # language is part of the key
    assert c["cached"] is False and len(net.calls) == 2


def test_cache_survives_a_new_process(net):
    llm.ask_llm("t", "persist me", lang="en")
    llm.reset_state()                                                  # new "session": memory cache is gone
    again = llm.ask_llm("t", "persist me", lang="en")
    assert again["cached"] and len(net.calls) == 1                     # served from the local llm_cache mirror


def test_rate_limit_falls_through_the_chain_with_cooldown(net):
    net.script["gemini"] = [429] * 4                                   # 2 models x (try + retry)
    out = llm.ask_llm("t", "p1", lang="en")
    assert out["provider"] == "groq" and out["text"] == "Q-answer"
    assert llm._cooldown_until["gemini"] > time.time()
    net.calls.clear()
    llm.ask_llm("t", "p2", lang="en")                                  # gemini is cooling: not even tried
    assert [p for p, _ in net.calls] != [] and "gemini" not in [p for p, _ in net.calls]
    # headroom ordering: groq just served a call, so the idle provider (mistral) is preferred next
    assert net.calls[0][0] == "mistral"
    assert llm.usage_stats()["gemini"]["failures"] == 4


def test_invalid_keys_end_in_llmunavailable_without_crashing(net):
    for p in net.script:
        net.script[p] = [401] * 5
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_llm("t", "bad keys", lang="en")
    assert all(s["failures"] >= 1 for s in llm.usage_stats().values())


def test_offline_mode_makes_zero_network_calls(net):
    llm.ask_llm("t", "warm", lang="en")
    net.calls.clear()
    llm.set_offline(True)
    assert llm.ask_llm("t", "warm", lang="en")["cached"] is True      # cache still served
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_llm("t", "never asked", lang="en")
    assert net.calls == []


def test_json_mode_rejects_invalid_json_and_tries_next_provider(net):
    net.script["gemini"] = [(200, "this is not json")]
    net.text["groq"] = '```json\n{"ok": true}\n```'
    out = llm.ask_llm("t", "give json", json_mode=True)
    assert out["provider"] == "groq" and llm.parse_json(out["text"]) == {"ok": True}


def test_unknown_model_name_falls_back_to_next_candidate(net):
    net.script["gemini"] = [404]
    out = llm.ask_llm("t", "model fallback", lang="en")
    assert out["provider"] == "gemini" and len(net.calls) == 2         # same provider, second model name


def test_mistral_is_last_for_non_english_and_cooldown_is_respected():
    assert llm._provider_order("hi")[-1] == "mistral"
    llm._cooldown_until["gemini"] = time.time() + 60
    assert "gemini" not in llm._provider_order("en")


def test_no_keys_means_unavailable_not_an_exception_storm(monkeypatch, net):
    monkeypatch.setattr(llm, "_config", {"keys": {"gemini": [], "groq": [], "mistral": []},
                                         "models": DEFAULT_MODELS, "whisper_models": []})
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_llm("t", "x")
    assert net.calls == []


# --------------------------------------------------------------------------- explanations
@pytest.fixture
def world():
    px = synth_prices(500, k=5)
    return px, {"S0.NS": 0.6, "S1.NS": 0.2, "S2.NS": 0.2}


def test_explain_portfolio_english_is_template_only(world, net):
    px, w = world
    out = explain.explain_portfolio_ex(px, w, "High")
    assert out["translated"] is False and out["fallback"] is False
    assert "High risk" in out["text"] and "volatility" in out["text"] and "%" in out["text"]
    assert net.calls == []                                             # no LLM for English


def test_explain_translation_is_one_call_and_falls_back_gracefully(world, net, monkeypatch):
    px, w = world
    net.text["gemini"] = "यह पोर्टफोलियो उच्च जोखिम का है"
    r = explain.explain_portfolio_ex(px, w, "High", "hi")
    assert r["translated"] and r["text"].startswith("यह") and len(net.calls) == 1
    again = explain.explain_portfolio_ex(px, w, "High", "hi")
    assert again["translated"] and len(net.calls) == 1                 # cached
    llm.set_offline(True)
    fb = explain.explain_portfolio_ex(px, w, "High", "mr")
    assert fb["fallback"] and fb["lang"] == "en" and "risk" in fb["text"]


def test_explain_allocation_lists_every_big_change_with_a_reason(world):
    px, w = world
    opt = {"S0.NS": 0.25, "S1.NS": 0.25, "S2.NS": 0.2, "S3.NS": 0.15, "S4.NS": 0.15}
    txt = explain.explain_allocation(px, w, opt, "Low")
    assert "Reduce S0.NS from 60% to 25%" in txt
    assert txt.count("Increase") == 3                                  # S1 (+5), S3 (+15), S4 (+15)
    assert all(": " in line for line in txt.splitlines()[:-1])
    tiny = explain.explain_allocation(px, w, {**w}, "Low")
    assert "very close" in tiny


def test_explain_scenario_mentions_label_change(world):
    px, w = world
    sc = simulate.apply_scenario(px, w, {"type": "market_shock", "pct": -0.2})
    txt = explain.explain_scenario(sc, "Medium", "High")
    assert "-20%" in txt and "from Medium to High" in txt
    assert "stay" in explain.explain_scenario(sc, "Low", "Low")


# --------------------------------------------------------------------------- voice
def test_twenty_sample_utterances_extract_exactly():
    llm.set_offline(True)                                              # rule-based path only
    samples = voice.demo_samples()
    assert len(samples) == 20 and {s["lang"] for s in samples} == {"en", "hi", "mr"}
    for s in samples:
        r = voice.extract_portfolio(s["text"])
        e = s["expected"]
        assert r["tickers"] == e["tickers"], s["text"]
        assert {k: round(v) for k, v in r["amounts"].items()} == e["amounts"], s["text"]
        assert r["risk_level"] == e["risk_level"], s["text"]
        assert json.loads(json.dumps(r)) == r                          # valid JSON round trip
        assert 0.0 <= r["confidence"] <= 1.0


def test_extract_portfolio_llm_path_validates_tickers(net):
    net.text["gemini"] = json.dumps({"tickers": ["TCS.NS", "FAKECO", "infy"], "amounts": {"TCS.NS": "30000", "BOGUS": 5},
                                     "risk_level": "low", "unresolved": ["Zomato"]})
    r = voice.extract_portfolio("some free text about stocks")
    assert r["source"] == "llm"
    assert r["tickers"] == ["TCS.NS", "INFY.NS"] and r["amounts"] == {"TCS.NS": 30000.0}
    assert set(r["unresolved"]) == {"Zomato", "FAKECO"} and r["risk_level"] == "Low"


def test_extract_portfolio_falls_back_to_rules_on_garbage_or_outage(net):
    net.text["gemini"] = "I am sorry, I cannot help"                   # not JSON -> rejected, chain continues
    net.text["groq"] = "also not json"
    net.text["mistral"] = "nope"
    r = voice.extract_portfolio("Invest 2 lakh in Wipro and ITC, low risk")
    assert r["source"] == "rules" and r["tickers"] == ["WIPRO.NS", "ITC.NS"] and r["risk_level"] == "Low"
    assert voice.extract_portfolio("")["tickers"] == []


def test_transcribe_groq_rotation_fallback_and_error_codes(monkeypatch):
    calls = []

    def fake_post(url, headers, files, data, timeout=30.0):
        calls.append((headers["authorization"], data["model"], data.get("language")))
        key = headers["authorization"]
        if key.endswith("bad"):
            return 401, {"error": "invalid key"}
        if data["model"].endswith("turbo"):
            return 404, {"error": "model not found"}
        return 200, {"text": " मेरे पास पचास हजार रुपये हैं "}

    monkeypatch.setattr(llm, "_config", {"keys": {"gemini": [], "groq": ["bad", "good"], "mistral": []},
                                         "models": DEFAULT_MODELS,
                                         "whisper_models": ["whisper-large-v3-turbo", "whisper-large-v3"]})
    monkeypatch.setattr(voice, "_http_post_multipart", fake_post)
    out = voice.transcribe_ex(b"RIFFfake", "hi")
    assert out["text"] == "मेरे पास पचास हजार रुपये हैं" and out["provider"] == "groq-whisper"
    assert calls[0][0].endswith("bad") and calls[-1][1] == "whisper-large-v3" and calls[-1][2] == "hi"
    assert voice.transcribe(b"RIFFfake") == out["text"]
    assert voice.transcribe_ex(b"", "hi")["error"] == "empty_audio"
    llm.set_offline(True)
    assert voice.transcribe_ex(b"x", "hi")["error"] == "offline"


def test_transcribe_failure_returns_code_not_exception(monkeypatch):
    monkeypatch.setattr(voice, "_http_post_multipart", lambda *a, **k: (500, {}))
    out = voice.transcribe_ex(b"RIFFfake", "mr")
    assert out["text"] == "" and out["error"] == "stt_failed"


# --------------------------------------------------------------------------- news
HEADLINES = [{"title": f"Headline {i} about {t}", "source": "Src", "url": f"http://x/{t}/{i}", "ticker": t}
             for t in ("TCS.NS", "INFY.NS") for i in range(3)]
GOOD_JSON = json.dumps({"sentiment": "positive", "per_ticker": [
    {"ticker": "TCS.NS", "sentiment": "Positive", "drivers": "Strong results"},
    {"ticker": "ZZZ", "sentiment": "Negative", "drivers": "not requested"}],
    "sector_trends": "IT rallies", "risks": ["Visa policy", "", "Demand"], "summary": "Mostly positive."})


def test_news_insight_shape_cache_and_snapshot(net, monkeypatch):
    monkeypatch.setattr(news, "fetch_headlines", lambda tickers: HEADLINES)
    net.text["gemini"] = GOOD_JSON
    r = news.get_news_insight(["tcs.ns", "INFY.NS"], "en")
    assert r["source"] == "live" and r["ai_available"] and r["sentiment"] == "Positive"
    assert [p["ticker"] for p in r["per_ticker"]] == ["TCS.NS"]        # unrequested ticker dropped
    assert r["risks"] == ["Visa policy", "Demand"] and len(r["headlines"]) == 6
    assert {"title", "source", "url"} <= set(r["headlines"][0])
    assert len(net.calls) == 1                                         # ONE call for all tickers
    assert news.get_news_insight(["TCS.NS", "INFY.NS"], "en")["source"] == "cache" and len(net.calls) == 1
    llm.set_offline(True)
    off = news.get_news_insight(["INFY.NS", "TCS.NS"], "en")           # order-insensitive snapshot
    assert off["source"] == "offline" and off["summary"] == "Mostly positive."


def test_news_never_raises_when_everything_fails(net, monkeypatch):
    monkeypatch.setattr(news, "fetch_headlines", lambda tickers: HEADLINES)
    for p in net.script:
        net.script[p] = [500] * 6
    r = news.get_news_insight(["TCS.NS"], "hi")
    assert r["ai_available"] is False and r["sentiment"] == "Neutral" and len(r["headlines"]) == 6
    monkeypatch.setattr(news, "fetch_headlines", lambda tickers: [])
    r2 = news.get_news_insight(["WIPRO.NS"], "en")
    assert r2["headlines"] == [] and r2["source"] == "offline"
    assert news.get_news_insight([], "en")["headlines"] == []


def test_news_parses_both_yfinance_schemas_and_caps_headlines(monkeypatch):
    old = {"title": "Old", "publisher": "Reuters", "link": "http://o"}
    new = {"id": "1", "content": {"title": "New", "provider": {"displayName": "Bloomberg"},
                                  "clickThroughUrl": {"url": "http://n"}}}
    assert news._parse_yf_item(old) == {"title": "Old", "source": "Reuters", "url": "http://o"}
    assert news._parse_yf_item(new) == {"title": "New", "source": "Bloomberg", "url": "http://n"}
    assert news._parse_yf_item({"content": {}}) is None
    many = [{"title": f"Story {t} {i}", "source": "S", "url": ""} for t in "ABCDE" for i in range(9)]
    monkeypatch.setattr(news, "_yf_news", lambda t: many + [many[0]])    # duplicate title within the feed
    monkeypatch.setattr(news, "_rss_news", lambda t: [])
    out = news.fetch_headlines(["T1", "T2", "T3", "T4", "T5"])
    assert len(out) <= 20
    assert all(sum(1 for h in out if h["ticker"] == t) <= 5 for t in ("T1", "T2", "T3", "T4", "T5"))
    assert len({news._norm_title(h["title"]) for h in out}) == len(out)  # globally de-duplicated


# --------------------------------------------------------------------------- i18n + repo contracts
def test_translations_are_complete_and_safe():
    assert i18n.missing_keys("hi") == [] and i18n.missing_keys("mr") == []
    assert set(i18n.STRINGS["hi"]) == set(i18n.STRINGS["mr"]) == set(i18n.STRINGS["en"]) and len(i18n.STRINGS["en"]) >= 80
    assert i18n.t("tab_risk", "hi") != i18n.t("tab_risk", "en")
    assert i18n.t("does_not_exist", "mr") == "does_not_exist"          # falls back to the key, never raises
    assert i18n.t("app_title", "xx") == i18n.STRINGS["en"]["app_title"]
    assert "अस्थिरता" in i18n.glossary_prompt("hi") and "घसरण" in i18n.glossary_prompt("mr")
    assert i18n.glossary_prompt("en") == ""


def test_mock_files_match_real_shapes(monkeypatch):
    m = lambda n: json.loads((ROOT / "data" / "mocks" / f"mock_{n}.json").read_text(encoding="utf-8"))
    px = synth_prices(300)
    real = classify.classify_risk(px, {"S0.NS": 0.5, "S1.NS": 0.5})
    assert set(m("classify")) == set(real)
    assert set(m("classify")["top_factors"][0]) == set(real["top_factors"][0])
    assert set(m("explain")) == {"explain_portfolio", "explain_allocation", "explain_scenario"}
    llm.set_offline(True)
    assert set(m("voice")["extract_portfolio"]) == set(voice.extract_portfolio("50000 in TCS and Infosys low risk"))
    monkeypatch.setattr(news, "fetch_headlines", lambda t: HEADLINES)
    monkeypatch.setattr(news, "_load_snapshot", lambda *a, **k: None)
    live = news._shape(json.loads(GOOD_JSON), ["TCS.NS"], HEADLINES)
    live.update(source="live", ai_available=True)
    assert set(m("news")) == set(live)
    sample = m("i18n_sample")
    assert set(sample) == {"en", "hi", "mr"} and all(sample["hi"][k] == i18n.t(k, "hi") for k in sample["hi"])


def test_ml_package_never_imports_streamlit():
    for path in (ROOT / "ml").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] == "streamlit" for n in names), f"{path.name} imports streamlit"


def test_one_bad_model_does_not_cool_down_the_provider(net):
    net.script["gemini"] = [503, 503]                                   # first model: try + retry both 503
    out = llm.ask_llm("t", "model 1 is overloaded", lang="en")
    assert out["provider"] == "gemini"                                  # the second model answered
    assert llm._cooldown_until["gemini"] == 0.0
    assert len(net.calls) == 3


def test_gemini_request_leaves_room_for_thinking_tokens():
    _, _, body = llm._build_request("gemini", "m", "k", "p", "", False, 600)
    assert body["generationConfig"]["maxOutputTokens"] == 600 + llm.GEMINI_THINKING_HEADROOM


# --------------------------------------------------------------------------- v2: features, labels, imputation
def test_short_term_features_react_to_a_volatility_regime_change():
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.004, 230)
    wild = rng.normal(0, 0.03, 23)                                     # last ~month turns violent
    idx = pd.bdate_range("2023-01-02", periods=254)
    px = pd.DataFrame({f"R{i}.NS": 100 * np.exp(np.cumsum(np.r_[0, calm, wild] + rng.normal(0, 0.002, 254)))
                       for i in range(3)}, index=idx)
    f = features.build_features(px, {c: 1 / 3 for c in px}, fetch_market=False)
    assert f["vol_21"] > 2 * f["vol_126"] and f["vol_ratio_21"] > 1.5
    assert f["ewma_vol"] > f["ann_vol"]                               # recency-weighted vol reacts faster
    assert f["max_asset_vol"] >= f["undiversified_vol"] >= f["ann_vol"] * 0.5


def test_sector_concentration_and_market_beta():
    px = synth_prices(300, k=2)
    px.columns = ["TCS.NS", "HDFCBANK.NS"]                              # IT and Banking in universe.json
    both = features.build_features(px, {"TCS.NS": 0.5, "HDFCBANK.NS": 0.5}, fetch_market=False)
    one = features.build_features(px[["TCS.NS"]], {"TCS.NS": 1.0}, fetch_market=False)
    assert both["sector_hhi"] == pytest.approx(0.5) and one["sector_hhi"] == pytest.approx(1.0)
    idx = px.index
    mkt = pd.Series(np.diff(np.log(px["TCS.NS"].to_numpy())), index=idx[1:])         # market == TCS exactly
    f = features.build_features(px[["TCS.NS"]], {"TCS.NS": 1.0}, market=mkt, fetch_market=False)
    assert f["beta_to_market"] == pytest.approx(1.0, abs=1e-6) and np.isfinite(f["market_vol_21"])


def test_fixed_labels_do_not_depend_on_training_data(small_dataset):
    tr, te = dataset.time_split(small_dataset)
    a = dataset.fit_labeler(tr, mode="abs_vol")
    b = dataset.fit_labeler(te, mode="abs_vol")                         # different data, same definition
    assert a == b and a["thresholds"] == list(dataset.ABS_THRESHOLDS)
    y = dataset.apply_labels(te, a)
    expected = np.digitize(te["fwd_vol"], [0.12, 0.17])
    assert (y == expected).all()
    tercile = dataset.fit_labeler(tr, mode="vol_tercile")               # fitted on train only
    assert tercile["fitted_terciles"] and tercile["thresholds"][0] < tercile["thresholds"][1]
    assert dataset.fit_labeler(tr, mode="vol_tercile") == tercile


def test_classifier_imputes_missing_market_features(monkeypatch):
    classify.load_model(force=True)
    monkeypatch.setattr(features, "_auto_market", lambda index: None)   # offline: market proxy unavailable
    px = synth_prices(400)
    r = classify.classify_risk(px, {"S0.NS": 0.5, "S1.NS": 0.3, "S2.NS": 0.2})
    assert sum(r["probs"].values()) == pytest.approx(1.0)
    assert json.loads(json.dumps(r))["label"] == r["label"]             # strict-JSON safe (NaN -> null)
    assert r["features"]["beta_to_market"] is None
