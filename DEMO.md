# DEMO.md — 6-Minute Demo Script

> **Before starting:** Run `python scripts/prewarm.py`, wake up Supabase dashboard, verify Offline mode works.

---

## Step 1 — Voice Input in Hindi (~1 min)

1. Open sidebar → expand **Voice Input**.
2. Click 🎤 **Record Audio**.
3. Say in Hindi: *"Mere paas 1 lakh rupaye hain, TCS, Infosys aur HDFC Bank mein lagana hai, risk kam chahiye."*
4. Click **Fill Portfolio from Transcript** → watch sidebar auto-fill tickers, amounts, risk level.
5. Point out: unresolved names are flagged; user can fix before submitting.

---

## Step 2 — Risk Tab (~1 min)

1. Click **Risk** tab.
2. Show the risk badge (Low / Medium / High) with class probabilities bar.
3. Point to the **Risk Contribution vs Weight** bar chart → highlight TCS.NS in red (risk > weight).
4. Read the explanation text: "Rated Medium because volatility is 18.4%, 42% of risk comes from TCS, and average correlation is 0.61."

---

## Step 3 — Allocation Tab: Switch Risk Level (~1 min)

1. In sidebar, change risk level: **Low → Medium → High**.
2. Watch the **Allocation** tab update instantly → weights shift, before/after metrics change.
3. Show the delta table (Δ Weight column).
4. Point to **Detailed Explanation** text (explainable recommendations brownie point).

---

## Step 4 — Simulate + Compare (~1 min)

1. **Simulate tab** → click **COVID crash** preset → growth-of-₹100 dips sharply, then switch to **2021 bull run** → strong recovery.
2. Toggle **Log Scale** on.
3. **Compare tab** → show table (Original / Recommended / Equal Weight / NIFTY 50).
4. Show risk-return scatter → portfolio star vs benchmark diamond.
5. Read the verdict: "Your portfolio outperforms / underperforms NIFTY 50 on a risk-adjusted basis."

---

## Step 5 — Scenarios + Monte Carlo (~1 min)

1. **Scenarios tab** → select **Market Crash -20%**.
2. Show result cards: Value Before ₹1,00,000 → Value After ₹80,000 · New Risk Label → High.
3. Monte Carlo fan chart: 5th/50th/95th percentile paths over 9 months.
4. Point to: Probability of Loss = 38%, Expected Shortfall = -18.2%.

---

## Step 6 — Market Intel in Marathi + Offline Mode (~1 min)

1. In sidebar, switch language to **मराठी**.
2. All labels switch instantly.
3. **Market Intel tab** → click **अंतर्दृष्टी निर्माण करा** (Generate Insight).
4. Show sentiment chip, sector trends, risks and headlines.
5. Toggle **Offline Mode ON** in sidebar → banner appears "LLM result from cache".
6. Click Generate Insight again → still works (uses cache), zero new LLM calls.
7. Open **LLM Usage Panel** → shows 0 new calls since offline mode enabled.

---

## Step 7 — Save Portfolio + PDF Export (~30 sec)

1. Type a name in **Portfolio Name** → click **💾 Save Portfolio** → Saved ✅.
2. Click **📄 Export PDF** → download button appears → click to download.
3. Open PDF: show portfolio table, risk label badge, key metrics, recommended allocation, Hindi heading if language is हिन्दी, disclaimer at footer.

---

## Architecture Slide (Free-Tier Stack)

| Layer | Technology | Cost |
|---|---|---|
| UI | Streamlit Community Cloud | Free |
| DB + Storage | Supabase free tier | Free |
| Market data | yfinance | Free, no key |
| ML | scikit-learn + LightGBM | Free, CPU only |
| LLM | Gemini AI Studio → Groq → Mistral | Free tiers |
| STT | Groq Whisper | Free tier |
| PDF | fpdf2 + Noto Sans Devanagari | Free, SIL OFL |

> Zero paid services. Zero credit cards. Full demo works offline via cache.
