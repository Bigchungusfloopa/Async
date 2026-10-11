"""Voice input: speech-to-text and "what did the user say" -> portfolio JSON.

* ``transcribe`` : Groq Whisper (primary) -> local ``faster-whisper`` small model (if installed) -> error code.
* ``extract_portfolio`` : ONE JSON-mode LLM call (name->ticker map + few-shot examples in English, Hindi,
  Hinglish, Marathi); every ticker is validated against ``universe.json``. If the LLM is unavailable a
  deterministic rule-based parser (understands "50 hazaar", "2 lakh", Devanagari digits and company names)
  produces the same output shape, so the voice demo still works offline.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import unicodedata
from typing import Any

import httpx

from engine.universe import load_universe
from . import llm

log = logging.getLogger("ml.voice")

STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_TIMEOUT_S = 30.0
EXTRACT_TASK = "voice_extract"
LEVELS = ("Low", "Medium", "High")

_http_post_multipart = None  # set below; patched in tests
_local_model: Any = None
_local_lock = threading.Lock()


# --------------------------------------------------------------------------- speech to text
def _post_multipart(url: str, headers: dict, files: dict, data: dict, timeout: float = STT_TIMEOUT_S) -> tuple[int, Any]:
    try:
        r = httpx.post(url, headers=headers, files=files, data=data, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise llm._Transient(type(exc).__name__) from exc
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


_http_post_multipart = _post_multipart


def _bias_prompt() -> str:
    """Whisper `prompt`: a short list of company names nudges recognition toward tickers' spoken forms."""
    names = [e["name"].split(" (")[0] for e in load_universe() if e.get("type") == "stock"][:24]
    return "Stocks and amounts: " + ", ".join(names) + ". रुपये, हज़ार, लाख, जोखिम."


def _groq_transcribe(audio: bytes, lang: str, filename: str, errors: list[str]) -> str | None:
    keys = llm.get_keys("groq")
    if not keys:
        errors.append("no_keys")
        return None
    for key in keys:
        for model in llm.get_whisper_models():
            data = {"model": model, "response_format": "json", "temperature": "0", "prompt": _bias_prompt()}
            if lang in ("hi", "mr", "en"):
                data["language"] = lang
            llm._mark("groq", "calls")
            try:
                status, body = _http_post_multipart(
                    STT_URL, {"authorization": f"Bearer {key}"},
                    {"file": (filename, audio, "audio/wav" if filename.endswith(".wav") else "application/octet-stream")},
                    data)
            except llm._Transient:
                status, body = 599, None
            if status == 200 and isinstance(body, dict):
                return str(body.get("text", "")).strip()
            llm._mark("groq", "failures")
            if status == 404:
                continue                        # next model name
            if status in (401, 403):
                break                           # next key
            errors.append(f"groq_http_{status}")
            return None                         # 429 / 5xx / 400: let the local fallback try
    return None


def _local_transcribe(audio: bytes, lang: str) -> str | None:
    global _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    try:
        with _local_lock:
            if _local_model is None:
                _local_model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = _local_model.transcribe(io.BytesIO(audio), language=lang if lang in ("hi", "mr", "en") else None)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("local whisper failed: %s", type(exc).__name__)
        return None


def transcribe_ex(audio_bytes: bytes, lang_hint: str = "hi", filename: str = "audio.wav") -> dict:
    """``{"text", "provider", "error"}``; ``error`` is "" on success, else a short code
    (``empty_audio``, ``offline``, ``stt_failed``) so the UI can say "type it instead"."""
    if not audio_bytes:
        return {"text": "", "provider": "", "error": "empty_audio"}
    if llm.is_offline():
        return {"text": "", "provider": "", "error": "offline"}
    errors: list[str] = []
    text = _groq_transcribe(audio_bytes, lang_hint, filename, errors)
    if text:
        return {"text": text, "provider": "groq-whisper", "error": ""}
    text = _local_transcribe(audio_bytes, lang_hint)
    if text:
        return {"text": text, "provider": "faster-whisper", "error": ""}
    return {"text": "", "provider": "", "error": "stt_failed"}


def transcribe(audio_bytes: bytes, lang_hint: str = "hi") -> str:
    """Transcript text, or an empty string when every STT path failed (use :func:`transcribe_ex` for the
    reason). The UI should let the user edit the transcript before extraction."""
    return transcribe_ex(audio_bytes, lang_hint)["text"]


# --------------------------------------------------------------------------- aliases
# spoken / written names -> ticker. Latin aliases are matched on word boundaries, Devanagari ones as substrings
# (Marathi glues postpositions onto nouns: "इन्फोसिसमध्ये").
_ALIASES: dict[str, str] = {
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS", "टीसीएस": "TCS.NS", "टाटा कन्सल्टन्सी": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS", "इन्फोसिस": "INFY.NS", "इंफोसिस": "INFY.NS",
    "wipro": "WIPRO.NS", "विप्रो": "WIPRO.NS",
    "hcl": "HCLTECH.NS", "hcl tech": "HCLTECH.NS", "एचसीएल": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS", "टेक महिंद्रा": "TECHM.NS", "टेक महिंद्रा": "TECHM.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS", "एचडीएफसी": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS", "icici bank": "ICICIBANK.NS", "आईसीआईसीआई": "ICICIBANK.NS", "आयसीआयसीआय": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS", "एसबीआई": "SBIN.NS", "एसबीआय": "SBIN.NS",
    "स्टेट बैंक": "SBIN.NS", "स्टेट बँक": "SBIN.NS",
    "kotak": "KOTAKBANK.NS", "कोटक": "KOTAKBANK.NS", "axis bank": "AXISBANK.NS", "axis": "AXISBANK.NS",
    "ऐक्सिस": "AXISBANK.NS", "ॲक्सिस": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS", "बजाज फाइनेंस": "BAJFINANCE.NS", "बजाज फायनान्स": "BAJFINANCE.NS",
    "reliance": "RELIANCE.NS", "ril": "RELIANCE.NS", "रिलायंस": "RELIANCE.NS", "रिलायन्स": "RELIANCE.NS",
    "ongc": "ONGC.NS", "ओएनजीसी": "ONGC.NS", "ntpc": "NTPC.NS", "एनटीपीसी": "NTPC.NS",
    "power grid": "POWERGRID.NS", "पावर ग्रिड": "POWERGRID.NS",
    "itc": "ITC.NS", "आईटीसी": "ITC.NS", "आयटीसी": "ITC.NS",
    "hindustan unilever": "HINDUNILVR.NS", "hul": "HINDUNILVR.NS", "हिंदुस्तान यूनिलीवर": "HINDUNILVR.NS",
    "हिंदुस्तान युनिलिव्हर": "HINDUNILVR.NS", "nestle": "NESTLEIND.NS", "नेस्ले": "NESTLEIND.NS",
    "sun pharma": "SUNPHARMA.NS", "सन फार्मा": "SUNPHARMA.NS", "dr reddy": "DRREDDY.NS", "dr reddys": "DRREDDY.NS",
    "cipla": "CIPLA.NS", "सिप्ला": "CIPLA.NS",
    "maruti": "MARUTI.NS", "मारुति": "MARUTI.NS", "मारुती": "MARUTI.NS",
    "tata motors": "TMPV.NS", "टाटा मोटर्स": "TMPV.NS", "टाटा मोटर": "TMPV.NS",
    "mahindra": "M&M.NS", "m&m": "M&M.NS", "महिंद्रा": "M&M.NS", "महिंद्र": "M&M.NS",
    "l&t": "LT.NS", "larsen": "LT.NS", "एलएंडटी": "LT.NS", "एल एंड टी": "LT.NS", "एल अँड टी": "LT.NS",
    "ultratech": "ULTRACEMCO.NS", "अल्ट्राटेक": "ULTRACEMCO.NS",
    "airtel": "BHARTIARTL.NS", "bharti airtel": "BHARTIARTL.NS", "एयरटेल": "BHARTIARTL.NS", "एअरटेल": "BHARTIARTL.NS",
    "tata steel": "TATASTEEL.NS", "टाटा स्टील": "TATASTEEL.NS", "hindalco": "HINDALCO.NS", "हिंडाल्को": "HINDALCO.NS",
    "asian paints": "ASIANPAINT.NS", "एशियन पेंट": "ASIANPAINT.NS", "एशियन पेंट्स": "ASIANPAINT.NS",
    "titan": "TITAN.NS", "टाइटन": "TITAN.NS", "टायटन": "TITAN.NS",
    "apple": "AAPL", "एप्पल": "AAPL", "ऍपल": "AAPL", "ॲपल": "AAPL",
    "microsoft": "MSFT", "मायक्रोसॉफ्ट": "MSFT", "माइक्रोसॉफ्ट": "MSFT",
    "nvidia": "NVDA", "एनविडिया": "NVDA", "एनव्हिडिया": "NVDA", "google": "GOOGL", "alphabet": "GOOGL",
    "गूगल": "GOOGL", "meta": "META", "facebook": "META", "amazon": "AMZN", "अमेज़न": "AMZN", "अमेझॉन": "AMZN",
    "tesla": "TSLA", "टेस्ला": "TSLA", "jpmorgan": "JPM", "visa": "V", "exxon": "XOM", "johnson": "JNJ",
    "unitedhealth": "UNH", "procter": "PG", "coca cola": "KO", "coca-cola": "KO", "walmart": "WMT",
    "gold etf": "GOLDBEES.NS", "gold bees": "GOLDBEES.NS", "goldbees": "GOLDBEES.NS", "gold": "GOLDBEES.NS",
    "सोना": "GOLDBEES.NS", "सोने": "GOLDBEES.NS", "nifty bees": "NIFTYBEES.NS", "niftybees": "NIFTYBEES.NS",
    "nifty etf": "NIFTYBEES.NS", "spy": "SPY", "qqq": "QQQ", "tlt": "TLT",
}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _norm(s: str) -> str:
    """Lower-case + NFD so precomposed nukta letters (U+095B ज़) and ज+़ compare equal."""
    return unicodedata.normalize("NFD", s).lower()


def _universe_tickers() -> set[str]:
    return {e["ticker"].upper() for e in load_universe()}


def _compiled_aliases() -> list[tuple[str, str, re.Pattern]]:
    uni = _universe_tickers()
    out = []
    for alias, ticker in _ALIASES.items():
        if ticker not in uni:
            continue
        alias = _norm(alias)
        if _DEVANAGARI.search(alias):
            pat = re.compile(re.escape(alias))
        else:
            pat = re.compile(r"(?<![a-z0-9&])" + re.escape(alias) + r"(?![a-z0-9&])")
        out.append((alias, ticker, pat))
    out.sort(key=lambda x: -len(x[0]))          # longest alias first ("tata motors" before "tata")
    return out


_ALIAS_CACHE: list | None = None


def low_orig_upper(text: str) -> str:
    """Original text with positions aligned to the normalised text (ASCII-only uppercase tokens survive)."""
    return unicodedata.normalize("NFD", text)


def _find_companies(text: str) -> list[tuple[int, int, str]]:
    """``[(start, end, ticker)]`` in order of appearance; overlapping weaker matches are dropped."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        _ALIAS_CACHE = _compiled_aliases()
    low = _norm(text)
    taken: list[tuple[int, int, str]] = []
    for _alias, ticker, pat in _ALIAS_CACHE:
        for m in pat.finditer(low):
            if any(not (m.end() <= a or m.start() >= b) for a, b, _ in taken):
                continue
            taken.append((m.start(), m.end(), ticker))
    for m in re.finditer(r"\b([A-Z][A-Z0-9&]{1,}(?:\.(?:NS|BO))?)\b", low_orig_upper(text)):   # literal tickers: AAPL, TCS.NS
        tok = m.group(1)
        cand = [tok] if "." in tok else [tok, tok + ".NS"]
        for c in cand:
            if c in _universe_tickers() and not any(not (m.end() <= a or m.start() >= b) for a, b, _ in taken):
                taken.append((m.start(), m.end(), c))
                break
    return sorted(taken)


# --------------------------------------------------------------------------- numbers
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_WORD_NUM = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "panch": 5, "chhe": 6, "saat": 7, "aath": 8, "nau": 9,
    "das": 10, "bees": 20, "vees": 20, "tees": 30, "chalis": 40, "pachas": 50, "pachaas": 50, "pannas": 50,
    "saath": 60, "sau": 100, "dedh": 1.5, "dhai": 2.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100,
    "एक": 1, "दो": 2, "दोन": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पाच": 5, "दस": 10, "दहा": 10, "बीस": 20,
    "वीस": 20, "तीस": 30, "चालीस": 40, "चाळीस": 40, "पचास": 50, "पन्नास": 50, "सौ": 100, "शंभर": 100,
    "डेढ़": 1.5, "दीड": 1.5, "ढाई": 2.5, "अडीच": 2.5,
}
_MAGNITUDE = {
    "k": 1e3, "thousand": 1e3, "hazaar": 1e3, "hazar": 1e3, "hajar": 1e3, "हजार": 1e3, "हज़ार": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "लाख": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "करोड": 1e7, "करोड़": 1e7, "कोटी": 1e7, "million": 1e6, "m": 1e6,
}
_NUM_RE = re.compile(
    r"(?<![\w.])(?P<num>\d[\d,]*(?:\.\d+)?|" + "|".join(sorted(map(re.escape, _WORD_NUM), key=len, reverse=True)) +
    r")\s*(?P<mag>" + "|".join(sorted(map(re.escape, _MAGNITUDE), key=len, reverse=True)) + r")?(?![a-z])",
    re.IGNORECASE)


def _find_amounts(text: str) -> list[tuple[int, int, float]]:
    """``[(start, end, value)]``; bare small numbers without a magnitude (e.g. "2 stocks") are ignored."""
    low = _norm(text.translate(_DIGITS))
    out = []
    for m in _NUM_RE.finditer(low):
        raw, mag = m.group("num"), m.group("mag")
        if re.match(r"\d", raw):
            val = float(raw.replace(",", ""))
        else:
            val = float(_WORD_NUM[raw])
        if mag:
            val *= _MAGNITUDE[mag]
        elif val < 1000:
            continue                                    # "3" in "3 companies" is not money
        out.append((m.start(), m.end(), val))
    # a word number directly followed by a magnitude is captured as one match; merge "sau hazaar" no-op
    return out


# --------------------------------------------------------------------------- risk words
_RISK_NOUN = re.compile(r"risk|jokhim|jokham|jokhmi|जोखिम|जोखीम|जोखम|रिस्क", re.IGNORECASE)
_LEVEL_WORDS = {
    "Low": ["low", "kam", "kaam", "kum", "kamee", "safe", "conservative", "surakshit", "minimal", "कम", "कमी",
            "सुरक्षित", "कमीत", "लो", "कम से कम"],
    "High": ["high", "zyada", "jyada", "zyaada", "jyaada", "aggressive", "adhik", "ज़्यादा", "ज्यादा", "ज्यादा",
             "जास्त", "अधिक", "हाई", "जादा"],
    "Medium": ["medium", "moderate", "madhyam", "balanced", "average", "मध्यम", "मीडियम", "मध्यम"],
}


_UNAMBIGUOUS = {"High": ["aggressive", "high-risk"], "Low": ["conservative", "risk-free"],
                "Medium": ["moderate", "balanced"]}


def _risk_level(text: str) -> str | None:
    low = _norm(text)
    nouns = [m.start() for m in _RISK_NOUN.finditer(low)]
    if not nouns:
        for level, words in _UNAMBIGUOUS.items():
            if any(re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", low) for w in words):
                return level
        return None
    best: tuple[int, str] | None = None
    for level, words in _LEVEL_WORDS.items():
        for w in map(_norm, words):
            for m in re.finditer(r"(?<![\w])" + re.escape(w) + r"(?![\w])" if not _DEVANAGARI.search(w)
                                 else re.escape(w), low):
                d = min(abs(m.start() - n) for n in nouns)
                if d <= 40 and (best is None or d < best[0]):
                    best = (d, level)
    return best[1] if best else None


# --------------------------------------------------------------------------- rule-based extraction
def _rule_extract(text: str) -> dict:
    companies = _find_companies(text)
    amounts = _find_amounts(text)
    if len(amounts) > len(companies) >= 1:             # "50 hazaar hain, wipro mein 30 hazaar aur itc mein 20"
        for i, a in enumerate(amounts):
            rest = sum(b[2] for j, b in enumerate(amounts) if j != i)
            if abs(a[2] - rest) < 1e-6:
                amounts = [b for j, b in enumerate(amounts) if j != i]
                break
    tickers: list[str] = []
    for _, _, t in companies:
        if t not in tickers:
            tickers.append(t)
    per: dict[str, float] = {}
    total = None
    if tickers and amounts:
        if len(amounts) == 1 or (len(amounts) > 1 and len(tickers) == 1):
            total = amounts[0][2] if len(amounts) == 1 else sum(a[2] for a in amounts)
            share = total / len(tickers)
            per = {t: share for t in tickers}
        elif len(amounts) == len(companies):
            for (_, _, t), (_, _, v) in zip(companies, amounts):    # in order of mention
                per[t] = per.get(t, 0.0) + v
        else:                                                       # pair each amount with the nearest company
            for s, e, v in amounts:
                _, _, t = min(companies, key=lambda c: min(abs(c[0] - e), abs(c[1] - s)))
                per[t] = per.get(t, 0.0) + v
    elif amounts and not tickers:
        total = amounts[0][2]
    return {"tickers": tickers, "amounts": {k: round(v, 2) for k, v in per.items()},
            "risk_level": _risk_level(text), "total_amount": total}


# --------------------------------------------------------------------------- LLM extraction
_FEW_SHOT = [
    ("I have 50000 rupees and want to put it in TCS and Infosys, low risk",
     {"tickers": ["TCS.NS", "INFY.NS"], "amounts": {"TCS.NS": 25000, "INFY.NS": 25000}, "risk_level": "Low"}),
    ("Mere paas 50,000 rupaye hain, TCS aur Infosys mein lagana hai, risk kam chahiye.",
     {"tickers": ["TCS.NS", "INFY.NS"], "amounts": {"TCS.NS": 25000, "INFY.NS": 25000}, "risk_level": "Low"}),
    ("2 lakh hain, HDFC bank mein 1 lakh aur Reliance mein 1 lakh, zyada risk le sakta hoon",
     {"tickers": ["HDFCBANK.NS", "RELIANCE.NS"], "amounts": {"HDFCBANK.NS": 100000, "RELIANCE.NS": 100000},
      "risk_level": "High"}),
    ("मला ५०,००० रुपये गुंतवायचे आहेत, टीसीएस आणि इन्फोसिसमध्ये, जोखीम कमी हवी",
     {"tickers": ["TCS.NS", "INFY.NS"], "amounts": {"TCS.NS": 25000, "INFY.NS": 25000}, "risk_level": "Low"}),
    ("Put 10 thousand each in Apple and Microsoft, medium risk",
     {"tickers": ["AAPL", "MSFT"], "amounts": {"AAPL": 10000, "MSFT": 10000}, "risk_level": "Medium"}),
    ("I want to invest in some gold and Wipro",
     {"tickers": ["GOLDBEES.NS", "WIPRO.NS"], "amounts": {}, "risk_level": None}),
]


def _universe_map_text() -> str:
    return "; ".join(f"{e['ticker']}={e['name']}" for e in load_universe() if e.get("type") != "index")


def _llm_prompt(text: str) -> tuple[str, str]:
    shots = "\n".join(f"Input: {i}\nOutput: {__import__('json').dumps(o, ensure_ascii=False)}" for i, o in _FEW_SHOT)
    system = ("You convert a spoken or typed investment request (English, Hindi, Hinglish or Marathi) into JSON. "
              "Output ONLY valid JSON, no prose.")
    prompt = (
        "Extract the portfolio the user describes.\n"
        'Schema: {"tickers": [string], "amounts": {ticker: number}, "risk_level": "Low"|"Medium"|"High"|null, '
        '"unresolved": [string]}\n'
        "Rules:\n"
        "- Map company names to tickers ONLY from this list (name=ticker): " + _universe_map_text() + "\n"
        "- Names not in the list go to \"unresolved\" (as the user said them); never invent tickers.\n"
        "- Currency is INR by default. Understand: 50 hazaar=50000, 1 lakh=100000, 2 lakh=200000, 1 crore=10000000, "
        "'5 thousand'=5000, 'k'=thousand, Devanagari digits.\n"
        "- If one total amount is given for several stocks, split it equally. If amounts are given per stock, use them.\n"
        "- Risk words: kam/low/safe/कम/कमी -> Low; madhyam/medium/moderate/मध्यम -> Medium; zyada/high/aggressive/"
        "ज़्यादा/जास्त -> High.\n"
        "- Use null / empty values for anything the user did not say.\n\n"
        f"Examples:\n{shots}\n\nInput: {text}\nOutput:"
    )
    return system, prompt


def _validate(obj: Any) -> dict:
    """Coerce model output to the contract; unknown tickers move to ``unresolved``."""
    uni = _universe_tickers()
    obj = obj if isinstance(obj, dict) else {}
    tickers, unresolved = [], [str(x) for x in (obj.get("unresolved") or []) if str(x).strip()]
    for t in obj.get("tickers") or []:
        t = str(t).strip().upper()
        if t in uni:
            if t not in tickers:
                tickers.append(t)
        elif t + ".NS" in uni:
            if t + ".NS" not in tickers:
                tickers.append(t + ".NS")
        elif t:
            unresolved.append(t)
    amounts: dict[str, float] = {}
    raw = obj.get("amounts") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            kk = str(k).strip().upper()
            kk = kk if kk in uni else (kk + ".NS" if kk + ".NS" in uni else None)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if kk and v > 0:
                amounts[kk] = v
                if kk not in tickers:
                    tickers.append(kk)
    level = obj.get("risk_level")
    level = next((l for l in LEVELS if str(level).strip().lower() == l.lower()), None) if level else None
    # models sometimes list a word they DID map (e.g. "सोने" -> GOLDBEES.NS) as unresolved: drop those
    resolved = set(tickers)
    unresolved = [u for u in dict.fromkeys(unresolved)
                  if not ({t for _, _, t in _find_companies(u)} & resolved)]
    return {"tickers": tickers, "amounts": amounts, "risk_level": level, "unresolved": unresolved}


def _confidence(r: dict, source: str) -> float:
    if not r["tickers"]:
        return 0.0
    c = 0.5 + (0.2 if r["amounts"] else 0.0) + (0.15 if r["risk_level"] else 0.0) + (0.15 if not r["unresolved"] else 0.0)
    if source == "rules":
        c -= 0.1
    return round(max(0.0, min(1.0, c)), 2)


def extract_portfolio(text: str) -> dict:
    """``{"tickers", "amounts", "risk_level", "unresolved", "confidence", "source"}`` from free text.

    ``amounts`` are INR per ticker (an equal split when only a total was given); ``risk_level`` is
    "Low" | "Medium" | "High" | None. ``source`` is "llm" or "rules". Never raises.
    """
    text = (text or "").strip()
    empty = {"tickers": [], "amounts": {}, "risk_level": None, "unresolved": [], "confidence": 0.0, "source": "rules"}
    if not text:
        return empty
    rules = _rule_extract(text)
    result, source = None, "rules"
    try:
        system, prompt = _llm_prompt(text)
        out = llm.ask_llm(EXTRACT_TASK, prompt, system=system, json_mode=True, lang="en", max_tokens=600)
        parsed = _validate(llm.parse_json(out["text"]))
        if parsed["tickers"]:
            result, source = parsed, "llm"
    except (llm.LLMUnavailable, ValueError) as exc:
        log.info("LLM extraction unavailable (%s); using the rule-based parser", exc)
    if result is None:
        result = {"tickers": rules["tickers"], "amounts": rules["amounts"], "risk_level": rules["risk_level"],
                  "unresolved": []}
    else:                                              # fill gaps the LLM left but the rules found unambiguously
        if result["risk_level"] is None and rules["risk_level"]:
            result["risk_level"] = rules["risk_level"]
        if not result["amounts"] and rules["amounts"] and set(rules["amounts"]) <= set(result["tickers"]):
            result["amounts"] = rules["amounts"]
    result["confidence"] = _confidence(result, source)
    result["source"] = source
    return result


# --------------------------------------------------------------------------- saved demo samples
def demo_samples() -> list[dict]:
    """Saved utterances with their expected extraction (``ml/voice_samples.json``): lets the UI offer a
    "play demo sample" button that works even when STT and the LLM are both down."""
    import json
    from pathlib import Path

    try:
        return json.loads((Path(__file__).with_name("voice_samples.json")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
