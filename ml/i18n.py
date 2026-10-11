"""
ml/i18n.py — Internationalization labels for English, Hindi, and Marathi.
P3 calls t(key, lang) everywhere instead of hardcoding strings.
"""

_LABELS = {
    "en": {
        # Sidebar
        "sidebar_title": "Portfolio Settings",
        "language": "Language",
        "offline_mode": "Offline Mode",
        "llm_usage": "LLM Usage Panel",
        "voice_input": "Voice Input",
        "assets": "Select Assets",
        "free_text_ticker": "Add ticker manually",
        "amounts_weights": "Amounts / Weights",
        "amount_col": "Amount (₹)",
        "weight_col": "Weight (%)",
        "total_investment": "Total Investment (₹)",
        "risk_level": "Risk Preference",
        "Low": "Low",
        "Medium": "Medium",
        "High": "High",
        "date_range": "Date Range",
        "start_date": "Start Date",
        "end_date": "End Date",
        "benchmark": "Benchmark",
        "advanced": "Advanced",
        "risk_free_rate": "Risk-Free Rate (%)",
        "session_name": "Portfolio Name",
        "save_portfolio": "💾 Save Portfolio",
        "load_portfolio": "📂 Load Portfolio",
        "export_pdf": "📄 Export PDF",
        "record_audio": "🎤 Record Audio",
        "fill_portfolio": "Fill Portfolio from Transcript",
        "transcript": "Transcript (editable)",
        "unresolved": "Unresolved tickers",

        # Tabs
        "tab_risk": "📊 Risk",
        "tab_allocation": "⚖️ Allocation",
        "tab_simulate": "📈 Simulate",
        "tab_track": "🔍 Track",
        "tab_compare": "🔁 Compare",
        "tab_scenarios": "🌩 Scenarios",
        "tab_news": "📰 Market Intel",

        # Risk tab
        "risk_label": "Risk Label",
        "class_probs": "Classification Probabilities",
        "metric_return": "Ann. Return",
        "metric_vol": "Ann. Volatility",
        "metric_drawdown": "Max Drawdown",
        "metric_sharpe": "Sharpe Ratio",
        "metric_sortino": "Sortino Ratio",
        "metric_cvar": "CVaR 95%",
        "correlation_heatmap": "Correlation Heatmap",
        "risk_contribution": "Risk Contribution vs Weight",
        "top_factors": "Top Risk Factors",
        "explanation": "Explanation",

        # Allocation tab
        "original_weights": "Original Weights",
        "recommended_weights": "Recommended Weights",
        "weight_delta": "Δ Weight",
        "before_metrics": "Before Optimization",
        "after_metrics": "After Optimization",
        "reason": "Rationale",
        "tradeoffs": "Trade-offs",
        "allocation_explanation": "Detailed Explanation",
        "change_risk_update": "Change risk level to update allocation instantly.",

        # Simulate tab
        "regime_presets": "Regime Presets",
        "covid_crash": "COVID Crash (2020)",
        "bull_2021": "2021 Bull Run",
        "bear_2022": "2022 Bear / Volatile",
        "last_1y": "Last 1 Year",
        "custom": "Custom Range",
        "growth_chart": "Growth of ₹100",
        "log_scale": "Log Scale",
        "underwater_chart": "Drawdown (Underwater)",
        "compare_assets": "Compare Alternate Assets",

        # Compare tab
        "compare_title": "Strategy Comparison",
        "growth_comparison": "Growth Comparison",
        "risk_return_scatter": "Risk-Return Scatter",
        "verdict": "Verdict",
        "limitations": "Limitations: Past performance does not guarantee future results.",

        # Track tab
        "holdings_editor": "Holdings Editor",
        "ticker_col": "Ticker",
        "qty_col": "Quantity",
        "buy_price_col": "Buy Price",
        "current_price_col": "Current Price",
        "value_col": "Value",
        "pnl_col": "P&L",
        "pnl_pct_col": "P&L %",
        "refresh": "🔄 Refresh",
        "as_of": "As of",
        "data_delayed": "Market data may be delayed by 15-20 minutes.",
        "total_value": "Total Value",
        "total_pnl": "Total P&L",

        # Scenarios tab
        "scenario_picker": "Select Scenario",
        "crash_20": "Market Crash -20%",
        "crash_30": "Market Crash -30%",
        "recovery_15": "Recovery +15%",
        "vol_x2": "Volatility ×2",
        "sector_downturn": "Sector Downturn",
        "sector_select": "Sector",
        "downturn_pct": "Downturn (%)",
        "value_before": "Value Before",
        "value_after": "Value After",
        "pct_change": "% Change",
        "new_risk_label": "New Risk Label",
        "new_volatility": "New Volatility",
        "monte_carlo_chart": "Monte Carlo Fan Chart (1,000 paths)",
        "prob_of_loss": "Probability of Loss",
        "expected_shortfall": "Expected Shortfall",

        # News tab
        "news_disclaimer": "⚠️ Supplementary AI insight. It does not change the ML risk score.",
        "generate_insight": "Generate Insight",
        "sentiment": "Sentiment",
        "sector_trends": "Sector Trends",
        "risks": "Key Risks",
        "headlines": "Headlines",
        "source_live": "Source: Live",
        "source_cache": "Source: Cache",
        "source_offline": "Source: Offline",

        # LLM usage
        "provider": "Provider",
        "calls": "Calls",
        "cache_hits": "Cache Hits",
        "failures": "Failures",

        # Footer
        "footer": "Educational tool, not financial advice. Past performance does not guarantee future results.",

        # Errors / warnings
        "invalid_ticker": "Invalid ticker: {}. Please check and try again.",
        "mixed_currencies": "⚠️ Mixed currencies detected. Comparisons shown in percentage terms.",
        "using_fallback": "ℹ️ Using saved/cached data (live fetch failed).",
        "llm_offline": "ℹ️ LLM result from cache (offline mode or quota exceeded).",
        "no_data": "No data available. Please check your inputs.",
        "retry": "Please retry or check your internet connection.",
        "short_history": "⚠️ {} has less than 1 year of history. Results may be less reliable.",
        "single_asset": "ℹ️ Single-asset portfolio detected. Diversification metrics not applicable.",
        "dropped_tickers": "⚠️ Some tickers were dropped due to insufficient data: {}",
    },

    "hi": {
        # Sidebar
        "sidebar_title": "पोर्टफोलियो सेटिंग्स",
        "language": "भाषा",
        "offline_mode": "ऑफलाइन मोड",
        "llm_usage": "LLM उपयोग पैनल",
        "voice_input": "आवाज़ इनपुट",
        "assets": "संपत्तियां चुनें",
        "free_text_ticker": "टिकर मैन्युअल जोड़ें",
        "amounts_weights": "राशि / भार",
        "amount_col": "राशि (₹)",
        "weight_col": "भार (%)",
        "total_investment": "कुल निवेश (₹)",
        "risk_level": "जोखिम प्राथमिकता",
        "Low": "कम",
        "Medium": "मध्यम",
        "High": "अधिक",
        "date_range": "तारीख सीमा",
        "start_date": "शुरुआती तारीख",
        "end_date": "अंतिम तारीख",
        "benchmark": "बेंचमार्क",
        "advanced": "उन्नत",
        "risk_free_rate": "जोखिम मुक्त दर (%)",
        "session_name": "पोर्टफोलियो नाम",
        "save_portfolio": "💾 पोर्टफोलियो सहेजें",
        "load_portfolio": "📂 पोर्टफोलियो लोड करें",
        "export_pdf": "📄 PDF निर्यात करें",
        "record_audio": "🎤 ऑडियो रिकॉर्ड करें",
        "fill_portfolio": "ट्रांसक्रिप्ट से पोर्टफोलियो भरें",
        "transcript": "ट्रांसक्रिप्ट (संपादन योग्य)",
        "unresolved": "अज्ञात टिकर",

        # Tabs
        "tab_risk": "📊 जोखिम",
        "tab_allocation": "⚖️ आवंटन",
        "tab_simulate": "📈 सिमुलेशन",
        "tab_track": "🔍 ट्रैक",
        "tab_compare": "🔁 तुलना",
        "tab_scenarios": "🌩 परिदृश्य",
        "tab_news": "📰 बाज़ार जानकारी",

        # Risk tab
        "risk_label": "जोखिम श्रेणी",
        "class_probs": "वर्गीकरण संभावनाएं",
        "metric_return": "वार्षिक रिटर्न",
        "metric_vol": "वार्षिक अस्थिरता",
        "metric_drawdown": "अधिकतम गिरावट",
        "metric_sharpe": "शार्प अनुपात",
        "metric_sortino": "सोर्टिनो अनुपात",
        "metric_cvar": "CVaR 95%",
        "correlation_heatmap": "सहसंबंध हीटमैप",
        "risk_contribution": "जोखिम योगदान बनाम भार",
        "top_factors": "मुख्य जोखिम कारक",
        "explanation": "स्पष्टीकरण",

        # Allocation tab
        "original_weights": "मूल भार",
        "recommended_weights": "अनुशंसित भार",
        "weight_delta": "Δ भार",
        "before_metrics": "अनुकूलन से पहले",
        "after_metrics": "अनुकूलन के बाद",
        "reason": "कारण",
        "tradeoffs": "समझौते",
        "allocation_explanation": "विस्तृत स्पष्टीकरण",
        "change_risk_update": "जोखिम स्तर बदलें और आवंटन तुरंत अपडेट होगा।",

        # Simulate tab
        "regime_presets": "शासन पूर्व-निर्धारित",
        "covid_crash": "COVID क्रैश (2020)",
        "bull_2021": "2021 तेजी",
        "bear_2022": "2022 मंदी / अस्थिर",
        "last_1y": "पिछले 1 वर्ष",
        "custom": "कस्टम सीमा",
        "growth_chart": "₹100 की वृद्धि",
        "log_scale": "लॉग स्केल",
        "underwater_chart": "गिरावट (अंडरवाटर)",
        "compare_assets": "वैकल्पिक संपत्तियों की तुलना",

        # Compare tab
        "compare_title": "रणनीति तुलना",
        "growth_comparison": "वृद्धि तुलना",
        "risk_return_scatter": "जोखिम-रिटर्न स्कैटर",
        "verdict": "निर्णय",
        "limitations": "सीमाएं: पिछला प्रदर्शन भविष्य की गारंटी नहीं है।",

        # Track tab
        "holdings_editor": "होल्डिंग्स संपादक",
        "ticker_col": "टिकर",
        "qty_col": "मात्रा",
        "buy_price_col": "खरीद मूल्य",
        "current_price_col": "वर्तमान मूल्य",
        "value_col": "मूल्य",
        "pnl_col": "P&L",
        "pnl_pct_col": "P&L %",
        "refresh": "🔄 रिफ्रेश",
        "as_of": "इस समय तक",
        "data_delayed": "बाज़ार डेटा 15-20 मिनट देरी से हो सकता है।",
        "total_value": "कुल मूल्य",
        "total_pnl": "कुल P&L",

        # Scenarios tab
        "scenario_picker": "परिदृश्य चुनें",
        "crash_20": "बाज़ार क्रैश -20%",
        "crash_30": "बाज़ार क्रैश -30%",
        "recovery_15": "रिकवरी +15%",
        "vol_x2": "अस्थिरता ×2",
        "sector_downturn": "क्षेत्र गिरावट",
        "sector_select": "क्षेत्र",
        "downturn_pct": "गिरावट (%)",
        "value_before": "पहले का मूल्य",
        "value_after": "बाद का मूल्य",
        "pct_change": "% परिवर्तन",
        "new_risk_label": "नई जोखिम श्रेणी",
        "new_volatility": "नई अस्थिरता",
        "monte_carlo_chart": "मोंटे कार्लो फैन चार्ट (1,000 पथ)",
        "prob_of_loss": "नुकसान की संभावना",
        "expected_shortfall": "अपेक्षित कमी",

        # News tab
        "news_disclaimer": "⚠️ सहायक AI अंतर्दृष्टि। यह ML जोखिम स्कोर नहीं बदलती।",
        "generate_insight": "अंतर्दृष्टि उत्पन्न करें",
        "sentiment": "भावना",
        "sector_trends": "क्षेत्र रुझान",
        "risks": "मुख्य जोखिम",
        "headlines": "समाचार शीर्षक",
        "source_live": "स्रोत: लाइव",
        "source_cache": "स्रोत: कैश",
        "source_offline": "स्रोत: ऑफलाइन",

        # LLM usage
        "provider": "प्रदाता",
        "calls": "कॉल",
        "cache_hits": "कैश हिट",
        "failures": "विफलताएं",

        # Footer
        "footer": "शैक्षिक उपकरण, वित्तीय सलाह नहीं। पिछला प्रदर्शन भविष्य के परिणामों की गारंटी नहीं देता।",

        # Errors
        "invalid_ticker": "अमान्य टिकर: {}। कृपया जांचें और पुनः प्रयास करें।",
        "mixed_currencies": "⚠️ मिश्रित मुद्राएं पाई गईं। तुलना प्रतिशत में दिखाई गई है।",
        "using_fallback": "ℹ️ सहेजा गया डेटा उपयोग हो रहा है (लाइव फ़ेच विफल)।",
        "llm_offline": "ℹ️ LLM परिणाम कैश से (ऑफलाइन मोड या कोटा समाप्त)।",
        "no_data": "कोई डेटा उपलब्ध नहीं। कृपया अपने इनपुट जांचें।",
        "retry": "कृपया पुनः प्रयास करें या इंटरनेट कनेक्शन जांचें।",
        "short_history": "⚠️ {} का 1 वर्ष से कम इतिहास है। परिणाम कम विश्वसनीय हो सकते हैं।",
        "single_asset": "ℹ️ एकल-संपत्ति पोर्टफोलियो। विविधीकरण मेट्रिक्स लागू नहीं।",
        "dropped_tickers": "⚠️ अपर्याप्त डेटा के कारण कुछ टिकर हटाए गए: {}",
    },

    "mr": {
        # Sidebar
        "sidebar_title": "पोर्टफोलिओ सेटिंग्ज",
        "language": "भाषा",
        "offline_mode": "ऑफलाइन मोड",
        "llm_usage": "LLM वापर पॅनेल",
        "voice_input": "आवाज इनपुट",
        "assets": "मालमत्ता निवडा",
        "free_text_ticker": "टिकर मॅन्युअली जोडा",
        "amounts_weights": "रक्कम / वजन",
        "amount_col": "रक्कम (₹)",
        "weight_col": "वजन (%)",
        "total_investment": "एकूण गुंतवणूक (₹)",
        "risk_level": "जोखीम प्राधान्य",
        "Low": "कमी",
        "Medium": "मध्यम",
        "High": "जास्त",
        "date_range": "तारीख श्रेणी",
        "start_date": "सुरुवात तारीख",
        "end_date": "शेवट तारीख",
        "benchmark": "बेंचमार्क",
        "advanced": "प्रगत",
        "risk_free_rate": "जोखीम-मुक्त दर (%)",
        "session_name": "पोर्टफोलिओ नाव",
        "save_portfolio": "💾 पोर्टफोलिओ जतन करा",
        "load_portfolio": "📂 पोर्टफोलिओ लोड करा",
        "export_pdf": "📄 PDF निर्यात करा",
        "record_audio": "🎤 ऑडिओ रेकॉर्ड करा",
        "fill_portfolio": "ट्रान्सक्रिप्टमधून पोर्टफोलिओ भरा",
        "transcript": "ट्रान्सक्रिप्ट (संपादन करण्यायोग्य)",
        "unresolved": "अज्ञात टिकर",

        # Tabs
        "tab_risk": "📊 जोखीम",
        "tab_allocation": "⚖️ वाटप",
        "tab_simulate": "📈 सिम्युलेशन",
        "tab_track": "🔍 ट्रॅक",
        "tab_compare": "🔁 तुलना",
        "tab_scenarios": "🌩 परिस्थिती",
        "tab_news": "📰 बाजार माहिती",

        # Risk tab
        "risk_label": "जोखीम श्रेणी",
        "class_probs": "वर्गीकरण संभाव्यता",
        "metric_return": "वार्षिक परतावा",
        "metric_vol": "वार्षिक अस्थिरता",
        "metric_drawdown": "कमाल घसरण",
        "metric_sharpe": "शार्प गुणोत्तर",
        "metric_sortino": "सोर्टिनो गुणोत्तर",
        "metric_cvar": "CVaR 95%",
        "correlation_heatmap": "सहसंबंध हीटमॅप",
        "risk_contribution": "जोखीम योगदान विरुद्ध वजन",
        "top_factors": "मुख्य जोखीम घटक",
        "explanation": "स्पष्टीकरण",

        # Allocation tab
        "original_weights": "मूळ वजन",
        "recommended_weights": "शिफारस केलेले वजन",
        "weight_delta": "Δ वजन",
        "before_metrics": "ऑप्टिमायझेशनपूर्वी",
        "after_metrics": "ऑप्टिमायझेशननंतर",
        "reason": "कारण",
        "tradeoffs": "तडजोडी",
        "allocation_explanation": "तपशीलवार स्पष्टीकरण",
        "change_risk_update": "जोखीम पातळी बदला आणि वाटप लगेच अपडेट होईल।",

        # Simulate tab
        "regime_presets": "शासन पूर्वनिर्धारित",
        "covid_crash": "COVID क्रॅश (2020)",
        "bull_2021": "2021 तेजी",
        "bear_2022": "2022 मंदी / अस्थिर",
        "last_1y": "मागील 1 वर्ष",
        "custom": "सानुकूल श्रेणी",
        "growth_chart": "₹100 ची वाढ",
        "log_scale": "लॉग स्केल",
        "underwater_chart": "घसरण (अंडरवॉटर)",
        "compare_assets": "पर्यायी मालमत्तांची तुलना",

        # Compare tab
        "compare_title": "धोरण तुलना",
        "growth_comparison": "वाढ तुलना",
        "risk_return_scatter": "जोखीम-परतावा स्कॅटर",
        "verdict": "निर्णय",
        "limitations": "मर्यादा: मागील कामगिरी भविष्यातील परिणामांची हमी नाही।",

        # Track tab
        "holdings_editor": "होल्डिंग्ज संपादक",
        "ticker_col": "टिकर",
        "qty_col": "प्रमाण",
        "buy_price_col": "खरेदी किंमत",
        "current_price_col": "सध्याची किंमत",
        "value_col": "मूल्य",
        "pnl_col": "P&L",
        "pnl_pct_col": "P&L %",
        "refresh": "🔄 रिफ्रेश",
        "as_of": "पर्यंत",
        "data_delayed": "बाजार डेटा 15-20 मिनिटे उशीरा असू शकतो।",
        "total_value": "एकूण मूल्य",
        "total_pnl": "एकूण P&L",

        # Scenarios tab
        "scenario_picker": "परिस्थिती निवडा",
        "crash_20": "बाजार क्रॅश -20%",
        "crash_30": "बाजार क्रॅश -30%",
        "recovery_15": "रिकव्हरी +15%",
        "vol_x2": "अस्थिरता ×2",
        "sector_downturn": "क्षेत्र घसरण",
        "sector_select": "क्षेत्र",
        "downturn_pct": "घसरण (%)",
        "value_before": "आधीचे मूल्य",
        "value_after": "नंतरचे मूल्य",
        "pct_change": "% बदल",
        "new_risk_label": "नवीन जोखीम श्रेणी",
        "new_volatility": "नवीन अस्थिरता",
        "monte_carlo_chart": "मोंटे कार्लो फॅन चार्ट (1,000 मार्ग)",
        "prob_of_loss": "नुकसानाची शक्यता",
        "expected_shortfall": "अपेक्षित तूट",

        # News tab
        "news_disclaimer": "⚠️ सहायक AI अंतर्दृष्टी। हे ML जोखीम स्कोर बदलत नाही।",
        "generate_insight": "अंतर्दृष्टी निर्माण करा",
        "sentiment": "भावना",
        "sector_trends": "क्षेत्र कल",
        "risks": "मुख्य जोखीम",
        "headlines": "बातम्यांचे शीर्षक",
        "source_live": "स्रोत: लाइव्ह",
        "source_cache": "स्रोत: कॅश",
        "source_offline": "स्रोत: ऑफलाइन",

        # LLM usage
        "provider": "प्रदाता",
        "calls": "कॉल",
        "cache_hits": "कॅश हिट",
        "failures": "अपयश",

        # Footer
        "footer": "शैक्षणिक साधन, आर्थिक सल्ला नाही। मागील कामगिरी भविष्यातील परिणामांची हमी देत नाही।",

        # Errors
        "invalid_ticker": "अवैध टिकर: {}। कृपया तपासा आणि पुन्हा प्रयत्न करा।",
        "mixed_currencies": "⚠️ मिश्रित चलने आढळली. तुलना टक्केवारीत दाखवली आहे।",
        "using_fallback": "ℹ️ जतन केलेला डेटा वापरला जात आहे (लाइव्ह फेच अयशस्वी)।",
        "llm_offline": "ℹ️ LLM निकाल कॅशमधून (ऑफलाइन मोड किंवा कोटा संपला)।",
        "no_data": "कोणताही डेटा उपलब्ध नाही. कृपया तुमचे इनपुट तपासा।",
        "retry": "कृपया पुन्हा प्रयत्न करा किंवा इंटरनेट कनेक्शन तपासा।",
        "short_history": "⚠️ {} चा 1 वर्षापेक्षा कमी इतिहास आहे. परिणाम कमी विश्वसनीय असू शकतात।",
        "single_asset": "ℹ️ एकल-मालमत्ता पोर्टफोलिओ. विविधीकरण मेट्रिक्स लागू नाहीत।",
        "dropped_tickers": "⚠️ अपुऱ्या डेटामुळे काही टिकर काढले गेले: {}",
    },
}


def t(key: str, lang: str = "en", *args) -> str:
    """Return translated label for key in the given language."""
    labels = _LABELS.get(lang, _LABELS["en"])
    text = labels.get(key, _LABELS["en"].get(key, key))
    if args:
        try:
            text = text.format(*args)
        except Exception:
            pass
    return text


def set_offline(offline: bool) -> None:
    """Stub: sets offline mode flag. P2 implements the real version."""
    import streamlit as st
    st.session_state["offline"] = offline


def usage_stats() -> dict:
    """Return LLM usage statistics. Falls back to mock when P2 not available."""
    import streamlit as st
    return st.session_state.get("_llm_usage", {
        "gemini":  {"calls": 0, "cache_hits": 0, "failures": 0},
        "groq":    {"calls": 0, "cache_hits": 0, "failures": 0},
        "mistral": {"calls": 0, "cache_hits": 0, "failures": 0},
    })
