# app.py — Google Trends control panel (enhanced + Difficulty Proxy)
# Upgrades: progress bar, caching, retry/backoff, per-keyword fetch, resample & export,
#           difficulty proxy per keyword (base interest + trend slope + rising intensity)

import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
from pytrends.request import TrendReq

# -------------------------
# PAGE / THEME
# -------------------------
st.set_page_config(page_title="Trends Control Panel", layout="wide")
st.title("Google Trends Control Panel")

# -------------------------
# HELPERS / CONSTANTS
# -------------------------
COUNTRIES = {
    "Worldwide": "",
    "United Kingdom (GB)": "GB",
    "United States (US)": "US",
    "Ireland (IE)": "IE",
    "Netherlands (NL)": "NL",
    "Germany (DE)": "DE",
    "France (FR)": "FR",
    "Spain (ES)": "ES",
    "Italy (IT)": "IT",
    "Australia (AU)": "AU",
    "Canada (CA)": "CA",
}

PN_OPTIONS = {
    "United Kingdom": "united_kingdom",
    "United States": "united_states",
    "Ireland": "ireland",
    "Netherlands": "netherlands",
    "Germany": "germany",
    "France": "france",
    "Spain": "spain",
    "Italy": "italy",
    "Australia": "australia",
    "Canada": "canada",
    "India": "india",
    "Japan": "japan",
}

LANGS = {
    "English (UK)": "en-GB",
    "English (US)": "en-US",
    "Dutch": "nl-NL",
    "German": "de-DE",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "Italian": "it-IT",
}

TIMEFRAME_PRESETS = ["today 3-m", "today 12-m", "today 5-y", "all", "Custom…"]

RESAMPLE = {
    "No resample (native)": None,
    "Weekly average": "W",
    "Monthly average": "MS",  # month start
}

def make_timeframe(preset, start=None, end=None):
    if preset != "Custom…":
        return preset
    return f"{start:%Y-%m-%d} {end:%Y-%m-%d}"

def resample_df(df, rule):
    if not rule or df.empty:
        return df
    return df.resample(rule).mean().round(2)

def download_csv(df, label, filename):
    if df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

def band_from_score(score):
    if score is None:
        return "N/A"
    if score < 30:
        return "Easy"
    if score < 60:
        return "Medium"
    return "Hard"

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
with st.sidebar:
    st.header("Settings")

    lang = st.selectbox("Language (UI only)", list(LANGS.keys()), index=0)
    hl = LANGS[lang]

    geo_name = st.selectbox("Region (geo)", list(COUNTRIES.keys()), index=1)
    geo = COUNTRIES[geo_name]
    geo_override = st.text_input("Override geo (optional, e.g. GB, US)")
    if geo_override.strip():
        geo = geo_override.strip()

    tf_preset = st.selectbox("Timeframe", TIMEFRAME_PRESETS, index=1)
    if tf_preset == "Custom…":
        default_end = dt.date.today()
        default_start = default_end - relativedelta(years=1)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=default_end)
        timeframe = make_timeframe(tf_preset, start_date, end_date)
    else:
        timeframe = tf_preset

    gran = st.selectbox("Granularity", list(RESAMPLE.keys()), index=0)
    gran_rule = RESAMPLE[gran]

    st.divider()
    st.caption("Fetch behaviour")
    rpm = st.slider("Max requests per minute", 6, 60, 20,
                    help="Adds a small delay between requests to be kind to rate limits.")
    per_request_delay = 60.0 / float(rpm)
    max_retries = st.slider("Max retries on error", 0, 5, 2)
    backoff_base = st.slider("Backoff (seconds)", 1, 10, 3,
                             help="Wait time grows with each retry.")
    show_raw = st.toggle("Show raw results", value=False)

# -------------------------
# CACHED FETCHES WITH RETRY
# -------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_interest_for_term(term, timeframe, geo, hl):
    """Fetch interest_over_time for a single keyword (cached)."""
    py = TrendReq(hl=hl, tz=0)
    py.build_payload(kw_list=[term], timeframe=timeframe, geo=geo)
    df = py.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.columns = [term]  # rename single column for join
    return df

def fetch_with_retry(term, timeframe, geo, hl, max_retries, backoff_base):
    tries = 0
    while True:
        try:
            return cached_interest_for_term(term, timeframe, geo, hl)
        except Exception as e:
            tries += 1
            if tries > max_retries:
                raise
            time.sleep(backoff_base * tries)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_related_queries(keywords, timeframe, geo, hl):
    """Fetch related queries dict for a list (cached as a whole)."""
    py = TrendReq(hl=hl, tz=0)
    py.build_payload(kw_list=keywords, timeframe=timeframe, geo=geo)
    return py.related_queries() or {}

# -------------------------
# DIFFICULTY PROXY
# -------------------------
def slope_score_from_series(s: pd.Series) -> float:
    """
    Fit a simple linear regression to the interest series (index must be DatetimeIndex).
    Returns a 0–100 score scaled from the slope.
    """
    if s is None or s.empty:
        return 0.0
    # x as 0..n-1 (weekly-ish cadence in Trends); y as values
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    if len(x) < 3 or np.all(y == y[0]):
        return 0.0
    # slope via least squares
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    # Scale: map typical slopes to 0–100 using tanh for stability
    scale = 15.0  # adjust sensitivity; higher = less sensitive
    score = 50.0 + 50.0 * np.tanh(slope * scale)
    return float(np.clip(score, 0, 100))

def rising_intensity_from_rq(rq_entry) -> float:
    """
    Given related_queries()[kw] entry, compute mean 'value' of Rising queries.
    Returns 0–100 (already on a 0–100 relative scale from Google).
    """
    if not rq_entry:
        return 0.0
    rising_df = rq_entry.get("rising")
    if rising_df is None or rising_df.empty or "value" not in rising_df.columns:
        return 0.0
    # mean of top 10 rising values
    vals = rising_df["value"].astype(float).values[:10]
    return float(np.clip(np.mean(vals), 0, 100))

def difficulty_proxy(base_interest: float, slope_score: float, rising_intensity: float) -> float:
    """
    Combine components into a single 0–100 difficulty score.
    Tunable weights.
    """
    w_interest = 0.40
    w_rising   = 0.40
    w_slope    = 0.20
    score = w_interest * base_interest + w_rising * rising_intensity + w_slope * slope_score
    return float(np.clip(score, 0, 100))

# -------------------------
# UI — KEYWORDS INPUT
# -------------------------
st.markdown("Enter keywords (comma‑separated). Each term is fetched separately, then combined.")
terms_raw = st.text_input("Search terms", value="web design, website builder")
keywords = [t.strip() for t in terms_raw.split(",") if t.strip()]
keywords = list(dict.fromkeys(keywords))  # de‑dupe in order

tabs = st.tabs(["Interest over time", "Related queries", "Trending searches", "Suggestions", "Difficulty proxy"])

# -------------------------
# TAB 1 — INTEREST OVER TIME
# -------------------------
with tabs[0]:
    st.subheader("Interest over time")

    col1, col2 = st.columns([1,1])
    with col1:
        run_iot = st.button("Fetch interest over time")
    with col2:
        st.caption("Per‑keyword fetch with retry, progress bar, and caching.")

    if run_iot:
        if not keywords:
            st.error("Add at least one keyword.")
        else:
            progress = st.progress(0)
            status = st.empty()

            combined = None
            errors = {}
            total = len(keywords)

            for i, term in enumerate(keywords, start=1):
                status.write(f"Fetching **{term}** ({i}/{total}) …")
                try:
                    df = fetch_with_retry(term, timeframe, geo, hl,
                                          max_retries=max_retries, backoff_base=backoff_base)
                    combined = df if combined is None else combined.join(df, how="outer")
                except Exception as e:
                    errors[term] = str(e)
                progress.progress(int(i * 100 / total))
                time.sleep(per_request_delay)

            progress.empty(); status.empty()

            if combined is None or combined.empty:
                st.warning("No data returned. Try a broader timeframe or different terms.")
            else:
                combined = combined.sort_index()
                combined_resampled = resample_df(combined, gran_rule)

                st.line_chart(combined_resampled)
                st.dataframe(combined_resampled.reset_index(), use_container_width=True)
                download_csv(combined_resampled, "Download CSV (interest over time)", "interest_over_time.csv")

                if show_raw:
                    with st.expander("Raw (combined)", expanded=False):
                        st.write(combined)

            if errors:
                st.warning("Some terms failed. Details:")
                st.json(errors)

# -------------------------
# TAB 2 — RELATED QUERIES
# -------------------------
with tabs[1]:
    st.subheader("Related queries (Top & Rising)")
    run_rq = st.button("Fetch related queries")
    if run_rq:
        if not keywords:
            st.error("Add at least one keyword.")
        else:
            try:
                rq = cached_related_queries(tuple(keywords), timeframe, geo, hl)
                if not rq:
                    st.warning("No related queries returned.")
                else:
                    for kw in keywords:
                        st.markdown(f"**{kw}**")
                        d = rq.get(kw, {}) or {}
                        top_df = d.get("top")
                        rising_df = d.get("rising")

                        if top_df is not None and not top_df.empty:
                            st.write("Top")
                            st.dataframe(top_df, use_container_width=True, height=240)
                            download_csv(top_df, f"Download CSV (top) — {kw}", f"related_top_{kw}.csv")
                            if show_raw: st.caption(top_df.to_json(orient="records"))
                        else:
                            st.write("No Top results.")

                        if rising_df is not None and not rising_df.empty:
                            st.write("Rising")
                            st.dataframe(rising_df, use_container_width=True, height=240)
                            download_csv(rising_df, f"Download CSV (rising) — {kw}", f"related_rising_{kw}.csv")
                            if show_raw: st.caption(rising_df.to_json(orient="records"))
                        else:
                            st.write("No Rising results.")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------
# TAB 3 — TRENDING SEARCHES
# -------------------------
with tabs[2]:
    st.subheader("Trending searches (daily)")
    pn_name = st.selectbox("Country", list(PN_OPTIONS.keys()), index=0, key="pn")
    run_trending = st.button("Fetch trending searches")
    if run_trending:
        try:
            py = TrendReq(hl=hl, tz=0)
            ts = py.trending_searches(pn=PN_OPTIONS[pn_name])
            if ts.empty:
                st.warning("No data returned.")
            else:
                st.dataframe(ts, use_container_width=True)
                download_csv(ts, "Download CSV (trending searches)", "trending_searches.csv")
                if show_raw:
                    with st.expander("Raw", expanded=False):
                        st.write(ts)
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------
# TAB 4 — SUGGESTIONS
# -------------------------
with tabs[3]:
    st.subheader("Autocomplete suggestions")
    run_sug = st.button("Fetch suggestions")
    if run_sug:
        if not keywords:
            st.error("Add at least one keyword.")
        else:
            rows = []
            for kw in keywords:
                try:
                    py = TrendReq(hl=hl, tz=0)
                    sug = py.suggestions(keyword=kw)
                    for s in sug:
                        rows.append({
                            "seed": kw,
                            "title": s.get("title"),
                            "type": s.get("type"),
                            "mid": s.get("mid")
                        })
                except Exception as e:
                    rows.append({"seed": kw, "title": None, "type": None, "mid": None, "error": str(e)})
                time.sleep(per_request_delay)

            df = pd.DataFrame(rows)
            if df.empty:
                st.warning("No suggestions returned.")
            else:
                st.dataframe(df, use_container_width=True)
                download_csv(df, "Download CSV (suggestions)", "suggestions.csv")
                if show_raw:
                    with st.expander("Raw", expanded=False):
                        st.write(df)

# -------------------------
# TAB 5 — DIFFICULTY PROXY
# -------------------------
with tabs[4]:
    st.subheader("Difficulty proxy (0–100)")
    st.caption("Higher = more competitive/‘hotter’. Combines base interest, trend slope, and Rising related queries intensity.")

    run_diff = st.button("Compute difficulty proxy")
    if run_diff:
        if not keywords:
            st.error("Add at least one keyword.")
        else:
            # 1) Interest-over-time per term
            iot_all = {}
            progress = st.progress(0)
            status = st.empty()
            total = len(keywords)

            for i, term in enumerate(keywords, start=1):
                status.write(f"Fetching time series for **{term}** ({i}/{total}) …")
                try:
                    s = fetch_with_retry(term, timeframe, geo, hl,
                                         max_retries=max_retries, backoff_base=backoff_base)
                    if not s.empty:
                        iot_all[term] = s[term]
                except Exception:
                    pass
                progress.progress(int(i * 100 / total))
                time.sleep(per_request_delay)
            progress.empty(); status.empty()

            # 2) Related queries once for all
            rq = {}
            try:
                rq = cached_related_queries(tuple(keywords), timeframe, geo, hl) or {}
            except Exception:
                rq = {}

            # 3) Build difficulty table
            rows = []
            for term in keywords:
                series = iot_all.get(term, pd.Series(dtype=float))
                base = float(np.nanmean(series.values)) if series.size else 0.0
                slope_sc = slope_score_from_series(series)
                rising = rising_intensity_from_rq(rq.get(term))
                score = difficulty_proxy(base, slope_sc, rising)
                rows.append({
                    "keyword": term,
                    "base_interest_mean": round(base, 2),
                    "slope_score": round(slope_sc, 2),
                    "rising_intensity": round(rising, 2),
                    "difficulty_score": round(score, 1),
                    "band": band_from_score(score),
                })

            diff_df = pd.DataFrame(rows).sort_values("difficulty_score", ascending=False)
            st.dataframe(diff_df, use_container_width=True)
            download_csv(diff_df, "Download CSV (difficulty proxy)", "difficulty_proxy.csv")

            if show_raw:
                with st.expander("Raw inputs", expanded=False):
                    st.write("Interest series:", iot_all)
                    st.write("Related queries:", rq)
