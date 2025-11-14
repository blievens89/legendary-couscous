# app.py — Single-page Trends dashboard with Category selector + Difficulty proxy
# Features:
# - Single scrolling page (no tabs)
# - Category selector (built-in list OR load full list via JSON URL) + manual ID override
# - Input mode: Keywords or Topic MIDs
# - Interest over time, Related queries, Trending searches, Suggestions, Difficulty proxy
# - Progress bars, caching, retry/backoff, per-keyword fetching, CSV export

import time
import json
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pytrends.request import TrendReq

# -------------------------
# PAGE / THEME
# -------------------------
st.set_page_config(page_title="Google Trends Control Panel", layout="wide")
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

# A compact built-in starter set of categories (ID: Name).
# You can load the full list via URL (see sidebar).
BUILTIN_CATEGORIES = {
    0: "All categories",
    3: "Arts & Entertainment",
    5: "Autos & Vehicles",
    7: "Beauty & Fitness",
    12: "Business & Industrial",
    13: "Computers & Electronics (Internet & Telecom in some docs)",
    19: "Finance",
    184: "Food & Drink",
    20: "Games",
    533: "Health",
    44: "Hobbies & Leisure",
    1227: "Home & Garden",
    65: "Jobs & Education",
    958: "Law & Government",
    70: "News",
    299: "Online Communities",
    974: "People & Society",
    968: "Pets & Animals",
    882: "Real Estate",
    66: "Reference",
    452: "Science",
    174: "Shopping",
    673: "Sports",
    258: "Travel",
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

def build_category_options(cat_map: dict):
    # Return list of "Name (ID)" strings, sorted by name, with ID 0 on top.
    items = [(cid, name) for cid, name in cat_map.items()]
    items_sorted = sorted(items, key=lambda t: (t[0] != 0, t[1].lower()))
    return [f"{name} ({cid})" for cid, name in items_sorted]

def parse_category_choice(choice_str: str) -> int:
    if not choice_str:
        return 0
    if "(" in choice_str and choice_str.endswith(")"):
        try:
            return int(choice_str.rsplit("(", 1)[1].rstrip(")"))
        except:
            return 0
    try:
        return int(choice_str)
    except:
        return 0

# -------------------------
# SIDEBAR — GLOBAL CONTROLS
# -------------------------
with st.sidebar:
    st.header("Settings")

    # Input mode: plain keywords or topic MIDs
    input_mode = st.radio("Input type", ["Keywords", "Topic MIDs"], horizontal=True)

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
    st.caption("Category")
    cat_source = st.radio("Source", ["Built-in", "Load from URL"], horizontal=True)
    if cat_source == "Load from URL":
        url = st.text_input("Full categories JSON URL",
                            help="Must return a JSON array of {id: number, name: string}.")
        loaded = {}
        if url:
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                for row in data:
                    if isinstance(row, dict) and "id" in row and "name" in row:
                        loaded[int(row["id"])] = str(row["name"])
                if loaded:
                    st.success(f"Loaded {len(loaded)} categories.")
                else:
                    st.warning("No valid categories found in JSON.")
            except Exception as e:
                st.error(f"Failed to load categories: {e}")
                loaded = {}
        category_map = loaded if loaded else BUILTIN_CATEGORIES
    else:
        category_map = BUILTIN_CATEGORIES

    cat_choice = st.selectbox("Category (type to search)", build_category_options(category_map), index=0)
    cat_id_manual = st.text_input("Or enter Category ID (overrides selection)", value="")
    cat_id = parse_category_choice(cat_id_manual) if cat_id_manual.strip() else parse_category_choice(cat_choice)

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
def cached_interest_for_term(term, timeframe, geo, hl, cat):
    """Fetch interest_over_time for a single term (keyword or MID) with category filter (cached)."""
    py = TrendReq(hl=hl, tz=0)
    py.build_payload(kw_list=[term], timeframe=timeframe, geo=geo, cat=cat)
    df = py.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.columns = [term]  # rename single column for join
    return df

def fetch_with_retry(term, timeframe, geo, hl, cat, max_retries, backoff_base):
    tries = 0
    while True:
        try:
            return cached_interest_for_term(term, timeframe, geo, hl, cat)
        except Exception as e:
            tries += 1
            if tries > max_retries:
                raise
            time.sleep(backoff_base * tries)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_related_queries(keywords, timeframe, geo, hl, cat):
    """Fetch related queries dict for a list (cached as a whole)."""
    py = TrendReq(hl=hl, tz=0)
    py.build_payload(kw_list=list(keywords), timeframe=timeframe, geo=geo, cat=cat)
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
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    if len(x) < 3 or np.all(y == y[0]):
        return 0.0
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    scale = 15.0  # adjust sensitivity; higher = less sensitive
    score = 50.0 + 50.0 * np.tanh(slope * scale)
    return float(np.clip(score, 0, 100))

def rising_intensity_from_rq(rq_entry) -> float:
    """
    Given related_queries()[kw] entry, compute mean 'value' of Rising queries.
    Returns 0–100 (already relative scale).
    """
    if not rq_entry:
        return 0.0
    rising_df = rq_entry.get("rising")
    if rising_df is None or rising_df.empty or "value" not in rising_df.columns:
        return 0.0
    vals = rising_df["value"].astype(float).values[:10]
    return float(np.clip(np.mean(vals), 0, 100))

def difficulty_proxy(base_interest: float, slope_score: float, rising_intensity: float) -> float:
    w_interest = 0.40
    w_rising   = 0.40
    w_slope    = 0.20
    score = w_interest * base_interest + w_rising * rising_intensity + w_slope * slope_score
    return float(np.clip(score, 0, 100))

# -------------------------
# INPUTS + SECTION SELECT
# -------------------------
st.markdown("**Enter terms** (comma‑separated).")
placeholder = "/m/02k1b, /m/0k8z (for Topics)" if (st.session_state.get("input_mode") == "Topic MIDs" or input_mode == "Topic MIDs") \
              else "web design, website builder"
terms_raw = st.text_input("Search terms", value=placeholder if input_mode=="Topic MIDs" else "web design, website builder")
terms = [t.strip() for t in terms_raw.split(",") if t.strip()]
terms = list(dict.fromkeys(terms))  # de‑dupe, keep order

st.divider()
st.subheader("What to run")
c1, c2, c3, c4, c5 = st.columns(5)
do_iot = c1.checkbox("Interest over time", value=True)
do_rq  = c2.checkbox("Related queries", value=True)
do_ts  = c3.checkbox("Trending searches", value=False)
do_sug = c4.checkbox("Suggestions", value=False)
do_diff= c5.checkbox("Difficulty proxy", value=True)

run_all = st.button("Run selected sections")

# -------------------------
# SECTION: INTEREST OVER TIME
# -------------------------
def section_interest_over_time(terms, timeframe, geo, hl, cat_id, gran_rule):
    st.markdown("### Interest over time")
    if not terms:
        st.info("Add at least one term.")
        return None

    progress = st.progress(0)
    status = st.empty()

    combined = None
    errors = {}
    total = len(terms)

    for i, term in enumerate(terms, start=1):
        status.write(f"Fetching **{term}** ({i}/{total}) …")
        try:
            df = fetch_with_retry(term, timeframe, geo, hl, cat_id,
                                  max_retries=max_retries, backoff_base=backoff_base)
            combined = df if combined is None else combined.join(df, how="outer")
        except Exception as e:
            errors[term] = str(e)
        progress.progress(int(i * 100 / total))
        time.sleep(per_request_delay)

    progress.empty(); status.empty()

    if combined is None or combined.empty:
        st.warning("No data returned.")
        return None

    combined = combined.sort_index()
    combined_resampled = resample_df(combined, gran_rule)
    st.line_chart(combined_resampled)
    st.dataframe(combined_resampled.reset_index(), use_container_width=True)
    download_csv(combined_resampled, "Download CSV (interest over time)", "interest_over_time.csv")

    if show_raw:
        with st.expander("Raw (combined)", expanded=False):
            st.write(combined)

    if errors:
        st.warning("Some terms failed.")
        st.json(errors)

    return combined  # return for difficulty proxy if needed

# -------------------------
# SECTION: RELATED QUERIES
# -------------------------
def section_related_queries(terms, timeframe, geo, hl, cat_id):
    st.markdown("### Related queries (Top & Rising)")
    if not terms:
        st.info("Add at least one term.")
        return {}

    try:
        rq = cached_related_queries(tuple(terms), timeframe, geo, hl, cat_id)
        if not rq:
            st.warning("No related queries returned.")
            return {}
        for kw in terms:
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
        return rq
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

# -------------------------
# SECTION: TRENDING SEARCHES
# -------------------------
def section_trending_searches():
    st.markdown("### Trending searches (daily)")
    pn_col1, _ = st.columns([1,3])
    pn_name = pn_col1.selectbox("Country", list(PN_OPTIONS.keys()), index=0, key="pn")
    if st.button("Fetch trending searches"):
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
# SECTION: SUGGESTIONS
# -------------------------
def section_suggestions(terms):
    st.markdown("### Autocomplete suggestions")
    if not terms:
        st.info("Add at least one term.")
        return
    rows = []
    for kw in terms:
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
# SECTION: DIFFICULTY PROXY
# -------------------------
def section_difficulty_proxy(terms, timeframe, geo, hl, cat_id, iot_df=None, rq_dict=None):
    st.markdown("### Difficulty proxy (0–100)")
    st.caption("Higher = more competitive/‘hotter’. Combines base interest, trend slope, and Rising related queries intensity.")

    # Use what we already have if provided (saves requests)
    iot_all = {}
    if iot_df is not None and not iot_df.empty:
        for col in iot_df.columns:
            iot_all[col] = iot_df[col]
    else:
        progress = st.progress(0)
        status = st.empty()
        total = len(terms)
        for i, term in enumerate(terms, start=1):
            status.write(f"Fetching time series for **{term}** ({i}/{total}) …")
            try:
                df = fetch_with_retry(term, timeframe, geo, hl, cat_id,
                                      max_retries=max_retries, backoff_base=backoff_base)
                if not df.empty:
                    iot_all[term] = df[term]
            except Exception:
                pass
            progress.progress(int(i * 100 / total))
            time.sleep(per_request_delay)
        progress.empty(); status.empty()

    if rq_dict is None:
        try:
            rq_dict = cached_related_queries(tuple(terms), timeframe, geo, hl, cat_id) or {}
        except Exception:
            rq_dict = {}

    rows = []
    for term in terms:
        series = iot_all.get(term, pd.Series(dtype=float))
        base = float(np.nanmean(series.values)) if series.size else 0.0
        slope_sc = slope_score_from_series(series)
        rising = rising_intensity_from_rq(rq_dict.get(term))
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
            st.write("Related queries:", rq_dict)

# -------------------------
# RUN SELECTED SECTIONS
# -------------------------
iot_df = None
rq_dict = None

if run_all:
    if do_iot:
        iot_df = section_interest_over_time(terms, timeframe, geo, hl, cat_id, gran_rule)
    if do_rq:
        rq_dict = section_related_queries(terms, timeframe, geo, hl, cat_id)
    if do_ts:
        section_trending_searches()
    if do_sug:
        section_suggestions(terms)
    if do_diff:
        section_difficulty_proxy(terms, timeframe, geo, hl, cat_id, iot_df=iot_df, rq_dict=rq_dict)

# Also allow running sections individually below (without pressing Run selected)
st.divider()
st.markdown("#### Run sections individually (optional)")
if st.button("Run: Interest over time"):
    section_interest_over_time(terms, timeframe, geo, hl, cat_id, gran_rule)
if st.button("Run: Related queries"):
    section_related_queries(terms, timeframe, geo, hl, cat_id)
if st.button("Run: Trending searches"):
    section_trending_searches()
if st.button("Run: Suggestions"):
    section_suggestions(terms)
if st.button("Run: Difficulty proxy"):
    section_difficulty_proxy(terms, timeframe, geo, hl, cat_id)
