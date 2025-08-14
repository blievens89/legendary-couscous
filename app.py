# app.py
# A local Google Trends UI using Streamlit + pytrends
# Features: Interest over time, Related Queries, Trending Searches, Suggestions
# Controls: timeframe, resampling (daily/weekly/monthly), locale/language

import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
from pytrends.request import TrendReq
import streamlit as st

# ---------- Helpers ----------
COUNTRY_OPTIONS = {
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

# pytrends uses special codes for trending_searches 'pn' argument
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
    "Worldwide (Top Charts only)": "united_states",  # fallback; pn cannot be empty
}

TIMEFRAME_PRESETS = [
    "today 3-m",
    "today 12-m",
    "today 5-y",
    "all",
    "Custom date range…",
]

LANG_OPTIONS = {
    "English (UK)": "en-GB",
    "English (US)": "en-US",
    "Dutch": "nl-NL",
    "German": "de-DE",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "Italian": "it-IT",
}

RESAMPLE_OPTIONS = {
    "No resample (native)": None,
    "Weekly average": "W",
    "Monthly average": "MS",  # month start
}

def to_timeframe(preset, start_date=None, end_date=None):
    if preset != "Custom date range…":
        return preset
    # Format: 'YYYY-MM-DD YYYY-MM-DD'
    return f"{start_date:%Y-%m-%d} {end_date:%Y-%m-%d}"

def resample_df(df, rule):
    if rule is None or df.empty:
        return df
    # pytrends index is a DatetimeIndex; use mean for relative interest
    return df.resample(rule).mean().round(2)

def download_button(df, label, filename):
    if not df.empty:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ---------- UI ----------
st.set_page_config(page_title="Trends Control Panel", layout="wide")
st.title("Google Trends Control Panel (Local)")

with st.sidebar:
    st.header("Global settings")
    lang = st.selectbox("Language", list(LANG_OPTIONS.keys()), index=0)
    hl = LANG_OPTIONS[lang]

    country = st.selectbox("Region (geo)", list(COUNTRY_OPTIONS.keys()), index=1)
    geo = COUNTRY_OPTIONS[country]
    custom_geo = st.text_input("Override geo (optional, e.g. GB, US, NL)", value="")
    if custom_geo.strip():
        geo = custom_geo.strip()

    tf_preset = st.selectbox("Timeframe", TIMEFRAME_PRESETS, index=1)
    if tf_preset == "Custom date range…":
        default_end = dt.date.today()
        default_start = default_end - relativedelta(years=1)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=default_end)
        timeframe = to_timeframe(tf_preset, start, end)
    else:
        timeframe = tf_preset

    resample = st.selectbox("Granularity", list(RESAMPLE_OPTIONS.keys()), index=0)
    resample_rule = RESAMPLE_OPTIONS[resample]

    st.caption("Tip: ‘geo’ limits data to a country; leave blank for worldwide. ‘hl’ affects UI language, not data.")

st.markdown("Enter your search terms (comma-separated).")
terms_input = st.text_input("Search terms", value="web design, website builder")
keywords = [t.strip() for t in terms_input.split(",") if t.strip()]

tabs = st.tabs(["Interest over time", "Related queries", "Trending searches", "Suggestions"])

# Create one shared TrendReq configured once
pytrends = TrendReq(hl=hl, tz=0)

# ---------- Tab 1: Interest over time ----------
with tabs[0]:
    st.subheader("Interest over time")
    colA, colB = st.columns([1,1])
    with colA:
        run_iot = st.button("Fetch interest over time")
    with colB:
        st.write("")

    if run_iot:
        if not keywords:
            st.error("Add at least one search term.")
        else:
            try:
                pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()
                if df.empty:
                    st.warning("No data returned. Try a broader timeframe or different terms.")
                else:
                    if "isPartial" in df.columns:
                        df = df.drop(columns=["isPartial"])
                    df_resampled = resample_df(df, resample_rule)
                    st.line_chart(df_resampled)
                    st.dataframe(df_resampled.reset_index(), use_container_width=True)
                    download_button(df_resampled, "Download CSV (interest over time)", "interest_over_time.csv")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- Tab 2: Related queries ----------
with tabs[1]:
    st.subheader("Related queries")
    st.caption("Shows ‘top’ and ‘rising’ related searches per keyword.")
    run_rq = st.button("Fetch related queries")
    if run_rq:
        if not keywords:
            st.error("Add at least one search term.")
        else:
            try:
                pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=geo)
                rq = pytrends.related_queries()
                if not rq:
                    st.warning("No related queries returned.")
                else:
                    for kw in keywords:
                        st.markdown(f"**{kw}**")
                        d = rq.get(kw, {})
                        top_df = d.get("top")
                        rising_df = d.get("rising")

                        if top_df is not None and not top_df.empty:
                            st.write("Top related queries")
                            st.dataframe(top_df, use_container_width=True)
                            download_button(top_df, f"Download CSV (top) - {kw}", f"related_top_{kw}.csv")
                        else:
                            st.write("No ‘top’ results.")

                        if rising_df is not None and not rising_df.empty:
                            st.write("Rising related queries")
                            st.dataframe(rising_df, use_container_width=True)
                            download_button(rising_df, f"Download CSV (rising) - {kw}", f"related_rising_{kw}.csv")
                        else:
                            st.write("No ‘rising’ results.")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- Tab 3: Trending searches ----------
with tabs[2]:
    st.subheader("Trending searches (daily)")
    pn_name = st.selectbox("Country for trending searches", list(PN_OPTIONS.keys()), index=0)
    run_trending = st.button("Fetch trending searches")
    if run_trending:
        try:
            pn = PN_OPTIONS[pn_name]
            ts_df = pytrends.trending_searches(pn=pn)
            if ts_df.empty:
                st.warning("No data returned for trending searches.")
            else:
                st.dataframe(ts_df, use_container_width=True)
                download_button(ts_df, "Download CSV (trending searches)", "trending_searches.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------- Tab 4: Suggestions ----------
with tabs[3]:
    st.subheader("Autocomplete suggestions")
    st.caption("Google’s type‑ahead suggestions for each term.")
    run_sug = st.button("Fetch suggestions")
    if run_sug:
        if not keywords:
            st.error("Add at least one search term.")
        else:
            try:
                rows = []
                for kw in keywords:
                    sug = pytrends.suggestions(keyword=kw)
                    for s in sug:
                        rows.append({"seed": kw, "title": s.get("title"), "type": s.get("type"), "mid": s.get("mid")})
                sug_df = pd.DataFrame(rows)
                if sug_df.empty:
                    st.warning("No suggestions returned.")
                else:
                    st.dataframe(sug_df, use_container_width=True)
                    download_button(sug_df, "Download CSV (suggestions)", "suggestions.csv")
            except Exception as e:
                st.error(f"Error: {e}")