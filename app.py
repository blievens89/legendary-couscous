import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps

# RestClient class
class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
            ).decode("ascii")
            headers = {'Authorization': 'Basic %s' % base64_bytes, 'Content-Encoding': 'gzip'}
            
            if data:
                data_str = dumps(list(data.values()))
            else:
                data_str = None

            connection.request(method, path, headers=headers, body=data_str)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        return self.request(path, 'POST', data)

# Streamlit configuration
st.set_page_config(page_title="Labs Keyword Ideas + Intent (Live)", layout="wide")
st.title("DataForSEO Labs — Keyword & Intent Planner")

if 'results' not in st.session_state:
    st.session_state.results = None

if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    password_input = st.text_input("Password", type="password")
    if st.button("Enter"):
        if password_input == st.secrets.get("APP_PASSWORD"):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("The password you entered is incorrect.")
    st.stop()

# API credentials
DATAFORSEO_LOGIN = st.secrets.get("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = st.secrets.get("DATAFORSEO_PASSWORD")

if not (DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD):
    st.error("DataForSEO API credentials are not found. Please set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in your Streamlit secrets.")
    st.stop()

try:
    client = RestClient(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD)
    BASE_URL = "/v3"
except Exception as e:
    st.error(f"Failed to initialise the API client: {e}")
    st.stop()

def make_api_post_request(endpoint: str, payload: dict) -> dict:
    try:
        response = client.post(f"{BASE_URL}{endpoint}", payload)
        if response and response.get("status_code") == 20000:
            return response
        else:
            st.error(f"API Error on {endpoint}: {response.get('status_message', 'Unknown error')}")
            if st.session_state.get('show_raw_data', False):
                st.json(response)
            return {}
    except Exception as e:
        st.error(f"An exception occurred while calling the API endpoint {endpoint}: {e}")
        return {}

def extract_items_from_response(response: dict) -> list:
    try:
        if not response or response.get("tasks_error", 1) > 0:
            st.warning("The API task returned an error. See raw response for details.")
            return []
        
        tasks = response.get("tasks", [])
        if not tasks:
            return []
            
        result = tasks[0].get("result")
        if not result:
            return []
            
        return result
    except (KeyError, IndexError, TypeError) as e:
        st.error(f"Error extracting items from response: {e}")
        return []

def safe_average(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    return numeric_series.mean() if not numeric_series.empty else 0.0

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ("Generate from Seed Keyword", "Analyse My Keyword List", "Scan Website for Keywords"),
        key="analysis_mode"
    )

    language_code = st.text_input("Language Code (e.g., en, fr, de)", value="en")
    location_name = st.text_input("Location Name", value="United Kingdom")
    
    if analysis_mode == "Generate from Seed Keyword":
        seed_keyword = st.text_input("Seed Keyword", value="remortgage")
        limit = st.slider("Max Keyword Ideas", 10, 300, 50, step=10)
        uploaded_keywords = None
        target_url = None
    elif analysis_mode == "Analyse My Keyword List":
        st.subheader("Your Keywords")
        pasted_keywords = st.text_area("Paste keywords here (one per line)")
        uploaded_file = st.file_uploader("Or upload a TXT/CSV file", type=['txt', 'csv'])
        
        uploaded_keywords = []
        if pasted_keywords:
            lines = pasted_keywords.splitlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            uploaded_keywords.extend(cleaned_lines)
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file, header=None)
                lines = df_upload[0].dropna().astype(str).tolist()
                uploaded_keywords.extend(lines)
            except Exception as e:
                st.error(f"Error reading file: {e}")

        uploaded_keywords = list(dict.fromkeys(filter(None, uploaded_keywords)))
        seed_keyword = None
        target_url = None
    else: # Scan Website
        target_url = st.text_input("Enter URL to scan", value="https://www.gov.uk/remortgaging-your-home")
        limit = st.slider("Max Keyword Ideas", 10, 300, 50, step=10)
        uploaded_keywords = None
        seed_keyword = None

    usd_to_gbp_rate = st.number_input("USD to GBP Exchange Rate", 0.1, 2.0, 0.79, 0.01)

    st.divider()
    st.caption("CTR/CVR Assumptions by Intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational": 0.03, "navigational": 0.03, "commercial": 0.04, "transactional": 0.04}
    cvr_defaults = {"informational": 0.015, "navigational": 0.015, "commercial": 0.03, "transactional": 0.03}
    ctrs, cvrs = {}, {}
    for intent in intents:
        col1, col2 = st.columns(2)
        with col1:
            ctrs[intent] = st.number_input(f"{intent.title()} CTR", 0.0, 1.0, ctr_defaults[intent], 0.005, format="%.3f", key=f"ctr_{intent}")
        with col2:
            cvrs[intent] = st.number_input(f"{intent.title()} CVR", 0.0, 1.0, cvr_defaults[intent], 0.005, format="%.3f", key=f"cvr_{intent}")

    st.divider()
    st.header("Budgeting")
    use_custom_budget = st.toggle("Set Custom Budget", value=False)
    if use_custom_budget:
        total_budget = st.number_input("Total Monthly Budget (£)", min_value=0.0, value=1000.0, step=100.0)
        st.caption("Allocate budget by intent (%)")
        
        budget_allocations = {}
        for intent in intents:
            budget_allocations[intent] = st.number_input(f"{intent.title()} %", min_value=0, max_value=100, value=25, key=f"budget_{intent}")

    show_raw_data = st.toggle("Show Raw API Data (for debugging)", value=False)
    st.session_state.show_raw_data = show_raw_data

@st.cache_data(ttl=3600, show_spinner="Fetching keyword suggestions...")
def get_keyword_suggestions(seed: str, lang_code: str, loc_name: str, limit: int) -> pd.DataFrame:
    payload_item = {
        "keyword": seed.strip(),
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
        "limit": limit,
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/keyword_suggestions/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)
    
    if not items:
        st.warning("No data returned from keyword suggestions API")
        return pd.DataFrame()

    rows = []
    try:
        # Debug: Check what we actually got
        if st.session_state.get('show_raw_data', False):
            st.write("Items structure:", type(items))
            st.write("Items content:", items)
        
        # Handle different possible response structures
        items_data = None
        
        if isinstance(items, list):
            if len(items) > 0:
                first_item = items[0]
                if isinstance(first_item, dict):
                    if 'items' in first_item:
                        items_data = first_item['items']
                    elif 'keyword' in first_item:  # Direct keyword data
                        items_data = items
                    else:
                        # Look for other possible data containers
                        for key in ['data', 'results', 'keywords']:
                            if key in first_item:
                                items_data = first_item[key]
                                break
                else:
                    items_data = items
            else:
                items_data = []
        elif isinstance(items, dict):
            # If items is a dict, look for keyword data
            if 'items' in items:
                items_data = items['items']
            elif 'data' in items:
                items_data = items['data']
            elif 'results' in items:
                items_data = items['results']
            else:
                # Treat the dict as a single item
                items_data = [items]
        else:
            st.error(f"Unexpected items type: {type(items)}")
            return pd.DataFrame()

        if items_data is None or not items_data:
            st.warning("No keyword data found in API response")
            return pd.DataFrame()

        # Ensure items_data is iterable
        if not hasattr(items_data, '__iter__'):
            st.error(f"Items data is not iterable: {type(items_data)}")
            return pd.DataFrame()

        for item in items_data:
            if isinstance(item, dict):
                info = item.get("keyword_info", item)  # Some APIs return data directly
                cpc = info.get("cpc", 0)
                
                # Handle different CPC formats
                if isinstance(cpc, dict):
                    cpc_value = cpc.get("cpc", 0)
                else:
                    cpc_value = cpc
                    
                rows.append({
                    "keyword": item.get("keyword", ""),
                    "search_volume": info.get("search_volume", 0),
                    "cpc_usd": cpc_value,
                    "competition": info.get("competition", 0),
                })
    
    except Exception as e:
        st.error(f"Error processing keyword suggestions response: {e}")
        if st.session_state.get('show_raw_data', False):
            st.write("Raw items:", items)
        return pd.DataFrame()
    
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Scanning site for keywords...")
def get_keywords_from_site(url: str, lang_code: str, loc_name: str, limit: int) -> pd.DataFrame:
    payload_item = {
        "target": url.strip(),
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
        "limit": limit,
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/keywords_for_site/live", post_data)
    items = extract_items_from_response(response)
    
    if not items or 'items' not in items[0]:
        return pd.DataFrame()

    rows = []
    for item in items[0]['items']:
        info = item.get("keyword_info", {})  # Changed from keyword_data to keyword_info
        rows.append({
            "keyword": item.get("keyword"),
            "search_volume": info.get("search_volume"),
            "cpc_usd": info.get("cpc"),
            "competition": info.get("competition"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Fetching metrics for your keywords...")
def get_keyword_metrics(keywords: list, lang_code: str, loc_name: str) -> pd.DataFrame:
    payload_item = {
        "keywords": keywords,
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/keywords_data/google_ads/search_volume/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)

    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append({
                "keyword": item.get("keyword", ""),
                "search_volume": item.get("search_volume", 0),
                "cpc_usd": item.get("cpc", 0),
                "competition": item.get("competition", 0),
            })
    
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Analysing search intent...")
def get_search_intent(keywords: list, lang_code: str) -> pd.DataFrame:
    payload_item = {
        "keywords": keywords,
        "language_code": lang_code.strip()
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/search_intent/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)

    if not items:
        return pd.DataFrame()

    rows = []
    # Handle different possible response structures
    items_data = items
    if isinstance(items, list) and len(items) > 0:
        if 'items' in items[0]:
            items_data = items[0]['items']
        else:
            items_data = items

    for item in items_data:
        if isinstance(item, dict):
            intent_info = item.get("keyword_intent", {})
            rows.append({
                "keyword_clean": (item.get("keyword") or "").lower().strip(),
                "intent": intent_info.get("label", "unknown"),
                "intent_probability": intent_info.get("probability", 0)
            })
    
    return pd.DataFrame(rows)

# Main analysis button
if st.button("Analyse Keywords", type="primary"):
    df_metrics = pd.DataFrame()

    if analysis_mode == "Generate from Seed Keyword" and seed_keyword:
        df_metrics = get_keyword_suggestions(seed_keyword, language_code, location_name, limit)
    elif analysis_mode == "Analyse My Keyword List" and uploaded_keywords:
        keyword_chunks = [uploaded_keywords[i:i + 1000] for i in range(0, len(uploaded_keywords), 1000)]
        results_list = []
        for chunk in keyword_chunks:
            chunk_result = get_keyword_metrics(chunk, language_code, location_name)
            if not chunk_result.empty:
                results_list.append(chunk_result)
        
        if results_list:
            df_metrics = pd.concat(results_list, ignore_index=True)
    elif analysis_mode == "Scan Website for Keywords" and target_url:
        df_metrics = get_keywords_from_site(target_url, language_code, location_name, limit)

    if df_metrics.empty:
        st.warning("Could not retrieve keyword metrics. Please check your inputs or try different keywords.")
        st.session_state.results = None
    else:
        # Clean and prepare data
        df_metrics = df_metrics.dropna(subset=['keyword'])
        df_metrics = df_metrics[df_metrics['keyword'].str.strip() != '']
        
        # Ensure numeric columns are properly converted
        df_metrics['search_volume'] = pd.to_numeric(df_metrics['search_volume'], errors='coerce').fillna(0)
        df_metrics['cpc_usd'] = pd.to_numeric(df_metrics['cpc_usd'], errors='coerce').fillna(0)
        df_metrics['competition'] = pd.to_numeric(df_metrics['competition'], errors='coerce').fillna(0)
        
        df_metrics['keyword_clean'] = df_metrics['keyword'].str.lower().str.strip()
        
        intent_keywords = df_metrics['keyword_clean'].tolist()
        df_intent = get_search_intent(intent_keywords, language_code)

        if not df_intent.empty:
            df_merged = pd.merge(df_metrics, df_intent, on="keyword_clean", how="left")
            df_merged = df_merged.drop(columns=['keyword_clean'])
        else:
            df_merged = df_metrics.drop(columns=['keyword_clean'])
            df_merged['intent'] = 'unknown'
            df_merged['intent_probability'] = 0

        # Fill missing intents
        df_merged['intent'] = df_merged['intent'].fillna('unknown')
        
        st.session_state.results = {"df_merged": df_merged}

# Display results
if st.session_state.results:
    df_merged = st.session_state.results["df_merged"]

    # Convert CPC to GBP
    df_merged["cpc_gbp"] = (df_merged["cpc_usd"] * usd_to_gbp_rate).round(2)
    
    st.subheader("Keyword Analysis Results")
    display_cols = ['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent']
    st.dataframe(df_merged[display_cols], use_container_width=True)

    # Debug unmatched keywords
    unmatched_keywords = df_merged[df_merged['intent'].isna() | (df_merged['intent'] == 'unknown')]['keyword'].tolist()
    if unmatched_keywords:
        with st.expander(f"Debug: {len(unmatched_keywords)} keywords could not be assigned an intent"):
            st.write(unmatched_keywords)

    # Create summary
    summary_df = df_merged[df_merged['intent'] != 'unknown'].dropna(subset=['intent'])

    if not summary_df.empty:
        summary = summary_df.groupby("intent").agg(
            keywords=("keyword", "count"),
            total_volume=("search_volume", "sum"),
            avg_cpc_gbp=("cpc_gbp", safe_average),
        ).reset_index().rename(columns={"intent": "Intent"})

        # Ensure all numeric columns are properly typed
        summary["total_volume"] = pd.to_numeric(summary["total_volume"], errors='coerce').fillna(0)
        summary["avg_cpc_gbp"] = pd.to_numeric(summary["avg_cpc_gbp"], errors='coerce').fillna(0)

        # Add CTR and CVR
        summary["CTR"] = summary["Intent"].map(ctrs).fillna(0.03)
        summary["CVR"] = summary["Intent"].map(cvrs).fillna(0.015)
        
        # Calculate metrics
        summary["Max Clicks"] = (summary["total_volume"] * summary["CTR"]).round(0)
        summary["Max Spend £"] = (summary["Max Clicks"] * summary["avg_cpc_gbp"]).round(2)
        
        required_budget = summary["Max Spend £"].sum()
        if not use_custom_budget:
            st.sidebar.metric("Required Budget", f"£{required_budget:,.2f}")

        if use_custom_budget:
            summary["Budget £"] = summary["Intent"].map(budget_allocations).fillna(25) * total_budget / 100
            summary["Clicks"] = np.where(
                summary["Budget £"] < summary["Max Spend £"],
                (summary["Budget £"] / summary["avg_cpc_gbp"]).round(0),
                summary["Max Clicks"]
            )
            summary["Spend £"] = (summary["Clicks"] * summary["avg_cpc_gbp"]).round(2)
        else:
            summary["Clicks"] = summary["Max Clicks"]
            summary["Spend £"] = summary["Max Spend £"]

        summary["Conversions"] = (summary["Clicks"] * summary["CVR"]).round(0)
        summary["CPA £"] = np.where(
            summary["Conversions"] > 0,
            (summary["Spend £"] / summary["Conversions"]).round(2),
            0
        )

        st.subheader("Grouped by Search Intent")
        display_summary = summary[['Intent', 'keywords', 'total_volume', 'Clicks', 'Spend £', 'Conversions', 'CPA £']]
        st.dataframe(display_summary.fillna("—"), use_container_width=True)

        # Blended overview
        total_keywords = summary["keywords"].sum()
        total_volume = summary["total_volume"].sum()
        total_clicks = summary["Clicks"].sum()
        total_spend = summary["Spend £"].sum()
        total_conversions = summary["Conversions"].sum()

        blended_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        blended_ctr = total_clicks / total_volume if total_volume > 0 else 0
        blended_cvr = total_conversions / total_clicks if total_clicks > 0 else 0
        blended_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        blended_overview = pd.DataFrame({
            "Total Keywords": [int(total_keywords)],
            "Total Volume": [int(total_volume)],
            "Weighted Avg CPC £": [round(blended_cpc, 2)],
            "Weighted CTR": [round(blended_ctr, 3)],
            "Total Clicks": [int(total_clicks)],
            "Weighted CVR": [round(blended_cvr, 3)],
            "Total Conversions": [int(total_conversions)],
            "Total Spend £": [round(total_spend, 2)],
            "Blended CPA £": [round(blended_cpa, 2)]
        })
        st.subheader("Blended Overview (Weighted)")
        st.dataframe(blended_overview, use_container_width=True)

        # Visualisation
        st.subheader("Performance by Intent")
        
        metric_mapping_clean = {
            "Total Volume": "total_volume",
            "Clicks": "Clicks",
            "Spend (£)": "Spend_GBP",
            "Conversions": "Conversions",
            "CPA (£)": "CPA_GBP"
        }

        chart_metric_display = st.selectbox(
            "Choose a metric to visualise",
            list(metric_mapping_clean.keys())
        )
        
        chart_df = summary.copy()
        chart_df.columns = [col.replace('£', 'GBP').replace(' ', '_') for col in chart_df.columns]
        chart_metric_col = metric_mapping_clean[chart_metric_display]

        if chart_metric_col in chart_df.columns:
            chart = alt.Chart(chart_df).mark_bar(
                color="#48d597"
            ).encode(
                x=alt.X('Intent:N', sort='-y', title='Search Intent'),
                y=alt.Y(f'{chart_metric_col}:Q', title=chart_metric_display),
                tooltip=['Intent', alt.Tooltip(f'{chart_metric_col}:Q', title=chart_metric_display, format=',.0f')]
            ).properties(
                title=f'{chart_metric_display} by Search Intent'
            ).configure_axis(
                labelAngle=0
            ).configure_title(
                fontSize=16
            )
            st.altair_chart(chart, use_container_width=True, theme=None)

        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_data = df_merged.to_csv(index=False).encode("utf-8")
            st.download_button("Download Detailed Data (CSV)", csv_data, "keyword_intent_details.csv", "text/csv", key="d1")
        
        with col2:
            summary_csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button("Download Intent Summary (CSV)", summary_csv, "intent_summary.csv", "text/csv", key="d2")
        
        with col3:
            overview_csv = blended_overview.to_csv(index=False).encode("utf-8")
            st.download_button("Download Blended Overview (CSV)", overview_csv, "blended_overview.csv", "text/csv", key="d3")

    else:
        st.warning("Could not generate intent summary as no intent data was returned.")

    # Cost estimation
    if not df_merged.empty:
        num_keywords = len(df_merged)
        cost_sug = 0.01 + num_keywords * 0.0001
        cost_int = 0.001 + num_keywords * 0.0001
        approx_cost = cost_sug + cost_int
        st.caption(f"Approximate API cost for this run: ${approx_cost:.4f} for {num_keywords} keywords (estimate only).")
