import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pytrends.request import TrendReq
from datetime import datetime

st.set_page_config(
    page_title="Google Trends - 5 Year Overview",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Trends - 5 Year UK Overview")

# Initialize PyTrends in session state
if 'pytrends' not in st.session_state:
    st.session_state.pytrends = TrendReq(hl='en-GB', tz=0)

# Input
keyword = st.text_input("Enter search term", value="Mortgage")

if st.button("Get Trends Data", type="primary"):
    if keyword:
        try:
            with st.spinner(f"Fetching 5-year trend data for '{keyword}'..."):
                # Build payload for last 5 years, UK only
                st.session_state.pytrends.build_payload(
                    [keyword], 
                    cat=0, 
                    timeframe='today 5-y', 
                    geo='GB', 
                    gprop=''
                )
                
                # Get interest over time
                df = st.session_state.pytrends.interest_over_time()
                
                if df is not None and not df.empty:
                    # Remove isPartial column if it exists
                    if 'isPartial' in df.columns:
                        df = df.drop('isPartial', axis=1)
                    
                    # Convert index to datetime if not already
                    df.index = pd.to_datetime(df.index)
                    
                    # Calculate monthly average
                    df_monthly = df.resample('MS').mean().round(1)
                    
                    # Display chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_monthly.index,
                        y=df_monthly[keyword],
                        mode='lines+markers',
                        name=keyword,
                        line=dict(color='#48d597', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"Monthly Average Search Interest: {keyword} (UK, Last 5 Years)",
                        xaxis_title="Month",
                        yaxis_title="Search Interest (0-100)",
                        hovermode='x unified',
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("📊 Monthly Averages")
                    
                    # Format the dataframe for display
                    display_df = df_monthly.copy()
                    display_df.index = display_df.index.strftime('%B %Y')
                    display_df = display_df.rename(columns={keyword: 'Average Interest'})
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download button
                    csv = df_monthly.to_csv()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{keyword}_5yr_monthly_avg_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("No data returned from Google Trends. Try again in a few minutes.")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if '429' in str(e) or 'too many' in str(e).lower():
                st.warning("""
                **Rate limit hit.** Try:
                1. Wait 5-10 minutes
                2. Use a VPN
                3. Try from a different network
                """)
    else:
        st.warning("Please enter a search term")

# Footer
st.divider()
st.caption("Data from Google Trends | UK only | 5-year timeframe")
