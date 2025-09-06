import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pytrends.request import TrendReq
import time
from datetime import datetime, timedelta
import json

class GoogleTrendsAnalyzer:
    def __init__(self):
        """Initialize the Google Trends Analyzer"""
        self.pytrends = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'results_cache' not in st.session_state:
            st.session_state.results_cache = {}
        if 'last_search' not in st.session_state:
            st.session_state.last_search = None
            
    def create_connection(self, hl='en-US', tz=360, use_proxies=False, proxies_list=None):
        """Create a connection to Google Trends API"""
        try:
            if use_proxies and proxies_list:
                self.pytrends = TrendReq(
                    hl=hl, 
                    tz=tz,
                    timeout=(10, 25),
                    proxies=proxies_list,
                    retries=2,
                    backoff_factor=0.1
                )
            else:
                self.pytrends = TrendReq(
                    hl=hl, 
                    tz=tz,
                    timeout=(10, 25),
                    retries=2,
                    backoff_factor=0.1
                )
            return True
        except Exception as e:
            st.error(f"Failed to create connection: {str(e)}")
            return False
    
    def get_interest_over_time(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get interest over time data"""
        try:
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.interest_over_time()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching interest over time: {str(e)}")
            return None
    
    def get_interest_by_region(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop='', resolution='COUNTRY'):
        """Get interest by region data"""
        try:
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching interest by region: {str(e)}")
            return None
    
    def get_related_queries(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related queries"""
        try:
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.related_queries()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching related queries: {str(e)}")
            return None
    
    def get_related_topics(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related topics"""
        try:
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.related_topics()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching related topics: {str(e)}")
            return None
    
    def get_trending_searches(self, pn='united_states'):
        """Get trending searches"""
        try:
            return self.pytrends.trending_searches(pn=pn)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching trending searches: {str(e)}")
            return None
    
    def get_suggestions(self, keyword):
        """Get keyword suggestions"""
        try:
            return self.pytrends.suggestions(keyword)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching suggestions: {str(e)}")
            return None
    
    def handle_rate_limit_error(self, e=None):
        """Handle rate limit error with helpful message"""
        error_msg = f"Error details: {str(e)}" if e else ""
        st.error(f"""
        ⚠️ **Rate Limit Reached**
        
        Google Trends has rate limits. Try one of these solutions:
        1. Wait 60 seconds before trying again
        2. Use a VPN or different network
        3. Try again later (limits reset after some time)
        4. Reduce the number of keywords or locations
        
        {error_msg}
        """)
        
        # Show countdown timer
        placeholder = st.empty()
        for i in range(60, 0, -1):
            placeholder.info(f"⏱️ Waiting {i} seconds before you can retry...")
            time.sleep(1)
        placeholder.success("✅ You can try again now!")

def main():
    st.set_page_config(
        page_title="Google Trends Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Google Trends Analyzer")
    st.markdown("Analyze search trends, discover related queries, and explore regional interest patterns")
    
    # Initialize analyzer
    analyzer = GoogleTrendsAnalyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection settings
        st.subheader("Connection Settings")
        hl = st.selectbox("Language", ["en-US", "en-GB", "es", "fr", "de", "it", "pt", "ru", "ja", "zh-CN"], index=0)
        tz = st.number_input("Timezone Offset (minutes)", value=360, help="For US CST use 360")
        
        use_proxies = st.checkbox("Use Proxies (for rate limit issues)")
        proxies_list = None
        if use_proxies:
            proxy_input = st.text_area("Enter proxies (one per line)", 
                                      placeholder="https://34.203.233.13:80\nhttps://35.201.123.31:880")
            if proxy_input:
                proxies_list = proxy_input.strip().split('\n')
        
        # Create connection
        if analyzer.create_connection(hl, tz, use_proxies, proxies_list):
            st.success("✅ Connected to Google Trends")
        
        st.divider()
        
        # Search parameters
        st.subheader("Search Parameters")
        
        # Keywords input
        keywords_input = st.text_area(
            "Keywords (one per line, max 5)",
            value="Python\nJavaScript",
            help="Enter up to 5 keywords to compare"
        )
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()][:5]
        
        # Timeframe
        timeframe_option = st.selectbox(
            "Timeframe",
            ["Past hour", "Past 4 hours", "Past day", "Past 7 days", 
             "Past 30 days", "Past 90 days", "Past 12 months", 
             "Past 5 years", "All time", "Custom range"]
        )
        
        timeframe_map = {
            "Past hour": "now 1-H",
            "Past 4 hours": "now 4-H",
            "Past day": "now 1-d",
            "Past 7 days": "now 7-d",
            "Past 30 days": "today 1-m",
            "Past 90 days": "today 3-m",
            "Past 12 months": "today 12-m",
            "Past 5 years": "today 5-y",
            "All time": "all"
        }
        
        if timeframe_option == "Custom range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End date", datetime.now())
            timeframe = f"{start_date} {end_date}"
        else:
            timeframe = timeframe_map[timeframe_option]
        
        # Geographic location
        geo = st.text_input("Geographic Location", value="", 
                          placeholder="e.g., US, GB, US-CA",
                          help="Leave empty for worldwide")
        
        # Category
        cat = st.number_input("Category", value=0, min_value=0,
                            help="0 for all categories. See Google Trends for category codes")
        
        # Google property
        gprop = st.selectbox("Google Property", 
                           ["web", "images", "news", "youtube", "froogle"],
                           index=0,
                           help="Filter by Google property")
        if gprop == "web":
            gprop = ""
        
        st.divider()
        
        # Analysis options
        st.subheader("Analysis Options")
        show_interest_over_time = st.checkbox("Interest Over Time", value=True)
        show_interest_by_region = st.checkbox("Interest by Region", value=True)
        show_related_queries = st.checkbox("Related Queries", value=True)
        show_related_topics = st.checkbox("Related Topics", value=False)
        show_trending = st.checkbox("Trending Searches", value=False)
        show_suggestions = st.checkbox("Keyword Suggestions", value=False)
    
    # Main content area
    if st.button("🔍 Analyze Trends", type="primary"):
        if not keywords:
            st.error("Please enter at least one keyword")
            return
        
        # Store search parameters
        search_params = {
            'keywords': keywords,
            'timeframe': timeframe,
            'geo': geo,
            'cat': cat,
            'gprop': gprop
        }
        st.session_state.last_search = search_params
        
        # Create tabs for different analyses
        if show_interest_over_time:
            tab1 = st.tabs(["📈 Interest Over Time"])[0]
        if show_interest_by_region:
            tab2 = st.tabs(["🗺️ Interest by Region"])[0]
        if show_related_queries:
            tab3 = st.tabs(["🔍 Related Queries"])[0]
        if show_related_topics:
            tab4 = st.tabs(["📚 Related Topics"])[0]
        if show_trending:
            tab5 = st.tabs(["🔥 Trending Searches"])[0]
        if show_suggestions:
            tab6 = st.tabs(["💡 Suggestions"])[0]
        
        # Interest Over Time
        if show_interest_over_time:
            with st.container():
                st.subheader("📈 Interest Over Time")
                with st.spinner("Fetching interest over time data..."):
                    iot_data = analyzer.get_interest_over_time(keywords, timeframe, geo, cat, gprop)
                    
                if iot_data is not None and not iot_data.empty:
                    # Remove 'isPartial' column if it exists
                    if 'isPartial' in iot_data.columns:
                        iot_data = iot_data.drop('isPartial', axis=1)
                    
                    # Create line chart
                    fig = go.Figure()
                    for col in iot_data.columns:
                        fig.add_trace(go.Scatter(
                            x=iot_data.index,
                            y=iot_data[col],
                            mode='lines',
                            name=col,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Search Interest Over Time",
                        xaxis_title="Date",
                        yaxis_title="Interest (0-100)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    with st.expander("📊 View Data Table"):
                        st.dataframe(iot_data, use_container_width=True)
                        
                        # Download button
                        csv = iot_data.to_csv()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"interest_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        # Interest by Region
        if show_interest_by_region:
            with st.container():
                st.subheader("🗺️ Interest by Region")
                
                resolution = st.radio("Resolution", ["COUNTRY", "REGION", "CITY", "DMA"], horizontal=True)
                
                with st.spinner("Fetching regional interest data..."):
                    region_data = analyzer.get_interest_by_region(keywords, timeframe, geo, cat, gprop, resolution)
                
                if region_data is not None and not region_data.empty:
                    # Create choropleth map for countries
                    if resolution == "COUNTRY" and 'geoCode' in region_data.columns:
                        fig = px.choropleth(
                            region_data.reset_index(),
                            locations='geoCode',
                            color=region_data.columns[0],
                            hover_name='geoName' if 'geoName' in region_data.columns else 'index',
                            color_continuous_scale='Viridis',
                            title=f"Interest by Country: {keywords[0]}"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart for top regions
                    top_regions = region_data.head(20)
                    fig_bar = px.bar(
                        top_regions.reset_index(),
                        x=top_regions.columns[0],
                        y='index',
                        orientation='h',
                        title=f"Top 20 {resolution.title()}s by Interest"
                    )
                    fig_bar.update_layout(height=600)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Show data table
                    with st.expander("📊 View Data Table"):
                        st.dataframe(region_data, use_container_width=True)
                        
                        # Download button
                        csv = region_data.to_csv()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"interest_by_region_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        # Related Queries
        if show_related_queries:
            with st.container():
                st.subheader("🔍 Related Queries")
                
                with st.spinner("Fetching related queries..."):
                    queries_data = analyzer.get_related_queries(keywords, timeframe, geo, cat, gprop)
                
                if queries_data:
                    for keyword in keywords:
                        if keyword in queries_data:
                            st.write(f"**{keyword}**")
                            
                            col1, col2 = st.columns(2)
                            
                            # Top queries
                            with col1:
                                st.write("📊 Top Related Queries")
                                top_df = queries_data[keyword]['top']
                                if top_df is not None and not top_df.empty:
                                    st.dataframe(top_df, use_container_width=True)
                                else:
                                    st.info("No top queries found")
                            
                            # Rising queries
                            with col2:
                                st.write("📈 Rising Queries")
                                rising_df = queries_data[keyword]['rising']
                                if rising_df is not None and not rising_df.empty:
                                    st.dataframe(rising_df, use_container_width=True)
                                else:
                                    st.info("No rising queries found")
                            
                            st.divider()
        
        # Related Topics
        if show_related_topics:
            with st.container():
                st.subheader("📚 Related Topics")
                
                with st.spinner("Fetching related topics..."):
                    topics_data = analyzer.get_related_topics(keywords, timeframe, geo, cat, gprop)
                
                if topics_data:
                    for keyword in keywords:
                        if keyword in topics_data:
                            st.write(f"**{keyword}**")
                            
                            col1, col2 = st.columns(2)
                            
                            # Top topics
                            with col1:
                                st.write("📊 Top Related Topics")
                                top_df = topics_data[keyword]['top']
                                if top_df is not None and not top_df.empty:
                                    st.dataframe(top_df, use_container_width=True)
                                else:
                                    st.info("No top topics found")
                            
                            # Rising topics
                            with col2:
                                st.write("📈 Rising Topics")
                                rising_df = topics_data[keyword]['rising']
                                if rising_df is not None and not rising_df.empty:
                                    st.dataframe(rising_df, use_container_width=True)
                                else:
                                    st.info("No rising topics found")
                            
                            st.divider()
        
        # Trending Searches
        if show_trending:
            with st.container():
                st.subheader("🔥 Trending Searches")
                
                country_code = st.selectbox(
                    "Select Country",
                    ["united_kingdom", "united_states", "canada", "australia", "india", 
                     "germany", "france", "japan", "brazil", "mexico"],
                    index=0
                )
                
                with st.spinner("Fetching trending searches..."):
                    trending_data = analyzer.get_trending_searches(country_code)
                
                if trending_data is not None and not trending_data.empty:
                    st.dataframe(trending_data, use_container_width=True)
                    
                    # Download button
                    csv = trending_data.to_csv()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"trending_searches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Keyword Suggestions
        if show_suggestions:
            with st.container():
                st.subheader("💡 Keyword Suggestions")
                
                for keyword in keywords:
                    st.write(f"**Suggestions for: {keyword}**")
                    
                    with st.spinner(f"Fetching suggestions for {keyword}..."):
                        suggestions = analyzer.get_suggestions(keyword)
                    
                    if suggestions:
                        suggestions_df = pd.DataFrame(suggestions)
                        st.dataframe(suggestions_df, use_container_width=True)
                    else:
                        st.info(f"No suggestions found for {keyword}")
                    
                    st.divider()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Google Trends Analyzer | Built with Streamlit and PyTrends</p>
        <p>⚠️ Note: Google Trends has rate limits. If you encounter errors, please wait 60 seconds before retrying.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
