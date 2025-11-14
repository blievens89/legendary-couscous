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
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'results_cache' not in st.session_state:
            st.session_state.results_cache = {}
        if 'last_search' not in st.session_state:
            st.session_state.last_search = None
        if 'pytrends_instance' not in st.session_state:
            st.session_state.pytrends_instance = None
        if 'connection_settings' not in st.session_state:
            st.session_state.connection_settings = None
    
    @property
    def pytrends(self):
        """Property to access the pytrends instance from session state"""
        return st.session_state.pytrends_instance
            
    def create_connection(self, hl='en-US', tz=360, use_proxies=False, proxies_list=None):
        """Create a connection to Google Trends API - creates new instance if settings change"""
        try:
            # Store connection settings to detect changes
            current_settings = {
                'hl': hl,
                'tz': tz,
                'use_proxies': use_proxies,
                'proxies': tuple(proxies_list) if proxies_list else None
            }
            
            # Check if settings changed
            if st.session_state.connection_settings == current_settings and st.session_state.pytrends_instance is not None:
                return True
            
            # Settings changed or no instance - create new connection
            if use_proxies and proxies_list:
                st.session_state.pytrends_instance = TrendReq(
                    hl=hl, 
                    tz=tz,
                    timeout=(10, 25),
                    proxies=proxies_list,
                    retries=2,
                    backoff_factor=0.1,
                    requests_args={'verify': False}
                )
            else:
                st.session_state.pytrends_instance = TrendReq(
                    hl=hl, 
                    tz=tz,
                    timeout=(10, 25),
                    retries=2,
                    backoff_factor=0.1
                )
            
            # Store the settings
            st.session_state.connection_settings = current_settings
            return True
            
        except Exception as e:
            st.error(f"Failed to create connection: {str(e)}")
            return False
    
    def get_historical_interest_data(self, keywords, year_start, month_start, day_start, 
                                     year_end, month_end, day_end, geo='', cat=0, gprop=''):
        """Get historical hourly interest data with built-in rate limiting"""
        try:
            return self.pytrends.get_historical_interest(
                keywords,
                year_start=year_start,
                month_start=month_start,
                day_start=day_start,
                hour_start=0,
                year_end=year_end,
                month_end=month_end,
                day_end=day_end,
                hour_end=0,
                cat=cat,
                geo=geo,
                gprop=gprop,
                sleep=60  # This is the built-in rate limiting
            )
        except Exception as e:
            st.error(f"Error fetching historical interest: {str(e)}")
            return None
    
    def get_related_queries(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related queries"""
        try:
            time.sleep(60)  # Manual rate limiting
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.related_queries()
        except Exception as e:
            st.error(f"Error fetching related queries: {str(e)}")
            return None
    
    def get_related_topics(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related topics"""
        try:
            time.sleep(60)  # Manual rate limiting
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return self.pytrends.related_topics()
        except Exception as e:
            st.error(f"Error fetching related topics: {str(e)}")
            return None
    
    def get_trending_searches(self, pn='united_states'):
        """Get trending searches"""
        try:
            time.sleep(60)  # Manual rate limiting
            try:
                return self.pytrends.trending_searches(pn=pn)
            except:
                country_name = pn.replace('_', ' ').title()
                return self.pytrends.trending_searches(pn=country_name)
        except Exception as e:
            st.error(f"Error fetching trending searches: {str(e)}")
            st.info("Note: Trending searches may not be available for all countries or time periods")
            return None
    
    def get_suggestions(self, keyword):
        """Get keyword suggestions"""
        try:
            time.sleep(60)  # Manual rate limiting
            return self.pytrends.suggestions(keyword)
        except Exception as e:
            st.error(f"Error fetching suggestions: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Google Trends Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Google Trends Analyzer")
    st.markdown("Analyze search trends with built-in rate limiting")
    
    # Initialize analyzer
    analyzer = GoogleTrendsAnalyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection settings
        st.subheader("Connection Settings")
        hl = st.selectbox("Language", ["en-US", "en-GB", "es", "fr", "de", "it", "pt", "ru", "ja", "zh-CN"], index=1)
        tz = st.number_input("Timezone Offset (minutes)", value=0, help="For UK GMT use 0, for US CST use 360")
        
        use_proxies = st.checkbox("Use Proxies (for rate limit issues)")
        proxies_list = None
        if use_proxies:
            proxy_input = st.text_area("Enter proxies (one per line)", 
                                      placeholder="https://34.203.233.13:80\nhttps://35.201.123.31:880")
            if proxy_input:
                proxies_list = [p.strip() for p in proxy_input.strip().split('\n') if p.strip()]
        
        # Create connection
        if analyzer.create_connection(hl, tz, use_proxies, proxies_list):
            st.success("✅ Connected to Google Trends")
        
        st.divider()
        
        # Search parameters
        st.subheader("Search Parameters")
        
        # Keywords input
        keywords_input = st.text_area(
            "Keywords (one per line, max 5)",
            value="Mortgage",
            help="Enter up to 5 keywords to compare"
        )
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()][:5]
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.now() - timedelta(days=90))
        with col2:
            end_date = st.date_input("End date", datetime.now())
        
        # Geographic location
        geo = st.text_input("Geographic Location", value="GB", 
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
        show_interest_over_time = st.checkbox("Historical Interest", value=True)
        show_related_queries = st.checkbox("Related Queries", value=False)
        show_related_topics = st.checkbox("Related Topics", value=False)
        show_trending = st.checkbox("Trending Searches", value=False)
        show_suggestions = st.checkbox("Keyword Suggestions", value=False)
        
        st.warning("⏱️ Each analysis waits 60 seconds between API calls to avoid rate limits")
    
    # Main content area
    if st.button("🔍 Analyze Trends", type="primary"):
        if not keywords:
            st.error("Please enter at least one keyword")
            return
        
        # Check if connection exists
        if analyzer.pytrends is None:
            st.error("Connection not established. Please check configuration.")
            return
        
        # Store search parameters
        search_params = {
            'keywords': keywords,
            'start_date': start_date,
            'end_date': end_date,
            'geo': geo,
            'cat': cat,
            'gprop': gprop
        }
        st.session_state.last_search = search_params
        
        # Create tabs for different analyses
        tab_names = []
        if show_interest_over_time:
            tab_names.append("📈 Historical Interest")
        if show_related_queries:
            tab_names.append("🔍 Related Queries")
        if show_related_topics:
            tab_names.append("📚 Related Topics")
        if show_trending:
            tab_names.append("🔥 Trending Searches")
        if show_suggestions:
            tab_names.append("💡 Suggestions")
        
        if tab_names:
            tabs = st.tabs(tab_names)
            tab_index = 0
        else:
            st.warning("Please select at least one analysis option in the sidebar.")
            return
        
        # Historical Interest
        if show_interest_over_time:
            with tabs[tab_index]:
                st.subheader("📈 Historical Interest Over Time")
                st.info("Using get_historical_interest() with built-in 60-second rate limiting between requests")
                
                with st.spinner("Fetching historical interest data (this will take time due to rate limiting)..."):
                    historical_data = analyzer.get_historical_interest_data(
                        keywords,
                        year_start=start_date.year,
                        month_start=start_date.month,
                        day_start=start_date.day,
                        year_end=end_date.year,
                        month_end=end_date.month,
                        day_end=end_date.day,
                        geo=geo,
                        cat=cat,
                        gprop=gprop
                    )
                    
                if historical_data is not None and not historical_data.empty:
                    # Remove 'isPartial' column if it exists
                    if 'isPartial' in historical_data.columns:
                        historical_data = historical_data.drop('isPartial', axis=1)
                    
                    # Create line chart
                    fig = go.Figure()
                    for col in historical_data.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data[col],
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
                        st.dataframe(historical_data, use_container_width=True)
                        
                        # Download button
                        csv = historical_data.to_csv()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"historical_interest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No data returned. You may still be rate-limited.")
            tab_index += 1
        
        # Related Queries
        if show_related_queries:
            with tabs[tab_index]:
                st.subheader("🔍 Related Queries")
                st.info("Waiting 60 seconds before fetching...")
                
                with st.spinner("Fetching related queries..."):
                    queries_data = analyzer.get_related_queries(keywords, f"{start_date} {end_date}", geo, cat, gprop)
                
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
            tab_index += 1
        
        # Related Topics
        if show_related_topics:
            with tabs[tab_index]:
                st.subheader("📚 Related Topics")
                st.info("Waiting 60 seconds before fetching...")
                
                with st.spinner("Fetching related topics..."):
                    topics_data = analyzer.get_related_topics(keywords, f"{start_date} {end_date}", geo, cat, gprop)
                
                if topics_data:
                    for keyword in keywords:
                        if keyword in topics_data and topics_data[keyword]:
                            st.write(f"**{keyword}**")
                            
                            col1, col2 = st.columns(2)
                            
                            # Top topics
                            with col1:
                                st.write("📊 Top Related Topics")
                                try:
                                    top_df = topics_data[keyword]['top']
                                    if top_df is not None and not top_df.empty:
                                        st.dataframe(top_df, use_container_width=True)
                                    else:
                                        st.info("No top topics found")
                                except (KeyError, IndexError, TypeError):
                                    st.info("No top topics found")
                            
                            # Rising topics
                            with col2:
                                st.write("📈 Rising Topics")
                                try:
                                    rising_df = topics_data[keyword]['rising']
                                    if rising_df is not None and not rising_df.empty:
                                        st.dataframe(rising_df, use_container_width=True)
                                    else:
                                        st.info("No rising topics found")
                                except (KeyError, IndexError, TypeError):
                                    st.info("No rising topics found")
                            
                            st.divider()
                        else:
                            st.info(f"No topics data found for {keyword}")
            tab_index += 1
        
        # Trending Searches
        if show_trending:
            with tabs[tab_index]:
                st.subheader("🔥 Trending Searches")
                
                country_code = st.selectbox(
                    "Select Country",
                    ["united_kingdom", "united_states", "canada", "australia", "india", 
                     "germany", "france", "japan", "brazil", "mexico"],
                    index=0
                )
                
                pn_code = country_code
                
                st.info("Waiting 60 seconds before fetching...")
                with st.spinner("Fetching trending searches..."):
                    trending_data = analyzer.get_trending_searches(pn_code)
                
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
            tab_index += 1
        
        # Keyword Suggestions
        if show_suggestions:
            with tabs[tab_index]:
                st.subheader("💡 Keyword Suggestions")
                
                for keyword in keywords:
                    st.write(f"**Suggestions for: {keyword}**")
                    
                    st.info("Waiting 60 seconds before fetching...")
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
        <p>⚠️ Rate limiting: 60 seconds between each API call</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
