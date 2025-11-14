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
        if 'last_api_call_time' not in st.session_state:
            st.session_state.last_api_call_time = 0
            
    def create_connection(self, hl='en-US', tz=360, use_proxies=False, proxies_list=None):
        """Create a connection to Google Trends API - reuses existing instance"""
        try:
            # Reuse existing instance if available
            if st.session_state.pytrends_instance is not None:
                return True
                
            if use_proxies and proxies_list:
                st.session_state.pytrends_instance = TrendReq(
                    hl=hl, 
                    tz=tz,
                    timeout=(10, 25),
                    proxies=proxies_list,
                    retries=2,
                    backoff_factor=0.1
                )
            else:
                st.session_state.pytrends_instance = TrendReq(
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
    
    def enforce_rate_limit(self):
        """Enforce 60 second delay between API calls"""
        time_since_last_call = time.time() - st.session_state.last_api_call_time
        
        if time_since_last_call < 60:
            wait_time = int(60 - time_since_last_call)
            with st.spinner(f"⏱️ Rate limiting: waiting {wait_time} seconds..."):
                time.sleep(wait_time)
        
        st.session_state.last_api_call_time = time.time()
    
    def get_interest_over_time(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get interest over time data"""
        try:
            self.enforce_rate_limit()
            st.session_state.pytrends_instance.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return st.session_state.pytrends_instance.interest_over_time()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching interest over time: {str(e)}")
            return None
    
    def get_interest_by_region(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop='', resolution='COUNTRY'):
        """Get interest by region data"""
        try:
            self.enforce_rate_limit()
            st.session_state.pytrends_instance.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return st.session_state.pytrends_instance.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching interest by region: {str(e)}")
            return None
    
    def get_related_queries(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related queries"""
        try:
            self.enforce_rate_limit()
            st.session_state.pytrends_instance.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return st.session_state.pytrends_instance.related_queries()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching related queries: {str(e)}")
            return None
    
    def get_related_topics(self, keywords, timeframe='today 5-y', geo='', cat=0, gprop=''):
        """Get related topics"""
        try:
            self.enforce_rate_limit()
            st.session_state.pytrends_instance.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
            return st.session_state.pytrends_instance.related_topics()
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching related topics: {str(e)}")
            return None
    
    def get_trending_searches(self, pn='united_states'):
        """Get trending searches"""
        try:
            self.enforce_rate_limit()
            # Try different country code formats if the first one fails
            try:
                return st.session_state.pytrends_instance.trending_searches(pn=pn)
            except:
                # Try with just the country name
                country_name = pn.replace('_', ' ').title()
                return st.session_state.pytrends_instance.trending_searches(pn=country_name)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching trending searches: {str(e)}")
                st.info("Note: Trending searches may not be available for all countries or time periods")
            return None
    
    def get_suggestions(self, keyword):
        """Get keyword suggestions"""
        try:
            self.enforce_rate_limit()
            return st.session_state.pytrends_instance.suggestions(keyword)
        except Exception as e:
            if '429' in str(e):
                self.handle_rate_limit_error(e)
            else:
                st.error(f"Error fetching suggestions: {str(e)}")
            return None
    
    def handle_rate_limit_error(self, e=None):
        """Handle rate limit error with actual blocking wait"""
        error_msg = f"Error details: {str(e)}" if e else ""
        st.error(f"""
        ⚠️ **Rate Limit Reached**
        
        Google Trends has rate limits. Waiting 60 seconds before retry...
        
        {error_msg}
        """)
        
        # Actually wait 60 seconds
        progress_bar = st.progress(0)
        for i in range(60):
            time.sleep(1)
            progress_bar.progress((i + 1) / 60)
        
        progress_bar.empty()
        st.success("✅ Wait complete. You can try again now.")
        st.session_state.last_api_call_time = time.time()

# Rest of main() function stays the same...
