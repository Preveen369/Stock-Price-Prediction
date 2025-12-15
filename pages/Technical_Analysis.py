"""
Technical Analysis Page - AI-Powered Stock Technical Analysis
Provides detailed technical indicator analysis and trend interpretation

Features:
- Moving average analysis (50, 100, 200-day)
- Trend detection and classification
- AI-generated technical insights
- Price pattern analysis
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stock_utils import (
    init_local_llm, 
    prepare_stock_data_for_llm, display_llm_sidebar_status,
    get_trend_info, get_currency_symbol
)

st.set_page_config(
    page_title="Technical Analysis - Stockify", 
    page_icon="📊",
    layout="wide"
)

# Initialize LLM service
llm_service = init_local_llm()

st.title('📊 Technical Analysis')
st.markdown("*AI-powered technical analysis of stock trends and indicators*")

st.sidebar.markdown("### ℹ️ **About:**")
st.sidebar.info("""
This app provides AI-enhanced stock market predictions using:
- **LSTM Neural Network** for price prediction
- **Local LLM** for intelligent analysis
- **Technical Indicators** for trend analysis
""")

# Display LLM status in sidebar
display_llm_sidebar_status(llm_service)


st.sidebar.markdown("### 🔍 **Analysis Features:**")
st.sidebar.info("""
- **Moving Average Analysis**
- **Trend Detection**
- **Price Action Patterns**
- **Technical Indicators**
- **AI Interpretation**
""")

# Check if we have stock data from session state
if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
    st.warning("⚠️ No stock data available. Please go to the Main Dashboard and select a stock first.")
    if st.button("🏠 Go to Main Dashboard"):
        st.switch_page("app.py")
    st.stop()

data = st.session_state.stock_data
stock_symbol = st.session_state.current_stock
metrics = st.session_state.metrics
predictions = st.session_state.predictions

if predictions is None:
    st.warning("⚠️ No prediction data available. Please refresh the Main Dashboard.")
    if st.button("🏠 Go to Main Dashboard"):
        st.switch_page("app.py")
    st.stop()

predict, y = predictions

# Header with stock info
st.subheader(f'📊 Technical Analysis for {stock_symbol}')

# Quick metrics
col1, col2, col3, col4 = st.columns(4)

trend_emoji, trend_text, trend_color = get_trend_info(
    metrics['price_vs_ma50'], metrics['price_vs_ma100']
)

currency = get_currency_symbol(stock_symbol)

with col1:
    st.metric("💲 Current Price", f"{currency}{metrics['latest_price'].item():.2f}")

with col2:
    st.metric("📈 Trend", f"{trend_emoji} {trend_text}")

with col3:
    st.metric("📊 Volatility", f"{metrics['volatility'].item():.2f}%")

# Moving Average Analysis
st.subheader('📈 Moving Average Analysis')
    
col1, col2, col3 = st.columns(3)
    
with col1:
    ma50_val = metrics['price_vs_ma50'].item()
    ma50_status = "Above" if ma50_val > 0 else "Below"
    ma50_color = "green" if ma50_val > 0 else "red"
    st.metric(
        "50-Day MA Position", 
        f"{ma50_status} ({ma50_val:.1f}%)",
        delta=None
    )
    st.markdown(f"**MA50 Price:** {currency}{metrics['ma_50'].item():.2f}")
    
with col2:
    ma100_val = metrics['price_vs_ma100'].item()
    ma100_status = "Above" if ma100_val > 0 else "Below"
    ma100_color = "green" if ma100_val > 0 else "red"
    st.metric(
        "100-Day MA Position", 
        f"{ma100_status} ({ma100_val:.1f}%)",
        delta=None
    )
    st.markdown(f"**MA100 Price:** {currency}{metrics['ma_100'].item():.2f}")
    
with col3:
    ma200_val = metrics['price_vs_ma200'].item()
    ma200_status = "Above" if ma200_val > 0 else "Below"
    ma200_color = "green" if ma200_val > 0 else "red"
    st.metric(
        "200-Day MA Position", 
        f"{ma200_status} ({ma200_val:.1f}%)",
        delta=None
    )
    st.markdown(f"**MA200 Price:** {currency}{metrics['ma_200'].item():.2f}")

# Technical Analysis Content
if llm_service.check_connection():
    st.success("✅ AI Technical Analysis Available")

    # AI Analysis Section
    st.markdown("---")
    st.subheader('🤖 AI Technical Analysis Report')
    
    # Prepare stock data for LLM analysis
    stock_data = prepare_stock_data_for_llm(stock_symbol, metrics)
    
    # Add prediction accuracy to the data
    stock_data["technical_data"] += f"\nModel Prediction Accuracy: {accuracy:.1f}%"
    
    # Check if analysis already exists in cache
    cache_key = f"technical_{stock_symbol}"
    
    if cache_key in st.session_state.llm_cache:
        # Show expandable view with cached results
        with st.expander("📊 View AI Technical Analysis Results", expanded=True):
            st.markdown(st.session_state.llm_cache[cache_key])
        
        # Option to regenerate
        if st.button("🔄 Regenerate Analysis", key="regenerate_technical"):
            del st.session_state.llm_cache[cache_key]
            st.rerun()
    else:
        # Show generate button if no cached results
        st.info("🎯 Generate comprehensive AI technical analysis based on moving averages, trends, and price patterns.")
        if st.button("🚀 Generate Analysis", key="technical_generate"):
            # Create a contained streaming area
            with st.container():
                st.markdown("#### 📈 Generating AI Technical Analysis Report...")
                    
                # Create a placeholder for the streaming content
                streaming_placeholder = st.empty()
                    
                try:
                    # Use streaming response with a progress indicator
                    with st.spinner("🤖 AI is analyzing technical indicators..."):
                        def technical_analysis_stream():
                            return llm_service.analyze_stock_fundamentals_stream(stock_data)
                            
                        # Capture the streaming content in the placeholder
                        with streaming_placeholder.container():
                            st.markdown("**📊 Technical Analysis (Live Stream)**")
                            response = st.write_stream(technical_analysis_stream())
                            
                        # Cache the complete response
                        st.session_state.llm_cache[cache_key] = response
                            
                        # Clear the streaming area and show success message
                        streaming_placeholder.empty()
                        st.success("✅ Technical analysis complete! Results displayed below.")
                            
                        # Force rerun to show the cached results
                        st.rerun()
                            
                except Exception as e:
                    streaming_placeholder.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    st.info("💡 Make sure LM Studio is running with a model loaded.") 


else:
    st.markdown("---")
    st.warning("🤖 AI Technical Analysis Unavailable")
    st.info("💡 To enable AI analysis:")
    st.markdown("""
    1. Install and start **LM Studio** from [lmstudio.ai](https://lmstudio.ai)
    2. Download a financial analysis model (recommended: Gemma or Llama)
    3. Start the local server on `localhost:1234`
    4. Refresh this page to enable AI features
    """)

# Disclaimer
st.markdown("---")
st.markdown("""
> **⚠️ Disclaimer:** This analysis is for Educational purposes only. Not to be considered as financial advice. \
Consult a qualified financial advisor and verify data through Screener.in, NASDAQ, and Yahoo Finance. Past performance \
doesn't guarantee future results.
""")