"""
Stockify - AI-Enhanced Stock Market Predictor
Main application dashboard for stock price prediction and analysis

Features:
- Real-time stock data fetching
- LSTM neural network predictions
- Moving average analysis
- AI-powered insights via local LLM
"""

import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
from utils.stock_utils import (
    download_stock_data, load_prediction_model, init_local_llm,
    prepare_prediction_data, make_predictions, calculate_metrics,
    initialize_session_state, validate_stock_input, display_llm_sidebar_status,
    display_market_info
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Stockify - AI Stock Predictor", 
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Load model and LLM service
model = load_prediction_model()
llm_service = init_local_llm()

st.title('🚀 Stockify - AI-Enhanced Stock Market Predictor')
st.markdown("*Powered by Local LLM + LSTM Neural Network*")

st.sidebar.markdown("### ℹ️ **About:**")
st.sidebar.info("""
This app provides AI-enhanced stock market predictions using:
- **LSTM Neural Network** for price prediction
- **Local LLM** for intelligent analysis
- **Technical Indicators** for trend analysis
""")

# Display LLM status in sidebar
display_llm_sidebar_status(llm_service)


# Main content
st.header('📊 Stock Data & Price Prediction')

stock = st.text_input('Enter Stock Symbol', '', help="Enter a stock symbol (e.g., AAPL, TSLA, INFY.NS)")


# Validate stock input
stock_symbol, error_message = validate_stock_input(stock)
if error_message:
    display_market_info()
    st.stop()

start = '2012-01-01'
end = '2022-12-31'

# Clear cache if stock symbol changes
if stock_symbol != st.session_state.current_stock:
    st.session_state.llm_cache = {}
    st.session_state.current_stock = stock_symbol
    st.session_state.stock_data = None
    st.session_state.predictions = None
    st.session_state.metrics = None

# Download and process stock data
if stock_symbol:
    with st.spinner(f"Loading data for {stock_symbol}..."):
        data, error = download_stock_data(stock_symbol, start, end)
        
        if error:
            st.error(f'❌ Error downloading data for {stock_symbol}: {error}')
            st.info('💡 Please check your internet connection and stock symbol.')
            st.stop()
        
        if data is None or data.empty:
            st.error(f'❌ No data found for stock symbol: {stock_symbol}')
            st.info('💡 Please check if the stock symbol is correct and try again.')
            st.stop()
        
        # Check data sufficiency
        if len(data) < 200:
            st.error(f'❌ Insufficient data for analysis. Only {len(data)} days of data available.')
            st.info('💡 The model requires at least 200 days of data for accurate predictions.')
            st.stop()
    
    # Store data in session state
    st.session_state.stock_data = data
    
    # Display stock data
    st.subheader('📈 Stock Data Overview')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${data.Close.iloc[-1].item():.2f}")
    with col2:
        price_change = ((data.Close.iloc[-1] - data.Close.iloc[-2]) / data.Close.iloc[-2]) * 100
        st.metric("Daily Change", f"{price_change.item():.2f}%")
    with col3:
        st.metric("Volume", f"{data.Volume.iloc[-1].item():,.0f}" if 'Volume' in data.columns else "N/A")
    with col4:
        st.metric("Data Points", f"{len(data):,}")
    
    # Show expandable data table
    with st.expander("📊 View Full Dataset", expanded=False):
        st.dataframe(data, width='stretch')
    
    # Prepare data for prediction
    data_test_scale, scaler = prepare_prediction_data(data)
    predict, y = make_predictions(model, data_test_scale, scaler)
    
    # Store predictions in session state
    st.session_state.predictions = (predict, y)
    
    # Calculate and store metrics
    metrics = calculate_metrics(data)
    st.session_state.metrics = metrics
    
    # Display charts
    st.subheader('📊 Technical Analysis Charts')
    
    # Chart tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Price vs MA50", "Price vs MA50 vs MA100", "Price vs MA100 vs MA200", "Actual vs Predicted"])
    
    with tab1:
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = plt.figure(figsize=(10,6))
        plt.plot(ma_50_days, 'r', label='MA50', linewidth=2)
        plt.plot(data.Close, 'g', label='Closing Price', linewidth=1)
        plt.title(f'{stock_symbol} - Price vs 50-Day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig1)
    
    with tab2:
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(10,6))
        plt.plot(ma_50_days, 'r', label='MA50', linewidth=2)
        plt.plot(ma_100_days, 'b', label='MA100', linewidth=2)
        plt.plot(data.Close, 'g', label='Closing Price', linewidth=1)
        plt.title(f'{stock_symbol} - Price vs Moving Averages (50 & 100 Days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    with tab3:
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(10,6))
        plt.plot(ma_100_days, 'r', label='MA100', linewidth=2)
        plt.plot(ma_200_days, 'b', label='MA200', linewidth=2)
        plt.plot(data.Close, 'g', label='Closing Price', linewidth=1)
        plt.title(f'{stock_symbol} - Price vs Moving Averages (100 & 200 Days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig3)
    
    with tab4:
        fig4 = plt.figure(figsize=(10,6))
        plt.plot(y, 'g', label='Actual Price', linewidth=2)
        plt.plot(predict, 'r', label='Predicted Price', linewidth=2)
        plt.title(f'{stock_symbol} - Actual vs Predicted Stock Price')
        plt.xlabel('Number of Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig4)
    
    # Model Performance Metrics
    st.subheader('🎯 Model Performance')
    
    from utils.stock_utils import calculate_prediction_accuracy
    accuracy = calculate_prediction_accuracy(y, predict)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
    with col2:
        mae = abs(y - predict).mean()
        st.metric("Mean Absolute Error", f"${mae:.2f}")
    with col3:
        next_prediction = predict[-1]
        st.metric("Next Day Prediction", f"${next_prediction.item():.2f}")
