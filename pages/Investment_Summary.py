import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stock_utils import (
    init_local_llm, calculate_prediction_accuracy, 
    display_llm_sidebar_status
)

st.set_page_config(
    page_title="Investment Summary - Stockify", 
    page_icon="📈",
    layout="wide"
)

# Initialize LLM service
llm_service = init_local_llm()

st.title('📈 Investment Summary')
st.markdown("*AI-powered investment recommendations and risk assessment*")

st.sidebar.markdown("### ℹ️ **About:**")
st.sidebar.info("""
This app provides AI-enhanced stock market predictions using:
- **LSTM Neural Network** for price prediction
- **Local LLM** for intelligent analysis
- **Technical Indicators** for trend analysis
""")

# Display LLM status in sidebar
display_llm_sidebar_status(llm_service)


st.sidebar.markdown("### 💡 **Investment Focus:**")
st.sidebar.info("""
- **Risk Assessment**
- **Entry/Exit Points**
- **Portfolio Recommendations**
- **Market Outlook**
- **Investment Strategy**
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
st.subheader(f'📈 Investment Summary for {stock_symbol}')

# Investment Overview Metrics
col1, col2, col3 = st.columns(3)

next_prediction = predict[-1]
prediction_change = ((next_prediction - metrics['latest_price']) / metrics['latest_price']) * 100
range_min = data.Close.min()
range_max = data.Close.max()
range_position = ((metrics['latest_price'] - range_min) / (range_max - range_min)) * 100
accuracy = calculate_prediction_accuracy(y, predict)

with col1:
    pred_change_val = prediction_change.item()
    st.metric(
        label="⏭️ Next Prediction", 
        value=f"${next_prediction.item():.2f}"
    )

with col2:
    vol_level_val = metrics['volatility'].item()
    volatility_level = "High" if vol_level_val > 3 else "Medium" if vol_level_val > 1.5 else "Low"
    st.metric(label="⚡ Risk Level", value=volatility_level)
    
with col3:
    st.metric(label="📍 Position in Range", value=f"{range_position.item():.0f}%")
    

if llm_service.check_connection():
    st.success("✅ AI Investment Summary Available")

    # AI Investment Summary
    st.markdown("---")
    st.subheader('🤖 AI Investment Summary')
    
    # Prepare comprehensive data for investment analysis
    latest_price_val = metrics['latest_price'].item()
    volatility_val = metrics['volatility'].item()
    ma50_val = metrics['price_vs_ma50'].item()
    ma100_val = metrics['price_vs_ma100'].item()
    ma200_val = metrics['price_vs_ma200'].item()
    price_change_val = metrics['price_change'].item()
    accuracy_val = accuracy.item()
    next_pred_val = next_prediction.item()
    pred_change_val = prediction_change.item()
    range_min_val = range_min.item()
    range_max_val = range_max.item()
    range_pos_val = range_position.item()
    
    technical_summary = f"""
    Technical Analysis Summary for {stock_symbol}:
    - Current Price: ${latest_price_val:.2f}
    - Trend: {'Bullish' if ma50_val > 0 and ma100_val > 0 else 'Bearish' if ma50_val < 0 and ma100_val < 0 else 'Neutral'}
    - Model Prediction Accuracy: {accuracy_val:.1f}%
    - Volatility: {volatility_val:.2f}%
    - Price vs MA50: {ma50_val:+.1f}%
    - Price vs MA100: {ma100_val:+.1f}%
    - Price vs MA200: {ma200_val:+.1f}%
    - Model Next Day Prediction: ${next_pred_val:.2f} ({pred_change_val:+.1f}%)
    """
    
    price_metrics = f"""
    Price & Risk Metrics for {stock_symbol}:
    - 52-Week Range: ${range_min_val:.2f} - ${range_max_val:.2f}
    - Current Position in Range: {range_pos_val:.1f}%
    - Daily Change: {price_change_val:+.2f}%
    - Average Daily Volume: {metrics['volume_avg']}
    - Volatility Level: {'High' if volatility_val > 3 else 'Medium' if volatility_val > 1.5 else 'Low'}
    - Model Confidence: {'High' if accuracy_val > 85 else 'Medium' if accuracy_val > 75 else 'Low'}
    """
    
    # Check if summary already exists in cache
    summary_cache_key = f"summary_{stock_symbol}"
    
    if summary_cache_key in st.session_state.llm_cache:
        # Show expandable view with cached results
        with st.expander("📋 View AI Investment Summary Results", expanded=True):
            st.markdown(st.session_state.llm_cache[summary_cache_key])
        
        # Option to regenerate
        if st.button("🔄 Regenerate Summary", key="regenerate_summary"):
            del st.session_state.llm_cache[summary_cache_key]
            st.rerun()
    else:
        # Show generate button if no cached results
        st.info("💡 Generate comprehensive AI investment recommendations including entry/exit strategies, risk assessment, and portfolio advice.")
        
        if st.button("🚀 Generate Summary", key="summary_generate", use_container_width=True):
            # Create a contained streaming area
            with st.container():
                st.markdown("#### 📊 Generating AI Investment Summary...")
                    
                # Create a placeholder for the streaming content
                streaming_placeholder = st.empty()
                    
                try:
                    # Use streaming response with a progress indicator
                    with st.spinner("🤖 AI is generating investment recommendations..."):
                        def investment_summary_stream():
                            return llm_service.generate_investment_summary_stream(
                                technical_summary, price_metrics
                            )
                            
                        # Capture the streaming content in the placeholder
                        with streaming_placeholder.container():
                            st.markdown("**📋 Investment Summary (Live Stream)**")
                            response = st.write_stream(investment_summary_stream())
                            
                        # Cache the result
                        st.session_state.llm_cache[summary_cache_key] = response
                            
                        # Clear the streaming area and show success message
                        streaming_placeholder.empty()
                        st.success("✅ Investment summary complete! Results displayed below.")
                            
                        # Force rerun to show the cached results
                        st.rerun()
                            
                except Exception as e:
                    streaming_placeholder.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    st.info("💡 Make sure LM Studio is running with a model loaded.")

else:
    st.markdown("---")
    st.warning("🤖 AI Investment Summary Unavailable")
    st.info("💡 To enable AI analysis:")
    st.markdown("""
    1. Install and start **LM Studio** from [lmstudio.ai](https://lmstudio.ai)
    2. Download a financial analysis model (recommended: Gemma or Llama)
    3. Start the local server on `localhost:1234`
    4. Refresh this page to enable AI features
    """)
    

# Quick Action Recommendations
st.markdown("---")
st.subheader('🎯 Quick Recommendations')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 **Entry Strategy**")
    range_val = range_position.item()
    ma50_check = metrics['price_vs_ma50'].item()
    
    if range_val < 40 and ma50_check > 0:
        st.success("🟢 **GOOD BUY OPPORTUNITY**")
        st.markdown("- Price in favorable range")
        st.markdown("- Positive trend signals")
    elif range_val > 70:
        st.warning("🟡 **WAIT FOR PULLBACK**")
        st.markdown("- Price near resistance")
        st.markdown("- Consider waiting")
    else:
        st.info("🔵 **MONITOR CLOSELY**")
        st.markdown("- Mixed signals")
        st.markdown("- Wait for confirmation")

with col2:
    st.markdown("### 🎯 **Price Targets**")
    st.markdown(f"**Immediate Target:** ${next_prediction.item():.2f}")
    st.markdown(f"**Resistance Level:** ${range_max.item():.2f}")
    st.markdown(f"**Support Level:** ${range_min.item():.2f}")
    st.markdown(f"**Stop Loss:** ${(metrics['latest_price'] * 0.95).item():.2f}")

with col3:
    st.markdown("### ⏱️ **Time Horizon**")
    vol_val = metrics['volatility'].item()
    if vol_val < 2:
        st.success("**Long-term Hold ✓**")
        st.markdown("- Low volatility")
        st.markdown("- Suitable for long positions")
    elif vol_val < 4:
        st.warning("**Medium-term Trade**")
        st.markdown("- Moderate volatility")
        st.markdown("- Monitor closely")
    else:
        st.error("**Short-term Only**")
        st.markdown("- High volatility")
        st.markdown("- Day trading suitable")

# Disclaimer
st.markdown("---")
st.markdown("""
> **⚠️ Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. 
> Always conduct your own research and consult with a qualified financial advisor before making investment decisions. 
> Past performance does not guarantee future results.
""")