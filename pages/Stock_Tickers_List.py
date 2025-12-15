"""
Stock Tickers List Page - Browse Available Stock Symbols
Displays searchable lists of active stocks from NASDAQ and NSE markets

Features:
- NASDAQ (US) stock listings
- NSE (India) stock listings
- Search and filter functionality
- Market statistics
- Company information display
"""

import streamlit as st
import pandas as pd
import os
from utils.stock_utils import get_currency_symbol

# Page config
st.set_page_config(
    page_title="Stock Tickers List - Stockify",
    page_icon="📋",
    layout="wide"
)

# Title and description
st.title("📋 Stock Tickers List")
st.markdown("*Browse and search active stock tickers from NASDAQ and NSE markets*")

# Helper function to load CSV data
@st.cache_data
def load_stock_data(file_path):
    """
    Load stock ticker data from CSV file with caching
    
    Args:
        file_path: Path to the CSV file containing stock data
        
    Returns:
        tuple: (DataFrame with stock data, error message or None)
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df, None
        else:
            return None, f"File not found: {file_path}"
    except Exception as e:
        return None, str(e)

# Load data from both CSV files
nasdaq_path = "./resources/NASDAQ_Active_Stocks_List.csv"
nse_path = "./resources/NSE_Active_Stocks_List.csv"

nasdaq_data, nasdaq_error = load_stock_data(nasdaq_path)
nse_data, nse_error = load_stock_data(nse_path)

# Sidebar - Market selection and filters
st.sidebar.header("🔍 Filters")

# Market selection
market_options = []
if nasdaq_data is not None:
    market_options.append("NASDAQ (US)")
if nse_data is not None:
    market_options.append("NSE (India)")

if not market_options:
    st.error("❌ No stock data available. Please check if CSV files exist in the resources folder.")
    st.stop()

selected_market = st.sidebar.radio("Select Market", market_options, index=0)

# Select data based on market choice
if selected_market == "NASDAQ (US)":
    current_data = nasdaq_data
    market_code = "NASDAQ"
else:
    current_data = nse_data
    market_code = "NSE"


# Main content
if current_data is not None and not current_data.empty:
    
    # Search functionality
    st.subheader("🔎 Search Stocks")
    
    search_query = st.text_input(
        "Search by symbol or company name",
        "",
        help="Enter stock symbol or company name to search"
    )
    
    # Filter data based on search
    filtered_data = current_data.copy()
    
    if search_query:
        search_query = search_query.upper()
        if market_code == "NASDAQ":
            mask = (
                filtered_data['SYMBOL'].str.contains(search_query, na=False) |
                filtered_data['NAME OF COMPANY'].str.upper().str.contains(search_query, na=False)
            )
        else:  # NSE
            mask = (
                filtered_data['SYMBOL'].str.contains(search_query, na=False) |
                filtered_data['NAME OF COMPANY'].str.upper().str.contains(search_query, na=False)
            )
        filtered_data = filtered_data[mask]
    
    # Display statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Total Stocks", len(current_data))
    
    with col2:
        st.metric("✅ Filtered Results", len(filtered_data))
    
    with col3:
        if 'Last_Price(5d)' in filtered_data.columns:
            active_prices = filtered_data[filtered_data['Last_Price(5d)'] > 0]
            st.metric("💰 With Prices", len(active_prices))
        else:
            st.metric("💰 With Prices", "N/A")
    
    with col4:
        if 'Last_Price(5d)' in filtered_data.columns:
            avg_price = filtered_data[filtered_data['Last_Price(5d)'] > 0]['Last_Price(5d)'].mean()
            # Determine currency based on market
            currency = '₹' if market_code == 'NSE' else '$'
            st.metric("📊 Avg Price", f"{currency}{avg_price:.2f}" if not pd.isna(avg_price) else "N/A")
        else:
            st.metric("📊 Avg Price", "N/A")
    
    # Display results
    st.subheader("📊 Stock Tickers")
    
    # Add sorting options
    col1, col2 = st.columns([1, 4])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Symbol", "Company Name", "Price (High to Low)", "Price (Low to High)"],
            index=0
        )
    
    # Apply sorting
    if sort_by == "Symbol":
        filtered_data = filtered_data.sort_values('SYMBOL')
    elif sort_by == "Company Name":
        filtered_data = filtered_data.sort_values('NAME OF COMPANY')
    elif sort_by == "Price (High to Low)" and 'Last_Price(5d)' in filtered_data.columns:
        filtered_data = filtered_data.sort_values('Last_Price(5d)', ascending=False)
    elif sort_by == "Price (Low to High)" and 'Last_Price(5d)' in filtered_data.columns:
        filtered_data = filtered_data.sort_values('Last_Price(5d)', ascending=True)
    
    # Display data table
    if len(filtered_data) > 0:
        # Customize display columns based on market
        if market_code == "NASDAQ":
            display_columns = ['SYMBOL', 'NAME OF COMPANY', 'Status', 'Last_Price(5d)']
        else:  # NSE
            display_columns = ['SYMBOL', 'SUFFIX', 'NAME OF COMPANY', 'Status', 'Last_Price(5d)']
        
        # Create display dataframe
        display_df = filtered_data[display_columns].copy()
        
        # Format price column
        if 'Last_Price(5d)' in display_df.columns:
            currency = '₹' if market_code == 'NSE' else '$'
            display_df['Last_Price(5d)'] = display_df['Last_Price(5d)'].apply(
                lambda x: f"{currency}{x:.2f}" if pd.notna(x) and x > 0 else "N/A"
            )
        
        # Display with styling
        st.dataframe(
            display_df,
            width='stretch',
            height=500,
            hide_index=True
        )
        
        # Quick access section
        st.markdown("---")
        st.subheader("🚀 Quick Access - Popular Stocks")
        
        if market_code == "NASDAQ":
            popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX']
            popular_stocks = current_data[current_data['SYMBOL'].isin(popular_symbols)]
        else:  # NSE
            popular_symbols = ['INFY', 'TCS', 'RELIANCE', 'HDFC', 'ICICIBANK', 'SBIN', 'WIPRO', 'ITC', 'LT', 'HDFCBANK']
            popular_stocks = current_data[current_data['SYMBOL'].isin(popular_symbols)]
        
        if not popular_stocks.empty:
            cols = st.columns(5)
            for idx, (_, stock) in enumerate(popular_stocks.head(10).iterrows()):
                with cols[idx % 5]:
                    symbol = stock['SYMBOL']
                    if market_code == "NSE":
                        full_symbol = f"{symbol}.NS"
                    else:
                        full_symbol = symbol
                    
                    price = stock['Last_Price(5d)'] if pd.notna(stock['Last_Price(5d)']) else 0
                    currency = '₹' if market_code == 'NSE' else '$'
                    
                    st.markdown(f"""
                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px 0;'>
                        <b>{symbol}</b><br>
                        <small>{stock['NAME OF COMPANY'][:30]}...</small><br>
                        <span style='color: green; font-weight: bold;'>{currency}{price:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Popular stocks data not available")
        
        # Download section
        st.markdown("---")
        st.subheader("💾 Export Data")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("📥 Download the filtered stock list as CSV")
        with col2:
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{market_code}_stocks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("⚠️ No stocks found matching your search criteria.")
        st.info("💡 Try adjusting your search query or filters.")

else:
    st.error("❌ Unable to load stock data.")
    if nasdaq_error:
        st.error(f"NASDAQ Error: {nasdaq_error}")
    if nse_error:
        st.error(f"NSE Error: {nse_error}")

# Footer with helpful information
st.markdown("---")
st.markdown("""
### 📚 How to Use Stock Symbols

**US NASDAQ Stocks:**
- Use the symbol directly (e.g., `AAPL`, `MSFT`, `GOOGL`)
- Example: Enter `AAPL` in the main app to analyze Apple Inc.

**Indian NSE Stocks:**
- Add `.NS` suffix to the symbol (e.g., `INFY.NS`, `TCS.NS`, `RELIANCE.NS`)
- Example: Enter `INFY.NS` in the main app to analyze Infosys

**Tips:**
- Use the search box to quickly find stocks by symbol or company name
- Filter by status to show only active/listed stocks
- Sort results by symbol, company name, or price
- Download filtered results as CSV for offline reference
""")


st.sidebar.markdown("### ℹ️ **About:**")
st.sidebar.info("""
This page provides a comprehensive list of active stock tickers from:
- **NASDAQ**: US stock market
- **NSE**: Indian stock market

Use this to:
- Find stock symbols
- Browse available stocks
- Export stock lists
- Quick reference for trading
""")
