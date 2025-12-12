import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from services.local_llm_service import LocalLLMService
from config.llm_config import LM_STUDIO_CONFIG


@st.cache_data
def download_stock_data(stock_symbol, start_date='2012-01-01', end_date='2022-12-31'):
    """Download stock data with caching"""
    try:
        data = yf.download(stock_symbol.strip().upper(), start_date, end_date)
        return data, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_prediction_model():
    """Load the Keras model with caching"""
    return load_model('./models/Stock Predictions Model.keras')


@st.cache_resource
def init_local_llm():
    """Initialize and cache the local LLM service"""
    return LocalLLMService(LM_STUDIO_CONFIG["base_url"])


def prepare_prediction_data(data):
    """Prepare data for model prediction"""
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    
    return data_test_scale, scaler


def make_predictions(model, data_test_scale, scaler):
    """Generate predictions using the model"""
    x = []
    y = []
    
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])
    
    x, y = np.array(x), np.array(y)
    
    predict = model.predict(x)
    
    scale = 1/scaler.scale_
    predict = predict * scale
    y = y * scale
    
    return predict, y


def calculate_metrics(data):
    """Calculate key financial metrics"""
    latest_price = data.Close.iloc[-1]
    price_change = ((data.Close.iloc[-1] - data.Close.iloc[-2]) / data.Close.iloc[-2]) * 100
    volatility = data.Close.pct_change().std() * 100
    volume_avg = data.Volume.rolling(20).mean().iloc[-1] if 'Volume' in data.columns else "N/A"
    
    # Moving averages
    ma_50 = data.Close.rolling(50).mean().iloc[-1]
    ma_100 = data.Close.rolling(100).mean().iloc[-1]
    ma_200 = data.Close.rolling(200).mean().iloc[-1]
    
    # Price relative to moving averages
    price_vs_ma50 = ((latest_price - ma_50) / ma_50) * 100
    price_vs_ma100 = ((latest_price - ma_100) / ma_100) * 100
    price_vs_ma200 = ((latest_price - ma_200) / ma_200) * 100
    
    return {
        'latest_price': latest_price,
        'price_change': price_change,
        'volatility': volatility,
        'volume_avg': volume_avg,
        'ma_50': ma_50,
        'ma_100': ma_100,
        'ma_200': ma_200,
        'price_vs_ma50': price_vs_ma50,
        'price_vs_ma100': price_vs_ma100,
        'price_vs_ma200': price_vs_ma200
    }


def calculate_prediction_accuracy(y_actual, y_predicted):
    """Calculate model prediction accuracy"""
    return 100 - (np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100)


def prepare_stock_data_for_llm(stock_symbol, metrics):
    """Prepare stock data for LLM analysis"""
    return {
        "symbol": stock_symbol.strip().upper(),
        "company_name": f"Company {stock_symbol.strip().upper()}",
        "sector": "Technology",
        "financial_data": f"""
        Latest Price: ${metrics['latest_price'].item():.2f}
        Daily Change: {metrics['price_change'].item():.2f}%
        20-Day Volatility: {metrics['volatility'].item():.2f}%
        Average Volume: {metrics['volume_avg'].item()}
        """,
        "technical_data": f"""
        50-Day MA: ${metrics['ma_50'].item():.2f} (Price is {metrics['price_vs_ma50'].item():.1f}% {'above' if metrics['price_vs_ma50'].item() > 0 else 'below'})
        100-Day MA: ${metrics['ma_100'].item():.2f} (Price is {metrics['price_vs_ma100'].item():.1f}% {'above' if metrics['price_vs_ma100'].item() > 0 else 'below'})
        200-Day MA: ${metrics['ma_200'].item():.2f} (Price is {metrics['price_vs_ma200'].item():.1f}% {'above' if metrics['price_vs_ma200'].item() > 0 else 'below'})
        Trend: {'Bullish' if metrics['price_vs_ma50'].item() > 0 and metrics['price_vs_ma100'].item() > 0 else 'Bearish' if metrics['price_vs_ma50'].item() < 0 and metrics['price_vs_ma100'].item() < 0 else 'Neutral'}
        """
    }


def initialize_session_state():
    """Initialize session state variables"""
    if 'llm_cache' not in st.session_state:
        st.session_state.llm_cache = {}
    if 'current_stock' not in st.session_state:
        st.session_state.current_stock = ''
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None


def validate_stock_input(stock):
    """Validate stock input and return cleaned symbol"""
    if not stock or stock.strip() == '':
        return None, "Please enter a valid stock symbol to get predictions."
    return stock.strip().upper(), None


def display_llm_sidebar_status(llm_service):
    """Display LLM connection status in sidebar"""
    st.sidebar.header("🤖 Local AI Status")
    if llm_service.check_connection():
        st.sidebar.success("✅ Local LLM Connected")
        available_models = llm_service.get_available_models()
        if available_models:
            st.sidebar.info(f"📊 Models Available: {len(available_models)}")
            st.sidebar.write("Models:", available_models[:3])
        else:
            st.sidebar.warning("⚠️ No models loaded in LM Studio")
    else:
        st.sidebar.error("❌ LM Studio Not Running")
        st.sidebar.info("💡 Start LM Studio on localhost:1234")


def display_market_info():
    """Display market information when no stock is selected"""
    st.warning('⚠️ Please enter a valid stock symbol to get predictions.')
    st.info('💡 **US NASDAQ:** GOOG (Google), AAPL (Apple), MSFT (Microsoft), TSLA (Tesla)')
    st.info('💡 **Indian NSE:** INFY.NS (Infosys), TCS.NS (TCS), RELIANCE.NS (Reliance)')
    st.info('💡 **Indian BSE:** 500209.BO (Infosys), 532540.BO (TCS), 500325.BO (Reliance)')
    st.markdown('---')
    st.markdown('''
    **📊 Supported Markets:**
    - **US Markets (NASDAQ)**: Use ticker symbol directly (e.g., AAPL, GOOGL)
    - **Indian NSE (National Stock Exchange)**: Add `.NS` suffix (e.g., INFY.NS)
    - **Indian BSE (Bombay Stock Exchange)**: Add `.BO` suffix (e.g., 500209.BO)
    ''')


def get_trend_info(price_vs_ma50, price_vs_ma100):
    """Get trend information based on moving averages"""
    # Convert to scalar if pandas Series
    ma50_val = price_vs_ma50.item() if hasattr(price_vs_ma50, 'item') else price_vs_ma50
    ma100_val = price_vs_ma100.item() if hasattr(price_vs_ma100, 'item') else price_vs_ma100
    
    if ma50_val > 0 and ma100_val > 0:
        return '📈', 'Bullish', 'green'
    elif ma50_val < 0 and ma100_val < 0:
        return '📉', 'Bearish', 'red'
    else:
        return '➡️', 'Neutral', 'orange'