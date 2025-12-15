# 🚀 Stockify - AI-Powered Stock Market Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Web-UI](https://img.shields.io/badge/Web_UI-Streamlit-FF4B4B.svg)
![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)
![Deep-Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow/Keras-orange.svg)
![LLM](https://img.shields.io/badge/LLM-LM_Studio-purple.svg)
![Vector-DB](https://img.shields.io/badge/VectorDB-FAISS-emerald.svg)
![Notebook](https://img.shields.io/badge/Notebook-Google_Colab-F9AB00.svg)


**Stockify** is a comprehensive AI-powered stock market intelligence platform that combines advanced machine learning, natural language processing, and document analysis for complete investment research. Built with privacy as a core principle, it leverages:

- 🧠 **LSTM Neural Networks** for accurate 50/100/200-day price predictions
- 🤖 **Local LLM Integration** via LM Studio for on-device AI processing
- 📚 **RAG Technology** for intelligent document analysis and Q&A
- 📊 **Technical Analysis Tools** for in-depth market insights

Whether you're analyzing US stocks (NASDAQ) or Indian markets (NSE), Stockify provides AI-powered recommendations, real-time data visualization, and document-based research capabilities—all while keeping your data completely private on your local machine.

---

## ✨ Features & Pages

### 1. 🏠 Main Dashboard - Stock Price Predictions
**Core Features:**
- ✅ **LSTM Price Predictions** - 50/100/200-day forecasts trained on 2012-2022 market data
- ✅ **Real-time Stock Data** - Live market data analytics from Yahoo Finance
- ✅ **Historical Charts** - Interactive price visualization and trend analysis
- ✅ **AI Market Insights** - Automated analysis of stock performance

### 2. 📄 Financial Report Analysis - RAG-Powered Q&A
**AI RAG Features:**
- ✅ **Document Q&A** - Ask questions about financial PDFs in natural language
- ✅ **Source Citations** - Answers grounded in actual document content
- ✅ **Local LLM Processing** - Privacy-first AI analysis (no data leaves your machine)
- ✅ **Vector Embeddings** - FAISS-powered semantic search for accurate retrieval

### 3. 📈 Investment Summary - AI Recommendations
**Investment Features:**
- ✅ **BUY/HOLD/SELL Signals** - Intelligent recommendations with reasoning
- ✅ **Risk Assessment** - Comprehensive risk analysis and evaluation
- ✅ **Entry/Exit Points** - Strategic price targets for trading
- ✅ **Streaming Responses** - Real-time AI response generation

### 4. 📋 Stock Tickers List - Market Browse
**Data Features:**
- ✅ **Multi-Market Support** - 1000+ NASDAQ and 1000+ NSE stocks
- ✅ **Search & Filter** - Easy stock discovery and selection
- ✅ **Comprehensive Listings** - Full stock metadata and information

### 5. 📊 Technical Analysis - Indicators & Trends
**Technical Features:**
- ✅ **Moving Averages** - 50-day, 100-day and 200-day MA calculations
- ✅ **Volatility Analysis** - Price fluctuation and risk metrics
- ✅ **Trend Detection** - Automated pattern recognition
- ✅ **AI Interpretation** - Natural language explanations of technical data

---

## 🏗️ Architecture

```
Streamlit UI → Services (LocalLLM, Embeddings, RAG) → Utils (Stock LSTM, PDF)
```

**Tech Stack**: Streamlit • Python • TensorFlow/Keras • LangChain • FAISS • LM Studio • yfinance • PyPDF2 • Pandas • NumPy • Google Colab

---

## 💻 Installation

**Prerequisites**: Python 3.8+, Google Colab, LM Studio, 8GB+ RAM

```bash
# Clone and setup
git clone https://github.com/Preveen369/Stock-Price-Prediction.git
cd Stock-Price-Prediction
python -m venv venv
venv\Scripts\activate  # Windows 

# Install dependencies
pip install -r requirements.txt
pip install -r rag_requirements.txt

# Setup LM Studio (lmstudio.ai)
# Load models: gemma-3-4b + text-embedding-nomic-embed-text-v1.5
# Start server: http://localhost:1234

# Run application
streamlit run app.py  # Opens at http://localhost:8501
```

---

## ⚙️ Configuration

- **Environment (.env)**: `LM_STUDIO_URL`, `LM_STUDIO_MAX_TOKENS`, `LM_STUDIO_TEMPERATURE`, `LM_STUDIO_TIMEOUT`

- **LM Studio**: Load `gemma-3-4b` (chat) + `nomic-embed-text-v1.5` (embeddings) → Start server at `http://localhost:1234`

- **Settings**: See `config/llm_config.py` for defaults (max_tokens: 2048, temperature: 0.3)

---

## 🚦 Usage

### Quick Start Guide

```bash
# 1. Ensure LM Studio is running with models loaded
# 2. Start Stockify
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

### Workflow Examples

- **Analyze Stock**: Main Dashboard → Enter Symbol (AAPL) → View Charts & Predictions 
- **NASDAQ | NSE Stock Symbols**: US (AAPL, TSLA) | India (INFY.NS, TCS.NS) 
- **Technical Analysis**: Technical Analysis Page → View Moving Averages & Trends  
- **Investment Advice**: Investment Summary → Risk Assessment → AI Recommendations  
- **Document Q&A**: Financial Report → Upload PDF → Ask Questions → Get Answers

---

## 📚 API Documentation

For detailed API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### Key Modules

#### Services
Core services layer that handles AI/ML operations and external integrations.

- `LocalLLMService` - LM Studio integration for AI analysis
- `LMStudioEmbeddings` - Text embedding generation
- `RAGPipeline` - Retrieval-augmented generation pipeline

#### Utilities
Helper functions for data processing, calculations, and file operations.

- `stock_utils.py` - Stock data fetching and analysis
- `pdf_utils.py` - PDF processing and chunking

#### Example Usage
```python
from services.local_llm_service import LocalLLMService
from utils.stock_utils import download_stock_data, calculate_metrics

# Initialize LLM
llm = LocalLLMService()

# Get stock data
data, error = download_stock_data("AAPL")

# Calculate metrics
metrics = calculate_metrics(data)

# Get AI analysis
for chunk in llm.analyze_stock_fundamentals_stream(stock_data):
    print(chunk, end='')
```

---

## 📁 Project Structure

```
Stock-Price-Prediction/
├── 📄 app.py                            # Main dashboard application
├── 📁 pages/                        # Streamlit pages
│   ├── 📄 Financial_Report_Analysis.py  # RAG-based PDF analysis
│   ├── 📄 Investment_Summary.py         # AI investment recommendations
│   ├── 📄 Stock_Tickers_List.py         # Browse stock symbols
│   └── 📄 Technical_Analysis.py         # Technical indicators analysis
├── 📁 services/                     # Core services layer
│   ├── 📄 local_llm_service.py          # LM Studio integration
│   ├── 📄 embeddings_service.py         # Vector embedding generation
│   └── 📄 rag_pipeline.py               # RAG implementation
├── 📁 utils/                        # Utility functions
│   ├── 📄 stock_utils.py                # Stock data and metrics
│   └── 📄 pdf_utils.py                  # PDF processing utilities
├── 📁 config/                       # Configuration
│   └── 📄 llm_config.py                 # LLM settings and env vars
├── 📁 models/                       # Pre-trained models
│   └── 📄Stock Predictions Model.keras  # LSTM neural network
├── 📁 resources/                    # Static resources
│   ├── 📄 NASDAQ_Active_Stocks_List.csv # US stock listings
│   └── 📄 NSE_Active_Stocks_List.csv    # Indian stock listings
├── 📄 requirements.txt                  # Core dependencies
├── 📄 API_DOCUMENTATION.md              # Complete API reference
└── 📄 README.md                         # This file
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| **app.py** | Main dashboard, stock selection, LSTM predictions |
| **services/** | LLM integration, embeddings, RAG pipeline |
| **utils/** | Stock data fetching, calculations, PDF processing |
| **pages/** | Individual analysis pages (UI components) |
| **config/** | Environment variables, LLM configuration |
| **models/** | Pre-trained LSTM model for predictions |

---

## 🧠 Key Concepts Explained

### 1. LSTM (Long Short-Term Memory)
- **What**: Neural network architecture for time-series prediction
- **How**: Uses 100 days of historical prices to predict next 50/100/200 days
- **Framework**: Built with TensorFlow/Keras for deep learning
- **Preprocessing**: Scikit-learn for data scaling and normalization
- **Accuracy**: Typically achieves 75-90% accuracy on test data
- **Training**: Pre-trained on 2012-2022 historical data across thousands of stock patterns

### 2. RAG (Retrieval-Augmented Generation)
```
Document → Chunks → Embeddings → Vector Store
                                        ↓
Question → Embedding → Similarity Search → Retrieved Chunks
                                        ↓
                        Chunks + Question → LLM → Answer
```
- **Benefit**: Accurate answers grounded in document content
- **Advantage**: No hallucinations, includes source citations

### 3. Vector Embeddings
- **What**: Numerical representations of text (384 dimensions)
- **Purpose**: Enable semantic similarity search
- **Model**: Nomic Embed Text v1.5
- **Use Case**: Finding relevant document sections for RAG

### 4. Local LLM Processing
- **Privacy**: All AI processing happens on your machine
- **Speed**: Depends on your hardware (GPU recommended)
- **Models**: Gemma 3, Mistral, Llama, etc.
- **Benefits**: No API costs, full control, offline capable

---

## 🔧 Key Technologies

- **LSTM**: 100-day history → 50/100/200-day predictions (75-90% accuracy) 
- **RAG**: Document chunks + embeddings + vector search → cited answers  
- **Embeddings**: 384D vectors (Nomic v1.5) 
- **Local LLM**: Privacy-first AI (Gemma/Mistral/Llama)

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### ❌ LM Studio Not Connected
**Symptoms**: "LM Studio Not Connected" error in sidebar,

**Solutions**:
1. Verify LM Studio is running
2. Check server URL is `http://127.0.0.1:1234`


#### ❌ Slow Performance
**Symptoms**: Long processing times, lag

**Solutions**:
1. Use smaller chunk sizes for PDFs (500 instead of 1000)
2. Reduce `top_k` in RAG queries (3 instead of 4)
3. Use lighter LLM models (4B instead of 7B)
4. Close other applications
5. Enable GPU acceleration in LM Studio


#### ❌ Memory Errors
**Symptoms**: Out of memory, crashes

**Solutions**:
1. Use quantized models (Q4 or Q5)
2. Reduce max_tokens in config (1024 instead of 2048)

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repository and suggest improvements.

Steps to contribute:

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature-name
# 3. Commit your changes
git commit -m "Add feature description"
# 4. Push to GitHub
git push origin feature-name
# 5. Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

*For educational purposes only. NOT financial advice. Past performance ≠ future results. AI predictions may be inaccurate. Consult qualified advisors. Use at your own risk. No warranties or liability.*

---

## 📧 Contact
For queries or suggestions:
- 📩 Email: [spreveen123@gmail.com](mailto:spreveen123@gmail.com)
- 🌐 LinkedIn: [www.linkedin.com/in/preveen-s](https://www.linkedin.com/in/preveen-s/)

---

## 🌟 Show Your Support
If you like this project, please consider giving it a ⭐ on GitHub!

---

**Built with ❤️ using DL/ML LSTM + Local AI LLMs + RAG**
