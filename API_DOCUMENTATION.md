# 📚 API Documentation

> Complete API reference for Stockify - AI-Powered Stock Market Intelligence Platform

> To view the about project overview click [View Project README](README.md)

---

## 📑 Table of Contents

1. **[🔧 Services](#-services)**
   - [LocalLLMService](#localllmservice) - LM Studio integration
     - [Constructor](#-constructor)
     - [Methods](#️-methods)
   - [LMStudioEmbeddings](#lmstudioembeddings) - Text embeddings generation
     - [Constructor](#constructor)
     - [Methods](#methods)
     - [Factory Function](#factory-function)
   - [RAGPipeline](#ragpipeline) - Document Q&A pipeline
     - [Constructor](#constructor-1)
     - [Methods](#methods-1)

2. **[🛠️ Utilities](#️-utilities)**
   - [Stock Utilities](#stock-utilities) - Stock data & analysis functions
     - [download_stock_data()](#download_stock_datastock_symbol-start_date2012-01-01-end_date2022-12-31)
     - [load_prediction_model()](#load_prediction_model)
     - [init_local_llm()](#init_local_llm)
     - [prepare_prediction_data()](#prepare_prediction_datadata)
     - [make_predictions()](#make_predictionsmodel-data_test_scale-scaler)
     - [calculate_metrics()](#calculate_metricsdata)
     - [calculate_prediction_accuracy()](#calculate_prediction_accuracyy_actual-y_predicted)
     - [prepare_stock_data_for_llm()](#prepare_stock_data_for_llmstock_symbol-metrics)
     - [initialize_session_state()](#initialize_session_state)
     - [validate_stock_input()](#validate_stock_inputstock)
     - [display_llm_sidebar_status()](#display_llm_sidebar_statusllm_service)
     - [display_market_info()](#display_market_info)
     - [get_trend_info()](#get_trend_infoprice_vs_ma50-price_vs_ma100)
   - [PDF Utilities](#pdf-utilities) - PDF processing functions
     - [is_pdf_available()](#is_pdf_available---bool)
     - [get_text_splitter()](#get_text_splitterchunk_size-int--1000-chunk_overlap-int--200---recursivecharactertextsplitter)
     - [load_pdf_with_langchain()](#load_pdf_with_langchainfilepath-str---listlangchaindocument)
     - [split_documents()](#split_documentsdocuments-listlangchaindocument-chunk_size-int--1000-chunk_overlap-int--200---listlangchaindocument)

3. **[⚙️ Configuration](#️-configuration)**
   - [LLM Configuration](#llm-configuration) - Environment settings
     - [LM_STUDIO_CONFIG](#lm_studio_config)
     - [RECOMMENDED_MODELS](#recommended_models)

4. **[💻 Usage Examples](#-usage-examples)**
   - [Complete RAG Pipeline](#complete-rag-pipeline-example)
   - [Stock Analysis](#stock-analysis-example)

5. **[⚠️ Error Handling](#️-error-handling)**
   - [Connection Errors](#connection-errors)
   - [Stock Data Errors](#stock-data-errors)

6. **[⚡ Performance Considerations](#-performance-considerations)**
   - [💾 Caching](#-caching)
   - [📦 Batch Processing](#-batch-processing)
   - [🧠 Memory Management](#-memory-management)

7. **[📋 Version Information](#-version-information)**

8. **[💡 Support](#-support)**

---

## 🔧 Services

### LocalLLMService

**📦 Module**: `services.local_llm_service`  
**🎯 Purpose**: Interface with LM Studio for AI-powered financial analysis

#### 📝 Description

Service class for interfacing with LM Studio local LLM server. Provides methods for:
- ✅ Connection validation
- 🤖 Model management
- 📊 AI-powered financial analysis generation

#### 🏗️ Class: `LocalLLMService`

##### 🔨 Constructor

```python
LocalLLMService(base_url: str = "http://localhost:1234/v1")
```

**📥 Parameters**:
- `base_url` (str): Base URL for LM Studio API endpoint. Default: `"http://localhost:1234/v1"`

**💡 Example**:
```python
from services.local_llm_service import LocalLLMService

llm_service = LocalLLMService("http://localhost:1234/v1")
```

##### 🛠️ Methods

###### `check_connection() -> bool`

🔌 Check if LM Studio server is running and accessible.

**📤 Returns**:
- `bool`: `True` if server is reachable, `False` otherwise

**💡 Example**:
```python
if llm_service.check_connection():
    print("LM Studio is running")
```

---

###### `get_available_models() -> List[str]`

🤖 Get list of loaded models in LM Studio.

**📤 Returns**:
- `List[str]`: List of model identifiers currently loaded

**💡 Example**:
```python
models = llm_service.get_available_models()
print(f"Available models: {models}")
```

---

###### `analyze_stock_fundamentals_stream(stock_data: Dict) -> Generator`

📊 Analyze stock fundamentals using local LLM with streaming response.

**📥 Parameters**:
- `stock_data` (Dict): Dictionary containing:
  - `symbol` (str): Stock ticker symbol
  - `company_name` (str): Company name
  - `sector` (str): Business sector
  - `financial_data` (str): Financial metrics text
  - `technical_data` (str): Technical indicators text

**📤 Yields**:
- `str`: Chunks of AI-generated analysis text

**💡 Example**:
```python
stock_data = {
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "financial_data": "Latest Price: $150.00...",
    "technical_data": "50-Day MA: $145.00..."
}

for chunk in llm_service.analyze_stock_fundamentals_stream(stock_data):
    print(chunk, end='')
```

---

###### `generate_investment_summary_stream(technical_analysis: str, fundamental_analysis: str) -> Generator`

💼 Generate comprehensive investment summary with streaming response.

**📥 Parameters**:
- `technical_analysis` (str): Technical analysis text
- `fundamental_analysis` (str): Fundamental analysis text

**Yields**:
- `str`: Chunks of AI-generated investment summary text

**Example**:
```python
for chunk in llm_service.generate_investment_summary_stream(
    technical_analysis="Bullish trend observed...",
    fundamental_analysis="Strong fundamentals..."
):
    print(chunk, end='')
```

---

### LMStudioEmbeddings

**📦 Module**: `services.embeddings_service`  
**🎯 Purpose**: Generate text embeddings for semantic search and RAG

#### 📝 Description

Custom embeddings class for LM Studio that integrates with LangChain's Embeddings base class.

**Features**:
- 🔢 384-dimensional embeddings
- 🔄 Batch document processing
- 🔍 Query embedding for similarity search

#### Class: `LMStudioEmbeddings`

##### Constructor
```python
LMStudioEmbeddings(base_url: str = "http://127.0.0.1:1234")
```

**Parameters**:
- `base_url` (str): Base URL for LM Studio server. Default: `"http://127.0.0.1:1234"`

**Attributes**:
- `base_url` (str): LM Studio server URL
- `embeddings_endpoint` (str): Full endpoint URL for embeddings API
- `model_name` (str): Embedding model name (`"text-embedding-nomic-embed-text-v1.5"`)

**Example**:
```python
from services.embeddings_service import LMStudioEmbeddings

embeddings = LMStudioEmbeddings("http://127.0.0.1:1234")
```

##### Methods

###### `embed_documents(texts: List[str]) -> List[List[float]]`
Embed a list of documents using LM Studio's embedding API.

**Parameters**:
- `texts` (List[str]): List of text strings to embed

**Returns**:
- `List[List[float]]`: List of embedding vectors (384-dimensional)

**Example**:
```python
texts = ["First document", "Second document", "Third document"]
embeddings_list = embeddings.embed_documents(texts)
print(f"Generated {len(embeddings_list)} embeddings")
```

---

###### `embed_query(text: str) -> List[float]`
Embed a single query text for similarity search.

**Parameters**:
- `text` (str): Query text to embed

**Returns**:
- `List[float]`: Embedding vector (384-dimensional)

**Example**:
```python
query = "What is the revenue growth?"
query_embedding = embeddings.embed_query(query)
print(f"Query embedding dimension: {len(query_embedding)}")
```

##### Factory Function

###### `get_embeddings_model(base_url: str = "http://127.0.0.1:1234") -> LMStudioEmbeddings`
Factory function to create LM Studio embeddings model instance.

**Parameters**:
- `base_url` (str): Base URL for LM Studio server

**Returns**:
- `LMStudioEmbeddings`: Configured embeddings model

**Example**:
```python
from services.embeddings_service import get_embeddings_model

embeddings = get_embeddings_model()
```

---

### RAGPipeline

**📦 Module**: `services.rag_pipeline`  
**🎯 Purpose**: Document Q&A with retrieval-augmented generation

#### 📝 Description

Complete RAG pipeline using FAISS vector store and LLM.

**Workflow**:
1. 📝 Query embedding
2. 🔍 Similarity search
3. 📄 Context retrieval
4. 🤖 Answer generation with citations

#### Class: `RAGPipeline`

##### Constructor
```python
RAGPipeline(
    llm_service: LocalLLMService,
    vector_store: FAISS,
    top_k: int = 3
)
```

**Parameters**:
- `llm_service` (LocalLLMService): Configured LocalLLMService instance
- `vector_store` (FAISS): FAISS vector store containing document embeddings
- `top_k` (int): Number of top relevant documents to retrieve. Default: 3

**Example**:
```python
from services.rag_pipeline import RAGPipeline
from services.local_llm_service import LocalLLMService

llm_service = LocalLLMService()
# Assume vector_store is already created
rag = RAGPipeline(llm_service, vector_store, top_k=4)
```

##### Methods

###### `query(question: str, return_sources: bool = True) -> Dict`
Query the RAG pipeline and return answer with source documents.

**Parameters**:
- `question` (str): User's question about the document
- `return_sources` (bool): Whether to include source documents in response. Default: True

**Returns**:
- `Dict`: Dictionary containing:
  - `answer` (str): AI-generated response text
  - `sources` (List[Dict]): List of source documents with:
    - `content` (str): Snippet of source text
    - `metadata` (Dict): Document metadata (page, source, etc.)
    - `relevance_score` (float): Similarity score (0-1)

**Example**:
```python
result = rag.query("What was the total revenue in 2024?", return_sources=True)
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")

for i, source in enumerate(result['sources']):
    print(f"Source {i+1} (Relevance: {source['relevance_score']:.2%}):")
    print(source['content'])
```

---

## 🛠️ Utilities

### Stock Utilities

**📦 Module**: `utils.stock_utils`  
**🎯 Purpose**: Stock data fetching, analysis, and prediction functions

#### Functions

##### `download_stock_data(stock_symbol, start_date='2012-01-01', end_date=None)`
Download stock data from Yahoo Finance with caching.

**Parameters**:
- `stock_symbol` (str): Stock ticker symbol (e.g., 'AAPL', 'INFY.NS')
- `start_date` (str): Start date for historical data (YYYY-MM-DD). Default: '2012-01-01'
- `end_date` (str): End date for historical data (YYYY-MM-DD). Default: None (current system date)

**Returns**:
- `tuple`: (DataFrame with stock data, error message or None)

**Example**:
```python
from utils.stock_utils import download_stock_data
from datetime import datetime

# Download with dynamic end date (current date)
data, error = download_stock_data("AAPL", "2020-01-01")
if error:
    print(f"Error: {error}")
else:
    print(f"Downloaded {len(data)} rows")

# Or specify explicit end date
data, error = download_stock_data("AAPL", "2020-01-01", "2023-12-31")
```

---

##### `load_prediction_model()`

🧠 Load the pre-trained Keras LSTM model with caching.

**📤 Returns**:
- Keras Sequential model for stock price prediction

**Example**:
```python
from utils.stock_utils import load_prediction_model

model = load_prediction_model()
print(model.summary())
```

---

##### `init_local_llm()`

🤖 Initialize and cache the local LLM service connection.

**📤 Returns**:
- `LocalLLMService`: LocalLLMService instance connected to LM Studio

**Example**:
```python
from utils.stock_utils import init_local_llm

llm_service = init_local_llm()
if llm_service.check_connection():
    print("LLM connected successfully")
```

---

##### `prepare_prediction_data(data)`

📊 Prepare and scale stock data for LSTM model prediction.

**📥 Parameters**:
- `data` (DataFrame): DataFrame containing stock price data with 'Close' column

**Returns**:
- `tuple`: (scaled test data array, MinMaxScaler instance)

**Example**:
```python
from utils.stock_utils import prepare_prediction_data

data_test_scale, scaler = prepare_prediction_data(stock_data)
```

---

##### `make_predictions(model, data_test_scale, scaler)`

🔮 Generate stock price predictions using the trained LSTM model.

**📥 Parameters**:
- `model`: Trained Keras model
- `data_test_scale` (ndarray): Scaled test data array
- `scaler` (MinMaxScaler): MinMaxScaler instance for inverse transformation

**Returns**:
- `tuple`: (predicted prices array, actual prices array)

**Example**:
```python
from utils.stock_utils import make_predictions

predict, y = make_predictions(model, data_test_scale, scaler)
print(f"Predictions: {predict[:5]}")
```

---

##### `calculate_metrics(data)`

📈 Calculate key financial metrics and technical indicators.

**📥 Parameters**:
- `data` (DataFrame): DataFrame containing stock price data

**Returns**:
- `dict`: Dictionary containing:
  - `latest_price` (float): Most recent closing price
  - `price_change` (float): Daily price change percentage
  - `volatility` (float): 20-day volatility percentage
  - `volume_avg` (float): 20-day average volume
  - `ma_50` (float): 50-day moving average
  - `ma_100` (float): 100-day moving average
  - `ma_200` (float): 200-day moving average
  - `price_vs_ma50` (float): Price vs MA50 percentage
  - `price_vs_ma100` (float): Price vs MA100 percentage
  - `price_vs_ma200` (float): Price vs MA200 percentage

**Example**:
```python
from utils.stock_utils import calculate_metrics

metrics = calculate_metrics(stock_data)
print(f"Current Price: ${metrics['latest_price']:.2f}")
print(f"Volatility: {metrics['volatility']:.2f}%")
```

---

##### `calculate_prediction_accuracy(y_actual, y_predicted)`

🎯 Calculate model prediction accuracy using MAPE (Mean Absolute Percentage Error).

**📥 Parameters**:
- `y_actual` (ndarray): Array of actual stock prices
- `y_predicted` (ndarray): Array of predicted stock prices

**Returns**:
- `float`: Accuracy percentage (100 - MAPE)

**Example**:
```python
from utils.stock_utils import calculate_prediction_accuracy

accuracy = calculate_prediction_accuracy(actual_prices, predicted_prices)
print(f"Model Accuracy: {accuracy:.2f}%")
```

---

##### `prepare_stock_data_for_llm(stock_symbol, metrics)`

📝 Prepare stock data dictionary for LLM analysis.

**📥 Parameters**:
- `stock_symbol` (str): Stock ticker symbol
- `metrics` (dict): Dictionary containing calculated financial metrics

**Returns**:
- `dict`: Formatted data dictionary containing:
  - `symbol` (str): Stock symbol
  - `company_name` (str): Company name
  - `sector` (str): Business sector
  - `financial_data` (str): Formatted financial metrics
  - `technical_data` (str): Formatted technical indicators

**Example**:
```python
from utils.stock_utils import prepare_stock_data_for_llm

stock_data = prepare_stock_data_for_llm("AAPL", metrics)
```

---

##### `initialize_session_state()`
Initialize Streamlit session state variables for stock analysis.

**Example**:
```python
from utils.stock_utils import initialize_session_state

initialize_session_state()
```

---

##### `validate_stock_input(stock)`

✅ Validate and clean user stock symbol input.

**📥 Parameters**:
- `stock` (str): Raw stock symbol input from user

**Returns**:
- `tuple`: (cleaned stock symbol or None, error message or None)

**Example**:
```python
from utils.stock_utils import validate_stock_input

symbol, error = validate_stock_input("  aapl  ")
if error:
    print(error)
else:
    print(f"Valid symbol: {symbol}")  # "AAPL"
```

---

##### `display_llm_sidebar_status(llm_service)`
Display LLM connection status and available models in sidebar.

**Parameters**:
- `llm_service` (LocalLLMService): LocalLLMService instance

**Example**:
```python
from utils.stock_utils import display_llm_sidebar_status

display_llm_sidebar_status(llm_service)
```

---

##### `display_market_info()`
Display market information and supported stock exchanges.

**Example**:
```python
from utils.stock_utils import display_market_info

display_market_info()
```

---

##### `get_trend_info(price_vs_ma50, price_vs_ma100)`

📊 Determine market trend based on moving average positions.

**📥 Parameters**:
- `price_vs_ma50` (float): Price position relative to 50-day MA (percentage)
- `price_vs_ma100` (float): Price position relative to 100-day MA (percentage)

**Returns**:
- `tuple`: (emoji, trend_text, color)
  - Bullish: ('📈', 'Bullish', 'green')
  - Bearish: ('📉', 'Bearish', 'red')
  - Neutral: ('➡️', 'Neutral', 'orange')

**Example**:
```python
from utils.stock_utils import get_trend_info

emoji, trend, color = get_trend_info(2.5, 1.8)
print(f"{emoji} Trend: {trend}")  # "📈 Trend: Bullish"
```

---

##### `get_currency_symbol(stock_symbol)`

💱 Get currency symbol based on stock exchange.

**📥 Parameters**:
- `stock_symbol` (str): Stock ticker symbol (e.g., 'AAPL', 'INFY.NS')

**📤 Returns**:
- `str`: Currency symbol
  - '₹' for NSE (Indian stocks with .NS suffix)
  - '$' for NASDAQ and other exchanges

**Example**:
```python
from utils.stock_utils import get_currency_symbol

# US stock (NASDAQ)
currency = get_currency_symbol("AAPL")
print(f"Currency: {currency}")  # "$"

# Indian stock (NSE)
currency = get_currency_symbol("INFY.NS")
print(f"Currency: {currency}")  # "₹"

# Format price with currency
price = 150.50
symbol = "AAPL"
currency = get_currency_symbol(symbol)
print(f"Price: {currency}{price:.2f}")  # "Price: $150.50"
```

---

### PDF Utilities

**📦 Module**: `utils.pdf_utils`  
**🎯 Purpose**: PDF loading, parsing, and text chunking

#### 📋 Functions

##### `is_pdf_available() -> bool`

✅ Check if PDF processing support is available.

**📤 Returns**:
- `bool`: True if PyPDF2 is installed, False otherwise

**Example**:
```python
from utils.pdf_utils import is_pdf_available

if is_pdf_available():
    print("PDF processing is available")
else:
    print("Please install PyPDF2")
```

---

##### `get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter`

✂️ Get LangChain text splitter for optimal chunking.

**📥 Parameters**:
- `chunk_size` (int): Maximum size of each chunk. Default: 1000
- `chunk_overlap` (int): Number of characters to overlap between chunks. Default: 200

**Returns**:
- `RecursiveCharacterTextSplitter`: Configured text splitter for document processing

**Example**:
```python
from utils.pdf_utils import get_text_splitter

text_splitter = get_text_splitter(chunk_size=500, chunk_overlap=100)
```

---

##### `load_pdf_with_langchain(filepath: str) -> List[LangChainDocument]`

📄 Load PDF using LangChain's PyPDFLoader.

**📥 Parameters**:
- `filepath` (str): Absolute or relative path to the PDF file

**Returns**:
- `List[LangChainDocument]`: List of Document objects (one per page)

**Raises**:
- `ImportError`: If PyPDF2 is not installed
- `Exception`: If PDF loading or parsing fails

**Example**:
```python
from utils.pdf_utils import load_pdf_with_langchain

documents = load_pdf_with_langchain("./reports/financial_report.pdf")
print(f"Loaded {len(documents)} pages")
```

---

##### `split_documents(documents: List[LangChainDocument], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[LangChainDocument]`

✂️ Split documents into smaller chunks for embedding and retrieval.

**📥 Parameters**:
- `documents` (List[LangChainDocument]): List of LangChain Document objects to split
- `chunk_size` (int): Maximum size of each chunk in characters. Default: 1000
- `chunk_overlap` (int): Number of characters to overlap between chunks. Default: 200

**Returns**:
- `List[LangChainDocument]`: List of chunked Document objects with preserved metadata

**Example**:
```python
from utils.pdf_utils import load_pdf_with_langchain, split_documents

documents = load_pdf_with_langchain("./report.pdf")
chunks = split_documents(documents, chunk_size=800, chunk_overlap=150)
print(f"Created {len(chunks)} chunks from {len(documents)} pages")
```

---

## ⚙️ Configuration

### LLM Configuration

**📦 Module**: `config.llm_config`  
**🎯 Purpose**: LM Studio server settings and model configuration

#### Constants

##### `LM_STUDIO_CONFIG`
Dictionary containing LM Studio configuration settings.

**Structure**:
```python
{
    "base_url": str,      # LM Studio server URL
    "api_key": str,       # API key (not required for LM Studio)
    "model_name": str,    # Default model name
    "max_tokens": int,    # Maximum tokens for generation
    "temperature": float, # Sampling temperature
    "timeout": int        # Request timeout in seconds
}
```

**Environment Variables**:
- `LM_STUDIO_URL`: Override base URL (default: "http://localhost:1234/v1")
- `LM_STUDIO_MAX_TOKENS`: Override max tokens (default: 2048)
- `LM_STUDIO_TEMPERATURE`: Override temperature (default: 0.3)
- `LM_STUDIO_TIMEOUT`: Override timeout (default: 30)

**Example**:
```python
from config.llm_config import LM_STUDIO_CONFIG

print(f"LM Studio URL: {LM_STUDIO_CONFIG['base_url']}")
print(f"Max Tokens: {LM_STUDIO_CONFIG['max_tokens']}")
```

---

##### `RECOMMENDED_MODELS`
List of recommended models for financial analysis.

**Value**:
```python
["google/gemma-3-4b"]
```

**Example**:
```python
from config.llm_config import RECOMMENDED_MODELS

print(f"Recommended models: {RECOMMENDED_MODELS}")
```

---

## 💻 Usage Examples

### Complete RAG Pipeline Example

**Scenario**: Load a financial PDF and answer questions using RAG

```python
from services.local_llm_service import LocalLLMService
from services.embeddings_service import get_embeddings_model
from services.rag_pipeline import RAGPipeline
from utils.pdf_utils import load_pdf_with_langchain, split_documents
from langchain_community.vectorstores import FAISS

# Initialize services
llm_service = LocalLLMService()
embeddings = get_embeddings_model()

# Load and process PDF
documents = load_pdf_with_langchain("./financial_report.pdf")
chunks = split_documents(documents)

# Create vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# Initialize RAG pipeline
rag = RAGPipeline(llm_service, vector_store, top_k=3)

# Query the document
result = rag.query("What was the revenue growth in Q4?")
print(f"Answer: {result['answer']}")
```

### Stock Analysis Example

**Scenario**: Download stock data, make predictions, and get AI analysis

```python
from utils.stock_utils import (
    download_stock_data,
    load_prediction_model,
    prepare_prediction_data,
    make_predictions,
    calculate_metrics,
    calculate_prediction_accuracy,
    init_local_llm,
    prepare_stock_data_for_llm
)
from datetime import datetime

# Download stock data (uses current date by default)
data, error = download_stock_data("AAPL", "2020-01-01")

# Load model and make predictions
model = load_prediction_model()
data_test_scale, scaler = prepare_prediction_data(data)
predict, y = make_predictions(model, data_test_scale, scaler)

# Calculate metrics
metrics = calculate_metrics(data)
accuracy = calculate_prediction_accuracy(y, predict)

# Get LLM analysis
llm_service = init_local_llm()
stock_data = prepare_stock_data_for_llm("AAPL", metrics)

for chunk in llm_service.analyze_stock_fundamentals_stream(stock_data):
    print(chunk, end='')
```

---

## ⚠️ Error Handling

### Common Exceptions

#### Connection Errors
```python
from services.local_llm_service import LocalLLMService

llm_service = LocalLLMService()
if not llm_service.check_connection():
    print("Error: LM Studio is not running")
    print("Please start LM Studio server at http://localhost:1234")
```

#### Stock Data Errors
```python
from utils.stock_utils import download_stock_data

data, error = download_stock_data("INVALID_SYMBOL")
if error:
    print(f"Error downloading stock data: {error}")
```

---

## ⚡ Performance Considerations

### 💾 Caching
The following functions use Streamlit caching for improved performance:
- `download_stock_data()` - Uses `@st.cache_data`
- `load_prediction_model()` - Uses `@st.cache_resource`
- `init_local_llm()` - Uses `@st.cache_resource`

### 📦 Batch Processing

For better performance when processing multiple documents:
```python
# Embed multiple documents in batch
texts = ["doc1", "doc2", "doc3", ...]
embeddings_list = embeddings.embed_documents(texts)
```

### 🧠 Memory Management

When processing large PDFs, consider adjusting chunk sizes:
```python
# Smaller chunks for better memory usage
chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
```

---

## 📋 Version Information

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| LangChain Community | Latest |
| FAISS | CPU version |
| Keras | 3.x compatible |
| Streamlit | Latest |
| TensorFlow | Latest |

---

## 💡 Support

**Troubleshooting Checklist**:
1. ✅ Check if LM Studio is running at `http://localhost:1234`
2. ✅ Verify all dependencies are installed: `pip install -r requirements.txt`
3. ✅ Ensure embedding model is loaded in LM Studio
4. ✅ Check API endpoint availability

**Need Help?**  
- 📩 Email: [spreveen123@gmail.com](mailto:spreveen123@gmail.com)
- 📖 [View Project README](README.md)

