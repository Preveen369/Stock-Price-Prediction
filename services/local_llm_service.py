"""
Local LLM Service - LM Studio Integration
Handles communication with local LLM server for financial analysis
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import requests
import json
from typing import Dict, List, Optional

class LocalLLMService:
    """
    Service class for interfacing with LM Studio local LLM server
    
    Provides methods for:
    - Connection validation
    - Model management
    - Stock analysis generation
    - Investment summary generation
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        """
        Initialize Local LLM Service with LM Studio
        
        Args:
            base_url: Base URL for LM Studio API endpoint
        """
        self.base_url = base_url
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """
        Initialize the ChatOpenAI client with error handling
        
        Attempts to connect to LM Studio and configure the LangChain client
        with appropriate model settings and timeouts.
        """
        try:
            # Get the first available model if possible
            available_models = self.get_available_models()
            model_name = available_models[0] if available_models else "local-model"
            
            self.client = ChatOpenAI(
                base_url=self.base_url,
                api_key="lm-studio",  # LM Studio doesn't validate this
                model=model_name,
                temperature=0.3,
                max_tokens=1500,  # Reduced for faster responses
                timeout=60,       # Increased timeout
                request_timeout=60  # Add explicit request timeout
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            # Create a basic client anyway
            self.client = ChatOpenAI(
                base_url=self.base_url,
                api_key="lm-studio",
                model="local-model",
                temperature=0.3,
                max_tokens=1500,  # Reduced for faster responses
                timeout=60,       # Increased timeout
                request_timeout=60  # Add explicit request timeout
            )
        
    def check_connection(self) -> bool:
        """
        Check if LM Studio server is running and accessible
        
        Returns:
            bool: True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of loaded models in LM Studio
        
        Returns:
            List[str]: List of model identifiers currently loaded
        """
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                return [model['id'] for model in models.get('data', [])]
            return []
        except:
            return []
    
    def analyze_stock_fundamentals_stream(self, stock_data: Dict):
        """
        Analyze stock fundamentals using local LLM with streaming response
        
        Args:
            stock_data: Dictionary containing stock symbol, company info, and metrics
            
        Yields:
            str: Chunks of AI-generated analysis text
        """
        
        prompt = PromptTemplate(
            input_variables=["stock_symbol", "company_name", "sector", "financial_data", "technical_data"],
            template="""
You are a Financial AI Analyst. Your task is to analyze a stock using:

1. The AI model's internal predictive metrics
2. Verified fundamental & technical data typically available on:
   - Screener.in (Indian markets)
   - NASDAQ.com (US markets)
   - Yahoo Finance (global markets)

Always produce:
- Accurate, concise, actionable insights
- No unnecessary narrative or generic explanations
- No fabricated numbers; only interpret the data provided in inputs

---

### STOCK DETAILS:
Symbol: {stock_symbol}
Company: {company_name}
Sector: {sector}

### MODEL OUTPUTS:
Financial Data (Fundamentals): {financial_data}
Technical Indicators: {technical_data}

---

### Generate the following sections:

1. **Current Market Position (2–3 sentences)**  
   Provide a clear snapshot of the stock’s positioning based on fundamentals, technical trend, sector context, and validation cues from typical data found on Screener.in / NASDAQ / Yahoo Finance.

2. **Investment Recommendation (BUY / HOLD / SELL)**  
   Give a single recommendation with 2–3 precise reasons:  
   - Momentum trend (bullish/bearish/neutral)  
   - Valuation insights (e.g., whether P/E, P/B, ROE appear strong/weak based on given data)  
   - Financial strength indicators  

3. **Key Financial Insights (3 bullet points)**  
   Include:  
   - Fundamental strength/weakness  
   - Most important technical signal  
   - Major risk or opportunity visible  

4. **Price Direction & Timeline**  
   Give a short-term (1–3 months) directional outlook (uptrend/downtrend/sideways) based on:  
   - Technical setup  
   - Expected fundamental catalysts  
   - Market sentiment  

5. **Risk Assessment (2–3 bullets)**  
   Identify the primary risks relevant to this stock (company-level, market-level, or sector-level).

---

### VALIDATION GUIDANCE (DO NOT FABRICATE):  
Base your interpretation on typical data available from:  
- Screener.in → ratios, earnings, peer comparison  
- NASDAQ → valuations, analyst expectations  
- Yahoo Finance → price trend, sentiment, news  

Only interpret; do not generate fictional numerical data.

---
"""
        )
        
        chain = prompt | self.client
        
        try:
            if not self.client:
                yield "Error: LLM service not properly initialized. Please check LM Studio connection."
                return
            
            # Stream the response
            for chunk in chain.stream({
                "stock_symbol": stock_data.get("symbol", ""),
                "company_name": stock_data.get("company_name", ""),
                "sector": stock_data.get("sector", ""),
                "financial_data": stock_data.get("financial_data", ""),
                "technical_data": stock_data.get("technical_data", "")
            }):
                yield chunk.content
                
        except Exception as e:
            yield f"Error generating analysis: {str(e)}. Please ensure LM Studio is running with a model loaded."
    
    def generate_investment_summary_stream(self, technical_analysis: str, fundamental_analysis: str):
        """
        Generate comprehensive investment summary with streaming response
        
        Args:
            technical_analysis: Technical analysis text from previous analysis
            fundamental_analysis: Fundamental analysis text from previous analysis
            
        Yields:
            str: Chunks of AI-generated investment summary text
        """
        
        prompt = PromptTemplate(
            input_variables=["technical_analysis", "fundamental_analysis"],
            template="""
You are an Investment Strategy Analyst. Combine the model's technical and fundamental outputs with insights that investors typically confirm on:

- Screener.in (India fundamentals)
- NASDAQ.com (US valuations & earnings)
- Yahoo Finance (market sentiment & trend)

Your summary must be:
- Concise but comprehensive
- Action-oriented
- Free from fabricated metrics
- Suitable for retail investors and portfolio managers

---

### MODEL OUTPUT:
Technical Analysis: {technical_analysis}
Fundamental Analysis: {fundamental_analysis}

---

### Produce the following:

1. **Executive Summary (3–4 sentences)**  
   Provide the combined investment thesis:  
   - Current stock outlook  
   - Strength of fundamentals vs technicals  
   - Risk-adjusted return potential  
   - How it positions relative to its sector  

2. **Consolidated Investment Rating**  
   Give one rating → **BUY / HOLD / SELL**  
   Include:  
   - Confidence level (High / Medium / Low)  
   - Key supporting evidence from both analyses  
   - Comparison to typical benchmark behavior (Nifty/S&P trend characteristics; no fabricated numbers)

3. **Risk–Return Analysis**  
   Describe:  
   - Expected upside/downside scenarios (qualitative; no made-up '%' unless provided)  
   - Key market/sector/company risks  
   - Volatility or trend behavior as inferred from technical analysis  

4. **Investment Horizon Strategy**  
   - **Short-term (1–3 months):** Trading view  
   - **Medium-term (3–12 months):** Portfolio positioning  
   - **Long-term (1+ years):** Allocation guidance  

5. **Action Plan with Metrics**  
   Provide:  
   - Suggested entry zone  
   - Stop-loss zone  
   - Take-profit zone  
   - Ideal portfolio weight (low/moderate/high conviction)  

   *Do not invent specific numeric values unless provided. Present ranges qualitatively.*

---

### VALIDATION NOTE  
Your interpretations should align with the type of data available on:  
- Screener.in (fundamentals, ratios, peers)  
- NASDAQ.com (analyst view, earnings sentiment)  
- Yahoo Finance (charts, news sentiment)  

---
"""
        )
        
        chain = prompt | self.client
        
        try:
            for chunk in chain.stream({
                "technical_analysis": technical_analysis,
                "fundamental_analysis": fundamental_analysis
            }):
                yield chunk.content
        except Exception as e:
            yield f"Error generating summary: {str(e)}"