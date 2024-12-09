from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from quantum_options_analyzer import EnhancedQuantumOptionsAnalyzer, RiskParameters

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
risk_params = RiskParameters()
analyzer = EnhancedQuantumOptionsAnalyzer(risk_params=risk_params)

class MarketDataResponse(BaseModel):
    symbol: str
    name: str
    lastPrice: float
    dailyChange: float
    environment: dict
    historicalPrices: List[dict]

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    try:
        # Fetch market data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Calculate daily change
        daily_change = ((hist['Close'][-1] - hist['Close'][-2]) / hist['Close'][-2]) * 100
        
        # Analyze market environment
        environment = analyzer.analyze_market_environment(hist)
        
        # Format historical prices
        historical_prices = []
        for date, row in hist.iterrows():
            historical_prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": row['Volume']
            })
        
        return MarketDataResponse(
            symbol=symbol,
            name=ticker.info.get('longName', symbol),
            lastPrice=hist['Close'][-1],
            dailyChange=daily_change,
            environment=environment,
            historicalPrices=historical_prices
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indices")
async def get_available_indices():
    indices = [
        {"symbol": "^GSPC", "name": "S&P 500"},
        {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
        {"symbol": "^IXIC", "name": "NASDAQ Composite"},
        {"symbol": "^RUT", "name": "Russell 2000"},
        {"symbol": "^VIX", "name": "CBOE Volatility Index"}
    ]
    return indices

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)