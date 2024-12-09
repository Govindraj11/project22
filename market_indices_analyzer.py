import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
import pytz
from typing import Dict, List, Optional, Any
from quantum_options_analyzer import EnhancedQuantumOptionsAnalyzer, RiskParameters
import requests
import argparse
import sys
import asyncio
from time import sleep
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API key from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Market indices using ETFs and supported symbols
INDIAN_INDICES = {
    'NIFTYBEES.NS': 'NIFTY 50 ETF',          # Nippon India ETF Nifty BeES
    'SETFNIF50.NS': 'SBI NIFTY 50 ETF',      # SBI Nifty 50 ETF
    'BANKBEES.NS': 'NIFTY Bank ETF',         # Nippon India ETF Bank BeES
    'SETFBANK.NS': 'SBI NIFTY Bank ETF',     # SBI Banking ETF
    'SETFNIFBK.NS': 'Nifty Bank ETF'         # SBI Nifty Bank ETF
}

US_INDICES = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'NASDAQ 100 ETF',
    'IWM': 'Russell 2000 ETF',
    'DIA': 'Dow Jones ETF',
    'UVXY': 'VIX ETF'
}

def get_market_hours(current_time=None):
    """Get Indian market hours status"""
    ist = pytz.timezone('Asia/Kolkata')
    if current_time is None:
        current_time = datetime.now(ist)
    elif not current_time.tzinfo:
        current_time = ist.localize(current_time)
        
    current_time = current_time.astimezone(ist)
    
    # Market hours
    market_open = time(9, 15)  # 9:15 AM IST
    market_close = time(15, 30)  # 3:30 PM IST
    
    # Check if it's a weekday
    if current_time.weekday() >= 5:  # Saturday or Sunday
        next_open = current_time.replace(hour=market_open.hour, minute=market_open.minute)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
    else:
        current_time_only = current_time.time()
        if current_time_only < market_open:
            next_open = current_time
        elif current_time_only >= market_close:
            next_open = current_time + timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
        else:
            next_open = current_time
        next_open = next_open.replace(hour=market_open.hour, minute=market_open.minute)
    
    # Determine market status
    is_open = (
        current_time.weekday() < 5 and  # Weekday
        market_open <= current_time.time() < market_close
    )
    
    return {
        'status': 'OPEN' if is_open else 'CLOSED',
        'trading_hours': '09:15 - 15:30 IST',
        'next_open': next_open.strftime('%Y-%m-%d %H:%M IST')
    }

def get_us_market_hours(current_time=None):
    """Get US market hours status"""
    est = pytz.timezone('America/New_York')
    if current_time is None:
        current_time = datetime.now(est)
    elif not current_time.tzinfo:
        current_time = est.localize(current_time)
        
    current_time = current_time.astimezone(est)
    
    # Market hours
    market_open = time(9, 30)  # 9:30 AM EST
    market_close = time(16, 0)  # 4:00 PM EST
    
    # Check if it's a weekday
    if current_time.weekday() >= 5:  # Saturday or Sunday
        next_open = current_time.replace(hour=market_open.hour, minute=market_open.minute)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
    else:
        current_time_only = current_time.time()
        if current_time_only < market_open:
            next_open = current_time
        elif current_time_only >= market_close:
            next_open = current_time + timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
        else:
            next_open = current_time
        next_open = next_open.replace(hour=market_open.hour, minute=market_open.minute)
    
    # Determine market status
    is_open = (
        current_time.weekday() < 5 and  # Weekday
        market_open <= current_time.time() < market_close
    )
    
    return {
        'status': 'OPEN' if is_open else 'CLOSED',
        'trading_hours': '09:30 - 16:00 EST',
        'next_open': next_open.strftime('%Y-%m-%d %H:%M EST')
    }

async def fetch_market_data(symbol: str, interval: str = '5min') -> Optional[pd.DataFrame]:
    """
    Fetch market data using Alpha Vantage API with enhanced error handling
    """
    base_url = "https://www.alphavantage.co/query"
    
    # Parameters for the API request
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    
    try:
        # Make the API request with timeout
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        data = response.json()
        
        # Check if we hit API limits
        if "Note" in data:
            logger.error(f"API Limit reached: {data['Note']}")
            return None
            
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API Error for {symbol}: {data['Error Message']}")
            return None
            
        # Get the correct time series key
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.error(f"No {interval} data available for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Check if DataFrame is empty
        if df.empty:
            logger.error(f"Empty data received for {symbol}")
            return None
            
        df.index = pd.to_datetime(df.index)
        
        # Rename columns (remove number prefix)
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to float
        df = df.astype(float)
        
        logger.info(f"Successfully fetched data for {symbol}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {str(e)}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Data parsing error for {symbol}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
        return None

async def analyze_index(symbol: str, name: str, analyzer: EnhancedQuantumOptionsAnalyzer) -> Optional[Dict[str, Any]]:
    """Analyze a single market index"""
    try:
        # Fetch data with retry logic
        retries = 3
        data = None
        
        for _ in range(retries):
            data = await fetch_market_data(symbol)
            if data is not None and not data.empty:
                break
        
        if data is None or data.empty:
            logger.error(f"No data available for {name} ({symbol})")
            return None

        # Calculate daily change
        daily_change = ((data['Close'][-1] - data['Close'][0]) / data['Close'][0]) * 100

        # Analyze market environment
        environment = analyzer.analyze_market_environment(data)
        
        # Get market regime
        regime = {
            'current_regime': environment['market_condition'],
            'volatility': environment['volatility_regime'],
            'trend': environment['trend']
        }

        # Calculate sentiment
        sentiment = {
            'technical': analyzer.calculate_technical_sentiment(data),
            'composite': analyzer.calculate_composite_sentiment(data)
        }

        # Extract features
        features = {
            'volatility': data['Close'].pct_change().std() * np.sqrt(252) * 100,
            'volume': data['Volume'][-1] if 'Volume' in data else 0,
            'momentum': data['Close'].pct_change(5).mean() * 100 if len(data) >= 6 else 0
        }

        return {
            'symbol': symbol,
            'name': name,
            'last_price': data['Close'][-1],
            'daily_change': daily_change,
            'environment': environment,
            'regime': regime,
            'sentiment': sentiment,
            'features': features
        }

    except Exception as e:
        logger.error(f"Error analyzing {name} ({symbol}): {str(e)}")
        return None

async def analyze_market(symbols: Dict[str, str], market_name: str) -> None:
    """Analyze the specified market indices"""
    try:
        logger.info(f"\n{'=' * 20} {market_name} Market Analysis {'=' * 20}")
        
        # Get market hours
        if market_name == "INDIA":
            market_hours = get_market_hours()
        else:
            market_hours = get_us_market_hours()
        
        logger.info(f"\nMarket Status: {market_hours['status']}")
        logger.info(f"Trading Hours: {market_hours['trading_hours']}")
        logger.info(f"Next Market Open: {market_hours['next_open']}")
        
        # Initialize analyzer
        risk_params = RiskParameters()
        analyzer = EnhancedQuantumOptionsAnalyzer(n_qubits=20, risk_params=risk_params)
        
        # Analyze each index
        results = []
        for symbol, name in symbols.items():
            result = await analyze_index(symbol, name, analyzer)
            if result:
                results.append(result)
                
                # Print result immediately
                logger.info(f"\n{'-' * 50}")
                logger.info(f"{name} ({symbol}):")
                logger.info(f"Last Price: {result['last_price']:.2f}")
                logger.info(f"Daily Change: {result['daily_change']:.2f}%")
                
                # Market condition and strength
                logger.info(f"\nMarket Condition: {result['environment']['market_condition']} ({result['environment']['strength']})")
                logger.info(f"Current Trend: {result['environment']['trend']}")
                logger.info(f"Volatility Regime: {result['environment']['volatility_regime']}")
                
                # Detailed signals
                signals = result['environment']['detailed_signals']
                
                logger.info("\nTechnical Signals:")
                for signal in signals['trend_signals']:
                    logger.info(f"- Trend: {signal}")
                for signal in signals['macd_signals']:
                    logger.info(f"- MACD: {signal}")
                for signal in signals['rsi_signals']:
                    logger.info(f"- RSI: {signal}")
                for signal in signals['volume_signals']:
                    logger.info(f"- Volume: {signal}")
                
                logger.info(f"- Volatility: {signals['volatility']}")
                logger.info(f"- Momentum: {signals['momentum']}")
                
                # Sentiment scores
                logger.info(f"\nSentiment Analysis:")
                logger.info(f"Technical Sentiment: {result['sentiment']['technical']:.2f}")
                logger.info(f"Composite Sentiment: {result['sentiment']['composite']:.2f}")
                
                # Key metrics
                logger.info(f"\nKey Metrics:")
                logger.info(f"Volatility (Annualized): {result['features']['volatility']:.2f}%")
                logger.info(f"Volume: {result['features']['volume']:,.0f}")
                logger.info(f"5-Day Momentum: {result['features']['momentum']:.2f}%")
        
        if not results:
            logger.error("No valid market data available")
            return
        
        # Calculate and display market-wide metrics
        market_sentiment = np.mean([r['sentiment']['composite'] for r in results])
        market_volatility = np.mean([r['features']['volatility'] for r in results])
        
        logger.info(f"\n{'-' * 50}")
        logger.info("\nMarket Summary:")
        logger.info(f"Overall Market Sentiment: {market_sentiment:.2f}")
        logger.info(f"Average Market Volatility: {market_volatility:.2f}%")
    
    except Exception as e:
        logger.error(f"Error in analyzing {market_name} market: {str(e)}")

def select_market_and_indices():
    """Interactive menu for selecting market and indices"""
    print("\nAvailable Markets:")
    print("1. Indian Market")
    print("2. US Market")
    print("3. Both Markets")
    
    while True:
        try:
            market_choice = input("\nSelect market (1-3): ").strip()
            if market_choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please try again.")
    
    selected_indices = {}
    
    if market_choice in ['1', '3']:
        print("\nIndian Market Indices:")
        indices = list(INDIAN_INDICES.items())
        for i, (symbol, name) in enumerate(indices, 1):
            print(f"{i}. {name} ({symbol})")
        
        while True:
            try:
                choices = input("\nSelect indices (comma-separated numbers, or 'all'): ").strip()
                if choices.lower() == 'all':
                    selected_indices['INDIA'] = dict(indices)
                    break
                
                selected = [int(x.strip()) for x in choices.split(',')]
                if all(1 <= x <= len(indices) for x in selected):
                    selected_indices['INDIA'] = {
                        indices[i-1][0]: indices[i-1][1] for i in selected
                    }
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    
    if market_choice in ['2', '3']:
        print("\nUS Market Indices:")
        indices = list(US_INDICES.items())
        for i, (symbol, name) in enumerate(indices, 1):
            print(f"{i}. {name} ({symbol})")
        
        while True:
            try:
                choices = input("\nSelect indices (comma-separated numbers, or 'all'): ").strip()
                if choices.lower() == 'all':
                    selected_indices['US'] = dict(indices)
                    break
                
                selected = [int(x.strip()) for x in choices.split(',')]
                if all(1 <= x <= len(indices) for x in selected):
                    selected_indices['US'] = {
                        indices[i-1][0]: indices[i-1][1] for i in selected
                    }
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    
    return selected_indices

def main():
    # Check for API key
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("Please set ALPHA_VANTAGE_API_KEY in .env file")
        sys.exit(1)
    
    # Get user selection
    selected_markets = select_market_and_indices()
    
    # Run analysis for selected markets and indices
    for market_name, indices in selected_markets.items():
        if indices:  # Only analyze if indices were selected
            asyncio.run(analyze_market(indices, market_name))

if __name__ == "__main__":
    main()
