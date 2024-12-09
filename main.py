import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple
import warnings
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantum_options_analyzer import EnhancedQuantumOptionsAnalyzer, RiskParameters

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def fetch_market_data(symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
    """Fetch market data with enhanced error handling"""
    try:
        # Fetch data with retry mechanism
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval='1d')
                if not data.empty:
                    return data
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
        return None

def create_analysis_plots(data: pd.DataFrame, analysis_results: Dict) -> None:
    """Create interactive analysis plots"""
    try:
        # Create subplot figure
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Price Action', 'Options Chain',
                                         'Greeks Analysis', 'Risk Profile'),
                           vertical_spacing=0.15)
        
        # Price Action Plot
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', opacity=0.3),
            row=1, col=1
        )
        
        # Options Chain Plot
        if 'options_data' in analysis_results:
            options = analysis_results['options_data']
            strikes = []
            ivs = []
            for option in options:
                strikes.append(option['strike'])
                ivs.append(option['iv'])
            
            fig.add_trace(
                go.Scatter(x=strikes, y=ivs, mode='lines+markers',
                          name='Implied Volatility'),
                row=1, col=2
            )
        
        # Greeks Analysis Plot
        if 'greeks' in analysis_results:
            greeks = analysis_results['greeks']
            fig.add_trace(
                go.Bar(x=list(greeks.keys()), y=list(greeks.values()),
                      name='Greeks'),
                row=2, col=1
            )
        
        # Risk Profile Plot
        if 'risk_metrics' in analysis_results:
            risk = analysis_results['risk_metrics']
            fig.add_trace(
                go.Scatter(x=[risk['risk_score']], y=[risk['reward_potential']],
                          mode='markers', name='Risk/Reward',
                          marker=dict(size=15, color='red')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(height=800, title_text="Options Analysis Dashboard")
        fig.show()
        
    except Exception as e:
        logger.error(f"Error creating analysis plots: {str(e)}")

def main():
    try:
        # Use a default symbol for testing
        symbol = "AAPL"  # Default to Apple Inc.
        logger.info(f"\nAnalyzing options for {symbol}...")
        
        # Initialize analyzer with default risk parameters
        risk_params = RiskParameters()
        analyzer = EnhancedQuantumOptionsAnalyzer(risk_params=risk_params)
        
        # Fetch market data
        market_data = fetch_market_data(symbol)
        if market_data is None:
            logger.error(f"Failed to fetch market data for {symbol}")
            return
            
        # Process market data
        features = analyzer.process_market_data(market_data)
        if features is None:
            logger.error(f"Failed to process market data for {symbol}")
            return
            
        # Display processed features
        logger.info("\nProcessed Features:")
        for name, value in features.items():
            logger.info(f"{name}: {value:.4f}")
            
        # Analyze market sentiment
        sentiment = analyzer.analyze_market_sentiment(market_data)
        logger.info("\nMarket Sentiment Analysis:")
        logger.info(f"News Sentiment: {sentiment.get('news_sentiment', 0.0):.2f}")
        logger.info(f"Technical Sentiment: {sentiment.get('technical_sentiment', 0.0):.2f}")
        logger.info(f"Composite Sentiment: {sentiment.get('composite_sentiment', 0.0):.2f}")
        
        # Analyze market regime
        regime = analyzer.analyze_market_regime(market_data)
        logger.info("\nMarket Regime Analysis:")
        logger.info(f"Current Regime: {regime.get('current_regime', 'unknown')}")
        logger.info("Regime Probabilities:")
        for r, p in regime.get('probabilities', {}).items():
            logger.info(f"  {r}: {p*100:.2f}%")
            
        # Analyze market environment
        environment = analyzer.analyze_market_environment(market_data)
        logger.info("\nMarket Environment:")
        logger.info(f"Trend: {environment.get('trend', 'UNKNOWN')}")
        logger.info(f"Volatility Regime: {environment.get('volatility_regime', 'UNKNOWN')}")
        logger.info(f"Market Condition: {environment.get('market_condition', 'UNKNOWN')}")
        
        # Create analysis plots
        create_analysis_plots(market_data, {
            'features': features,
            'sentiment': sentiment,
            'regime': regime,
            'environment': environment
        })
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
