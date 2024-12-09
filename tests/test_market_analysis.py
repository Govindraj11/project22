import unittest
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from quantum_options_analyzer import RiskParameters, EnhancedQuantumOptionsAnalyzer

class TestRiskParameters(unittest.TestCase):
    def setUp(self):
        self.risk_params = RiskParameters()
    
    def test_default_values(self):
        """Test default values of RiskParameters"""
        self.assertEqual(self.risk_params.volatility_threshold, 0.2)
        self.assertEqual(self.risk_params.momentum_threshold, 0.1)
        self.assertEqual(self.risk_params.trend_period, 20)
        self.assertEqual(self.risk_params.regime_lookback, 60)

class TestMarketAnalyzer(unittest.TestCase):
    def setUp(self):
        self.risk_params = RiskParameters()
        self.analyzer = EnhancedQuantumOptionsAnalyzer(risk_params=self.risk_params)
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.uniform(1000000, 2000000, len(dates))
        }, index=dates)
    
    def test_technical_sentiment(self):
        """Test technical sentiment calculation"""
        sentiment = self.analyzer.calculate_technical_sentiment(self.sample_data)
        self.assertIsInstance(sentiment, float)
        self.assertTrue(-1 <= sentiment <= 1)
    
    def test_composite_sentiment(self):
        """Test composite sentiment calculation"""
        sentiment = self.analyzer.calculate_composite_sentiment(self.sample_data)
        self.assertIsInstance(sentiment, float)
        self.assertTrue(-1 <= sentiment <= 1)
    
    def test_market_regime(self):
        """Test market regime detection"""
        regime = self.analyzer.detect_market_regime(self.sample_data)
        self.assertIsInstance(regime, dict)
        self.assertIn('current_regime', regime)
        self.assertIn(regime['current_regime'], ['BULLISH_TREND', 'BEARISH_TREND', 'SIDEWAYS', 'unknown'])
    
    def test_market_environment(self):
        """Test market environment analysis"""
        env = self.analyzer.analyze_market_environment(self.sample_data)
        self.assertIsInstance(env, dict)
        self.assertIn('trend', env)
        self.assertIn('volatility_regime', env)
        self.assertIn('market_condition', env)
        self.assertIn(env['trend'], ['UPTREND', 'DOWNTREND', 'SIDEWAYS', 'UNKNOWN'])
        self.assertIn(env['volatility_regime'], ['LOW', 'MODERATE', 'HIGH', 'UNKNOWN'])
        self.assertIn(env['market_condition'], ['BULLISH', 'BEARISH', 'VOLATILE', 'CONSOLIDATION', 'UNKNOWN'])

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.analyzer._calculate_rsi(self.sample_data['Close'])
        self.assertIsInstance(rsi, float)
        self.assertTrue(0 <= rsi <= 100)
    
    def test_trend_strength(self):
        """Test trend strength calculation"""
        strength = self.analyzer._calculate_trend_strength(self.sample_data['Close'])
        self.assertIsInstance(strength, float)
        self.assertTrue(-1 <= strength <= 1)
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        empty_data = pd.DataFrame()
        sentiment = self.analyzer.calculate_technical_sentiment(empty_data)
        self.assertEqual(sentiment, 0.0)
        
        regime = self.analyzer.detect_market_regime(empty_data)
        self.assertEqual(regime['current_regime'], 'unknown')
        
        env = self.analyzer.analyze_market_environment(empty_data)
        self.assertEqual(env['trend'], 'UNKNOWN')
        self.assertEqual(env['volatility_regime'], 'UNKNOWN')
        self.assertEqual(env['market_condition'], 'UNKNOWN')

if __name__ == '__main__':
    unittest.main()
