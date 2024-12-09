from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time
import pytz
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import logging

@dataclass
class RiskParameters:
    def __init__(self):
        self.volatility_threshold = 0.2
        self.momentum_threshold = 0.1
        self.trend_period = 20
        self.regime_lookback = 60
        self.max_position_size = 10000
        self.min_confidence = 0.5

class MarketRegimeDetector:
    def __init__(self):
        self.lookback_period = 60
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        if len(data) < self.lookback_period:
            return 'unknown'
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) < self.lookback_period:
            return 'unknown'
            
        trend = returns.mean() * 252  # Annualized return
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        if trend > 0.15 and volatility < 0.2:
            return 'BULLISH_TREND'
        elif trend < -0.15 and volatility < 0.2:
            return 'BEARISH_TREND'
        else:
            return 'SIDEWAYS'

class EnhancedQuantumOptionsAnalyzer:
    def __init__(self, n_qubits=20, risk_params: Optional[RiskParameters] = None):
        self.n_qubits = n_qubits
        self.risk_params = risk_params or RiskParameters()
        self.regime_detector = MarketRegimeDetector()  # Ensure this is initialized properly
        
    # Other methods remain unchanged
    
    def analyze_market_environment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market environment with detailed signals"""
        try:
            # Calculate moving averages and trends
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            ema_13 = data['Close'].ewm(span=13, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            
            # Get the latest values
            last_close = data['Close'].iloc[-1]
            last_sma20 = sma_20.iloc[-1]
            last_sma50 = sma_50.iloc[-1]
            last_ema13 = ema_13.iloc[-1]
            last_ema26 = ema_26.iloc[-1]
            
            # Calculate MACD
            macd = ema_13 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = sma_20 + (bb_std * 2)
            bb_lower = sma_20 - (bb_std * 2)
            
            # Determine trend signals
            trend_signals = []
            if last_close > last_sma20 and last_sma20 > last_sma50:
                trend = 'UPTREND'
                trend_signals.append("Price above both 20 and 50 SMAs")
            elif last_close < last_sma20 and last_sma20 < last_sma50:
                trend = 'DOWNTREND'
                trend_signals.append("Price below both 20 and 50 SMAs")
            else:
                trend = 'SIDEWAYS'
                trend_signals.append("Mixed signals between SMAs")
            
            # MACD signals
            macd_signals = []
            if macd.iloc[-1] > macd_signal.iloc[-1]:
                macd_signals.append("MACD above signal line (Bullish)")
            else:
                macd_signals.append("MACD below signal line (Bearish)")
            
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                macd_signals.append("MACD histogram increasing")
            else:
                macd_signals.append("MACD histogram decreasing")
            
            # RSI signals
            rsi_signals = []
            last_rsi = rsi.iloc[-1]
            if last_rsi > 70:
                rsi_signals.append(f"Overbought (RSI: {last_rsi:.2f})")
            elif last_rsi < 30:
                rsi_signals.append(f"Oversold (RSI: {last_rsi:.2f})")
            else:
                rsi_signals.append(f"Neutral RSI: {last_rsi:.2f}")
            
            # Volume analysis
            volume_signals = []
            avg_volume = data['Volume'].rolling(window=20).mean()
            if data['Volume'].iloc[-1] > avg_volume.iloc[-1] * 1.5:
                volume_signals.append("High volume (>50% above average)")
            elif data['Volume'].iloc[-1] < avg_volume.iloc[-1] * 0.5:
                volume_signals.append("Low volume (<50% below average)")
            else:
                volume_signals.append("Normal volume")
            
            # Calculate volatility regime
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            if volatility < 0.15:
                vol_regime = 'LOW'
                vol_desc = f"Low volatility ({volatility:.2%})"
            elif volatility < 0.25:
                vol_regime = 'MODERATE'
                vol_desc = f"Moderate volatility ({volatility:.2%})"
            else:
                vol_regime = 'HIGH'
                vol_desc = f"High volatility ({volatility:.2%})"
            
            # Determine overall market condition
            if trend == 'UPTREND' and vol_regime != 'HIGH' and last_rsi < 70:
                condition = 'BULLISH'
                strength = 'STRONG' if last_close > bb_upper.iloc[-1] else 'MODERATE'
            elif trend == 'DOWNTREND' and vol_regime != 'LOW' and last_rsi > 30:
                condition = 'BEARISH'
                strength = 'STRONG' if last_close < bb_lower.iloc[-1] else 'MODERATE'
            elif vol_regime == 'HIGH':
                condition = 'VOLATILE'
                strength = 'HIGH'
            else:
                condition = 'CONSOLIDATION'
                strength = 'NEUTRAL'
            
            # Price momentum
            momentum = data['Close'].pct_change(5).mean() * 100
            momentum_signal = "Positive" if momentum > 0 else "Negative"
            
            return {
                'trend': trend,
                'volatility_regime': vol_regime,
                'market_condition': condition,
                'strength': strength,
                'detailed_signals': {
                    'trend_signals': trend_signals,
                    'macd_signals': macd_signals,
                    'rsi_signals': rsi_signals,
                    'volume_signals': volume_signals,
                    'volatility': vol_desc,
                    'momentum': f"{momentum_signal} momentum ({momentum:.2f}%)"
                }
            }
            
        except Exception as e:
            print(f"Error analyzing market environment: {str(e)}")
            return {
                'trend': 'UNKNOWN',
                'volatility_regime': 'UNKNOWN',
                'market_condition': 'UNKNOWN',
                'strength': 'UNKNOWN',
                'detailed_signals': {
                    'trend_signals': [],
                    'macd_signals': [],
                    'rsi_signals': [],
                    'volume_signals': [],
                    'volatility': 'Unknown',
                    'momentum': 'Unknown'
                }
            }

    def calculate_technical_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate technical sentiment score (-1 to 1)"""
        try:
            # Calculate moving averages
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            # Get the latest values
            last_close = data['Close'].iloc[-1]
            last_sma20 = sma_20.iloc[-1]
            last_sma50 = sma_50.iloc[-1]
            
            # Calculate momentum
            momentum = data['Close'].pct_change(5).mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Combine signals
            ma_signal = 1 if last_close > last_sma20 and last_sma20 > last_sma50 else -1
            mom_signal = 1 if momentum > 0 else -1
            rsi_signal = -1 if rsi.iloc[-1] > 70 else 1 if rsi.iloc[-1] < 30 else 0
            
            # Weight and combine signals
            sentiment = np.average([ma_signal, mom_signal, rsi_signal],
                                 weights=[0.4, 0.4, 0.2])
            
            return float(sentiment)
        except Exception as e:
            print(f"Error calculating technical sentiment: {str(e)}")
            return 0.0

    def calculate_composite_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate composite sentiment including volume and volatility"""
        try:
            # Get technical sentiment
            tech_sentiment = self.calculate_technical_sentiment(data)
            
            # Calculate volume trend
            avg_volume = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'].iloc[-1] / avg_volume.iloc[-1]
            volume_signal = 1 if volume_ratio > 1.2 else -1 if volume_ratio < 0.8 else 0
            
            # Calculate volatility trend
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            vol_score = 1 - np.clip(volatility / 0.4, 0, 1)  # Normalize volatility
            
            # Combine signals with weights
            sentiment = np.average([tech_sentiment, volume_signal, vol_score],
                                 weights=[0.5, 0.3, 0.2])
            
            return float(sentiment)
        except Exception as e:
            print(f"Error calculating composite sentiment: {str(e)}")
            return 0.0
    
    def predict_enhanced_movement(self, features: dict, quantum_state: np.ndarray, 
                                market_conditions: dict) -> tuple:
        """Enhanced prediction with improved error handling and validation"""
        try:
            # Validate inputs
            if not all(k in features for k in ['momentum', 'volume_pressure', 'rsi', 'market_pressure']):
                raise ValueError("Missing required features")
                
            # Create and validate quantum circuit
            circuit = self.quantum_circuit.create_enhanced_circuit(
                features=list(features.values()),
                market_regime=market_conditions['regime'],
                options_greeks=market_conditions['greeks']
            )
            
            if circuit is None:
                raise ValueError("Failed to create quantum circuit")
                
            # Execute circuit with error mitigation
            result = self.execute_circuit_robust(circuit)
            if result is None:
                raise ValueError("Circuit execution failed")
                
            # Calculate probabilities with market adjustments
            up_prob, down_prob = self._calculate_probabilities_robust(
                result, market_conditions, features
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_robust(
                up_prob, down_prob, market_conditions
            )
            
            # Generate trading signal
            signal = self._generate_signal_robust(
                up_prob, down_prob, confidence
            )
            
            return signal, confidence, max(up_prob, down_prob)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 'NEUTRAL', 0.0, 0.5
            
    def execute_circuit_robust(self, circuit) -> Optional[np.ndarray]:
        """Execute quantum circuit with enhanced error mitigation"""
        try:
            # Configure backend with noise model
            backend_config = {
                'optimization_level': 1,
                'shots': 1024,
                'seed_simulator': 42,
                'noise_model': self.quantum_circuit.noise_model
            }
            
            # Execute with measurement error mitigation
            raw_result = self.quantum_circuit.backend.run(circuit, **backend_config).result()
            
            if self.quantum_circuit.measurement_fitter:
                mitigated_result = self.quantum_circuit.measurement_fitter.filter.apply(raw_result)
                return mitigated_result
                
            return raw_result
            
        except Exception as e:
            print(f"Circuit execution error: {str(e)}")
            return None
            
    def _calculate_probabilities_robust(self, result, market_conditions: dict, 
                                      features: dict) -> Tuple[float, float]:
        """Calculate probabilities with market condition adjustments"""
        try:
            # Extract base probabilities
            counts = result.get_counts()
            total_shots = sum(counts.values())
            
            # Calculate up/down probabilities
            up_prob = sum(counts.get(s, 0) for s in counts if s.count('1') > len(s)/2) / total_shots
            down_prob = 1.0 - up_prob
            
            # Apply market adjustments
            regime_factor = market_conditions['regime'][0]  # Use primary regime
            greek_factor = market_conditions['greeks'].get('delta', 0.0)
            
            # Adjust probabilities
            up_prob *= (1.0 + regime_factor * self.params['regime_sensitivity'])
            up_prob *= (1.0 + greek_factor * self.params['greek_weight'])
            
            # Normalize
            total = up_prob + down_prob
            return up_prob/total, down_prob/total
            
        except Exception as e:
            print(f"Probability calculation error: {str(e)}")
            return 0.5, 0.5
            
    def _calculate_confidence_robust(self, up_prob: float, down_prob: float, 
                                   market_conditions: dict) -> float:
        """Calculate confidence score with multiple factors"""
        try:
            # Calculate base confidence from probability spread
            base_confidence = abs(up_prob - down_prob)
            
            # Additional confidence factors
            regime_confidence = 1.0 - market_conditions.get('anomaly_score', 0.0)
            greek_confidence = min(abs(market_conditions['greeks'].get('delta', 0.0)), 1.0)
            
            # Weighted average of confidence factors
            confidence = np.mean([
                base_confidence * 2.0,  # Double weight to base confidence
                regime_confidence,
                greek_confidence
            ])
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            print(f"Confidence calculation error: {str(e)}")
            return 0.0
            
    def _generate_signal_robust(self, up_prob: float, down_prob: float, 
                              confidence: float) -> Tuple[str, float]:
        """Generate trading signal with validation"""
        try:
            # Apply confidence threshold
            if confidence < self.risk_params.min_confidence:
                return 'NEUTRAL'
                
            # Generate signal based on probability difference
            if up_prob > down_prob + 0.05:  # 5% threshold
                return 'LONG'
            elif down_prob > up_prob + 0.05:
                return 'SHORT'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            print(f"Signal generation error: {str(e)}")
            return 'NEUTRAL'

    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features to quantum amplitudes."""
        try:
            feature_values = np.array(list(features.values()))
            norm = np.linalg.norm(feature_values)
            if norm == 0:
                return np.zeros_like(feature_values)
            return feature_values / norm
        except Exception as e:
            raise ValueError(f"Feature normalization error: {str(e)}")

    def _apply_rotation(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply rotation gate to specified qubit."""
        cos_half = np.cos(angle/2)
        sin_half = np.sin(angle/2)
        
        new_state = np.zeros_like(state)
        n = len(state)
        for i in range(n):
            if i & (1 << qubit):  # If qubit is 1
                new_state[i] = -sin_half * state[i ^ (1 << qubit)] + cos_half * state[i]
            else:  # If qubit is 0
                new_state[i] = cos_half * state[i] + sin_half * state[i ^ (1 << qubit)]
        
        return new_state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate between control and target qubits."""
        new_state = np.zeros_like(state)
        n = len(state)
        
        for i in range(n):
            if i & (1 << control):  # If control qubit is 1
                new_state[i ^ (1 << target)] = state[i]  # Flip target qubit
            else:
                new_state[i] = state[i]  # Leave unchanged
        
        return new_state

    def create_enhanced_quantum_circuit(self, features: Dict[str, float], options_data: Dict[str, float]) -> np.ndarray:
        """Create an enhanced quantum circuit for options analysis."""
        # Initialize quantum state
        state = np.zeros(2**self.n_qubits)
        state[0] = 1.0  # Initialize to |0⟩ state
        
        # Normalize feature values to [-π/2, π/2]
        momentum = np.clip(features['momentum'] / 100, -1, 1) * np.pi/2
        volume = np.clip(features['volume_pressure'] - 1, -1, 1) * np.pi/2
        rsi = features['rsi_pressure'] * np.pi
        market = np.clip(features['market_pressure'] * 2, -1, 1) * np.pi/2
        
        # Enhanced quantum operations
        # Rotation gates for technical indicators
        state = self._apply_rotation(state, 0, momentum)
        state = self._apply_rotation(state, 1, volume)
        state = self._apply_rotation(state, 2, rsi)
        state = self._apply_rotation(state, 3, market)
        
        # Options-specific quantum operations
        iv_angle = np.clip(options_data['implied_volatility'] - 0.3, -1, 1) * np.pi/2
        vol_ratio = np.clip(options_data['volume_ratio'] - 1, -1, 1) * np.pi/2
        pc_ratio = np.clip(options_data['put_call_ratio'] - 1, -1, 1) * np.pi/2
        
        state = self._apply_rotation(state, 4, iv_angle)
        state = self._apply_rotation(state, 5, vol_ratio)
        state = self._apply_rotation(state, 6, pc_ratio)
        
        # Entanglement operations
        for i in range(self.n_qubits - 1):
            state = self._apply_cnot(state, i, i + 1)
        
        # Market regime detection
        if abs(market) > np.pi/4:  # Strong trend
            state = self._apply_rotation(state, 7, market * 1.5)
        
        # Volatility regime detection
        if abs(iv_angle) > np.pi/4:  # High volatility
            state = self._apply_rotation(state, 8, iv_angle * 1.5)
        
        return state

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for analysis."""
        try:
            # Ensure we have enough data
            if len(data) < 20:
                return {}
            
            # Convert to numpy arrays and ensure 1D
            close = data['Close'].values.reshape(-1)
            high = data['High'].values.reshape(-1)
            low = data['Low'].values.reshape(-1)
            volume = data['Volume'].values.reshape(-1)

            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # Calculate MACD
            macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            
            # Calculate VWAP
            typical_price = (high + low + close) / 3
            vwap = np.sum(typical_price * volume) / np.sum(volume)
            
            # Get the latest values, handling NaN
            latest_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            latest_macd = float(macd[-1] - signal[-1]) if not np.isnan(macd[-1]) and not np.isnan(signal[-1]) else 0.0
            latest_bb = float((close[-1] - middle[-1]) / (upper[-1] - lower[-1])) if not np.isnan(middle[-1]) else 0.0
            latest_vwap = float((close[-1] / vwap - 1)) if vwap != 0 else 0.0

            return {
                'rsi': latest_rsi,
                'macd_signal': latest_macd,
                'bollinger_signal': latest_bb,
                'vwap_signal': latest_vwap
            }
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return {}

    def calculate_position_size(self, confidence: float, option_price: float) -> float:
        """Calculate position size based on confidence and risk parameters."""
        try:
            # Base position size on confidence
            position_size = confidence * self.risk_params.max_position_size
            
            # Adjust for option price
            if option_price > 0:
                # Reduce position size for expensive options
                position_size *= np.exp(-option_price / 100)
            
            # Ensure within limits
            position_size = min(position_size, self.risk_params.max_position_size)
            
            return position_size
        except Exception as e:
            raise ValueError(f"Position size calculation error: {str(e)}")

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is currently open for the given symbol."""
        try:
            # Get current time in market timezone
            if symbol.endswith('.NS'):  # Indian market
                tz = pytz.timezone('Asia/Kolkata')
                market_open = time(9, 15)
                market_close = time(15, 30)
            else:  # US market
                tz = pytz.timezone('America/New_York')
                market_open = time(9, 30)
                market_close = time(16, 0)

            current_time = datetime.now(tz).time()
            
            # Check if it's a weekday and within market hours
            is_weekday = datetime.now(tz).weekday() < 5
            is_within_hours = market_open <= current_time <= market_close
            
            return is_weekday and is_within_hours
        except Exception as e:
            raise ValueError(f"Market hours check error: {str(e)}")

    def backtest_strategy(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Backtest the strategy on historical data."""
        try:
            # Download historical data
            data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
            if data.empty or len(data) < 20:
                print(f"Insufficient data for {symbol}")
                return pd.DataFrame()

            results = []
            for i in range(20, len(data)):
                try:
                    window = data.iloc[i-20:i+1]  # Include current day
                    if len(window) < 20:
                        continue
                    
                    # Calculate features with proper normalization
                    close_prices = window['Close'].values
                    volume_values = window['Volume'].values
                    
                    # Calculate RSI first
                    rsi_period = 14
                    if len(close_prices) > rsi_period:
                        delta = np.diff(close_prices)
                        gain = (delta > 0) * delta
                        loss = (delta < 0) * -delta
                        
                        # Use pandas rolling mean to avoid empty slice warning
                        avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean().iloc[-1]
                        avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean().iloc[-1]
                        
                        rs = avg_gain / avg_loss if avg_loss != 0 else 9.0  # Default to overbought if no losses
                        rsi = min(100.0, 100.0 - (100.0 / (1.0 + rs)))  # Ensure RSI is valid
                    else:
                        rsi = 50.0  # Default to neutral if not enough data
                    
                    # Normalize features to [-1, 1] with safety checks
                    try:
                        momentum = np.clip((close_prices[-1] / close_prices[-5] - 1) * 5, -1, 1)
                    except IndexError:
                        momentum = 0.0
                    
                    try:
                        recent_vol = np.mean(volume_values[-5:]) if len(volume_values) >= 5 else volume_values[-1]
                        avg_vol = np.mean(volume_values) if len(volume_values) > 0 else recent_vol
                        volume_pressure = np.clip(np.log(recent_vol / avg_vol) * 2, -1, 1) if avg_vol > 0 else 0.0
                    except (IndexError, ZeroDivisionError):
                        volume_pressure = 0.0
                    
                    rsi_norm = np.clip((rsi / 50.0) - 1.0, -1, 1)
                    
                    try:
                        market_pressure = np.clip(np.log(close_prices[-1] / close_prices[0]) * 5, -1, 1)
                    except (IndexError, ZeroDivisionError):
                        market_pressure = 0.0
                    
                    features = [float(momentum), float(volume_pressure), float(rsi_norm), float(market_pressure)]
                    
                    # Simplified options data for backtest
                    options_data = {
                        'implied_volatility': 0.3,  # Default values
                        'volume_ratio': 1.0,
                        'put_call_ratio': 1.0,
                        'gamma_exposure': 0.0,
                        'term_structure': 0.3,
                        'skew_signal': 0.0
                    }
                    
                    # Generate prediction
                    circuit = self.quantum_circuit.create_enhanced_circuit(
                        features=features,
                        market_regime=[0.4, 0.3, 0.2, 0.1],  # Default regime probabilities
                        options_greeks=options_data
                    )
                    
                    # Execute circuit
                    prediction = self.execute_circuit(circuit)
                    if prediction is None:
                        continue
                    
                    # Generate trading signals
                    confidence = abs(prediction - 0.5) * 2  # Scale to [0, 1]
                    signal = 'LONG' if prediction > 0.5 else 'SHORT'
                    
                    if confidence > self.risk_params.min_confidence and i + 1 < len(data):
                        # Calculate returns
                        next_return = (data['Close'].iloc[i+1] / data['Close'].iloc[i] - 1) * 100
                        expected_return = (prediction - 0.5) * 4  # Scale to [-2, 2]
                        hit = (expected_return * next_return) > 0
                        
                        results.append({
                            'Date': data.index[i],
                            'Signal': signal,
                            'Confidence': confidence,
                            'Expected_Return': expected_return,
                            'Actual_Return': next_return,
                            'Hit': hit,
                            'Close': data['Close'].iloc[i],
                            'Volume': data['Volume'].iloc[i],
                            'RSI': rsi,
                            'Momentum': momentum
                        })

                except Exception as e:
                    print(f"Error processing window for {symbol}: {str(e)}")
                    continue

            results_df = pd.DataFrame(results)
            if not results_df.empty:
                print(f"\nBacktest Results for {symbol}:")
                print(f"Total Trades: {len(results_df)}")
                print(f"Win Rate: {(results_df['Hit'].sum() / len(results_df)) * 100:.1f}%")
                print(f"Average Return: {results_df['Actual_Return'].mean():.2f}%")
                print(f"Sharpe Ratio: {(results_df['Actual_Return'].mean() / results_df['Actual_Return'].std()) * np.sqrt(252):.2f}")
            
            return results_df
            
        except Exception as e:
            print(f"Error in backtest for {symbol}: {str(e)}")
            return pd.DataFrame()

    def evaluate_accuracy(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Evaluate the model's accuracy across multiple symbols."""
        try:
            overall_results = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'by_symbol': {}
            }
            
            for symbol in symbols:
                print(f"\nAnalyzing {symbol}...")
                results_df = self.backtest_strategy(symbol, start_date, end_date)
                
                if not results_df.empty:
                    # Calculate metrics
                    win_rate = (results_df['Hit'].sum() / len(results_df)) * 100
                    avg_return = results_df['Actual_Return'].mean()
                    std_return = results_df['Actual_Return'].std()
                    sharpe = np.sqrt(252) * (avg_return / std_return) if std_return != 0 else 0
                    
                    # Calculate maximum drawdown
                    cumulative_returns = (1 + results_df['Actual_Return'] / 100).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdowns = (cumulative_returns - rolling_max) / rolling_max
                    max_drawdown = drawdowns.min() * 100
                    
                    symbol_results = {
                        'trades': len(results_df),
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_drawdown
                    }
                    
                    overall_results['by_symbol'][symbol] = symbol_results
                    overall_results['total_trades'] += len(results_df)
                    overall_results['win_rate'] += win_rate * len(results_df)
                    overall_results['avg_return'] += avg_return * len(results_df)
                    overall_results['sharpe_ratio'] += sharpe * len(results_df)
                    overall_results['max_drawdown'] = min(overall_results['max_drawdown'], max_drawdown)
            
            if overall_results['total_trades'] > 0:
                # Calculate weighted averages
                overall_results['win_rate'] /= overall_results['total_trades']
                overall_results['avg_return'] /= overall_results['total_trades']
                overall_results['sharpe_ratio'] /= overall_results['total_trades']
                
                print("\n=== Overall Performance ===")
                print(f"Total Trades: {overall_results['total_trades']}")
                print(f"Average Win Rate: {overall_results['win_rate']:.1f}%")
                print(f"Average Return per Trade: {overall_results['avg_return']:.2f}%")
                print(f"Sharpe Ratio: {overall_results['sharpe_ratio']:.2f}")
                print(f"Maximum Drawdown: {overall_results['max_drawdown']:.1f}%")
                
                print("\n=== Performance by Symbol ===")
                for symbol, metrics in overall_results['by_symbol'].items():
                    print(f"\n{symbol}:")
                    print(f"Trades: {metrics['trades']}")
                    print(f"Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"Avg Return: {metrics['avg_return']:.2f}%")
                    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
            
            return overall_results
            
        except Exception as e:
            raise ValueError(f"Accuracy evaluation error: {str(e)}")

    def execute_circuit(self, circuit):
        """Execute the quantum circuit with enhanced error mitigation"""
        try:
            # Configure backend
            backend_config = {
                'optimization_level': 1,
                'shots': 1024,
                'seed_simulator': 42
            }
            
            # Execute circuit
            result = self.backend.run(circuit, **backend_config).result()
            counts = result.get_counts()
            
            # Calculate probability of |1⟩ state
            total_shots = sum(counts.values())
            probability_one = sum(counts.get(s, 0) for s in counts if s.count('1') > len(s)/2) / total_shots
            
            # Scale prediction to [0.33, 1]
            prediction = (probability_one + 0.5) / 1.5
            
            return prediction
            
        except Exception as e:
            print(f"Error executing quantum circuit: {str(e)}")
            return None

    def _create_noise_model(self):
        """Create a realistic noise model for the quantum system"""
        try:
            from qiskit_aer.noise import NoiseModel
            from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error
            
            noise_model = NoiseModel()
            
            # Add realistic gate errors (single qubit)
            error_1 = depolarizing_error(0.001, 1)
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            
            # Add two-qubit gate errors
            error_2 = depolarizing_error(0.01, 2)
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            
            # Add T1/T2 relaxation errors
            t1, t2 = 50e-6, 70e-6  # Typical superconducting qubit values
            error_t1t2 = thermal_relaxation_error(t1, t2, 0.01)
            noise_model.add_all_qubit_quantum_error(error_t1t2, ['u1', 'u2', 'u3'])
            
            return noise_model
        except Exception as e:
            print(f"Error creating noise model: {str(e)}")
            return None

    def _execute_with_mitigation(self, circuit, config):
        """Execute circuit with error mitigation techniques"""
        try:
            # Execute main circuit without mitigation for now
            result = self.backend.run(circuit, **config).result()
            return result
        except Exception as e:
            print(f"Error in circuit execution: {str(e)}")
            return None

    def _create_measurement_fitter(self):
        """Create measurement calibration circuits and fitter"""
        try:
            # Try qiskit-ignis first
            try:
                from qiskit_ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
            except ImportError:
                # Fallback to newer qiskit version
                from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
            
            # Generate calibration circuits
            qr = QuantumRegister(self.n_qubits)
            meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
            
            # Execute calibration circuits
            cal_results = self.backend.run(meas_calibs).result()
            
            # Create measurement fitter
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)
            
            return meas_fitter
        except Exception as e:
            print(f"Error creating measurement fitter: {str(e)}")
            return None

    def _calculate_risk_metrics(self, option_data, market_data, quantum_probs):
        """Calculate risk metrics with improved error handling"""
        try:
            # Extract option data
            current_price = market_data['Close'].iloc[-1]
            strike_price = option_data['strike']
            days_to_expiry = option_data['dte']
            implied_vol = option_data['iv']
            
            # Calculate probability of profit
            prob_profit = quantum_probs.get('profit', 0.5)
            
            # Calculate risk/reward ratio
            max_loss = option_data['ask']  # Maximum loss is premium paid
            potential_profit = abs(strike_price - current_price) - option_data['ask']
            risk_reward = potential_profit / max_loss if max_loss > 0 else 0
            
            # Calculate risk score
            delta = abs(option_data['delta'])
            gamma = abs(option_data['gamma'])
            theta = abs(option_data['theta'])
            
            # Normalize metrics
            norm_delta = np.clip(delta / 0.5, 0, 1)  # Delta closer to 0.5 is better
            norm_gamma = np.clip(gamma / 0.1, 0, 1)  # Higher gamma means more risk
            norm_theta = np.clip(theta / 100, 0, 1)  # Higher theta means more risk
            
            # Combined risk score (lower is better)
            risk_score = (norm_delta + norm_gamma + norm_theta) / 3
            
            return {
                'risk_reward': float(risk_reward),
                'prob_profit': float(prob_profit * 100),
                'risk_score': float(risk_score)
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")
            return {
                'risk_reward': 0.0,
                'prob_profit': 0.0,
                'risk_score': 0.0
            }

    def analyze_market_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze market sentiment using multiple indicators"""
        try:
            # Calculate technical sentiment
            close = data['Close']
            returns = close.pct_change()
            
            # Calculate momentum
            momentum = (close[-1] / close[-20] - 1) * 100
            
            # Calculate trend strength
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()
            trend = 1 if ema20[-1] > ema50[-1] else -1
            
            # Calculate volatility regime
            volatility = returns.std() * np.sqrt(252)
            vol_score = 1 - np.clip(volatility / 0.4, 0, 1)  # Normalize volatility
            
            # Combine metrics
            technical_sentiment = np.clip((momentum * trend) / 100, -1, 1)
            
            return {
                'news_sentiment': 0.0,  # Placeholder for news sentiment
                'technical_sentiment': technical_sentiment,
                'composite_sentiment': technical_sentiment * vol_score
            }
            
        except Exception as e:
            print(f"Error analyzing market sentiment: {str(e)}")
            return {
                'news_sentiment': 0.0,
                'technical_sentiment': 0.0,
                'composite_sentiment': 0.0
            }
            
    def analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze market regime using the regime detector"""
        try:
            # Get regime probabilities
            probs, anomaly_score = self.regime_detector.detect_regime(data)
            
            # Map probabilities to regime names
            regime_names = ['bullish_trend', 'bearish_trend', 'sideways']
            regime_probs = {name: float(prob) for name, prob in zip(regime_names, probs)}
            
            # Determine current regime
            current_regime = regime_names[np.argmax(probs)]
            if anomaly_score > 0.8:
                current_regime = 'volatile'
                
            return {
                'current_regime': current_regime,
                'probabilities': regime_probs,
                'anomaly_score': float(anomaly_score)
            }
            
        except Exception as e:
            print(f"Error analyzing market regime: {str(e)}")
            return {
                'current_regime': 'unknown',
                'probabilities': {'unknown': 1.0},
                'anomaly_score': 0.0
            }
            
    def analyze_market_environment(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze overall market environment"""
        try:
            # Calculate trend
            returns = data['Close'].pct_change().dropna()
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            # Get the latest values
            last_close = data['Close'].iloc[-1]
            last_sma20 = sma_20.iloc[-1]
            last_sma50 = sma_50.iloc[-1]
            
            if last_close > last_sma20 and last_sma20 > last_sma50:
                trend = 'UPTREND'
            elif last_close < last_sma20 and last_sma20 < last_sma50:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            # Calculate volatility regime
            volatility = returns.std() * np.sqrt(252)
            if volatility < 0.15:
                vol_regime = 'LOW'
            elif volatility < 0.25:
                vol_regime = 'MODERATE'
            else:
                vol_regime = 'HIGH'
            
            # Determine market condition
            if trend == 'UPTREND' and vol_regime != 'HIGH':
                condition = 'BULLISH'
            elif trend == 'DOWNTREND' and vol_regime != 'LOW':
                condition = 'BEARISH'
            elif vol_regime == 'HIGH':
                condition = 'VOLATILE'
            else:
                condition = 'CONSOLIDATION'
            
            return {
                'trend': trend,
                'volatility_regime': vol_regime,
                'market_condition': condition
            }
            
        except Exception as e:
            print(f"Error analyzing market environment: {str(e)}")
            return {
                'trend': 'UNKNOWN',
                'volatility_regime': 'UNKNOWN',
                'market_condition': 'UNKNOWN'
            }

class EnhancedQuantumOptionsAnalyzerComprehensive:
    def __init__(self, risk_params=None):
        self.risk_params = risk_params or RiskParameters()
        
        # Initialize components with error handling
        try:
            self.regime_detector = MarketRegimeDetector()
            self.quantum_circuit = AdvancedQuantumCircuit()
            self.greeks_calculator = OptionsGreeksCalculator()
            self.microstructure_analyzer = MarketMicrostructureAnalyzer()
            self.parameter_optimizer = AdaptiveParameterOptimizer()
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            raise
        
        # Enhanced parameters
        self.params = {
            'volatility_window': 20,
            'momentum_window': 10,
            'regime_sensitivity': 0.5,
            'greek_weight': 1.0,
            'min_data_points': 30,
            'confidence_threshold': 0.6
        }
        
    def calculate_technical_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate technical sentiment score (-1 to 1)"""
        try:
            # Calculate technical indicators
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(data['Close'])
            
            # Generate sentiment signals
            price_trend = 1 if data['Close'].iloc[-1] > sma_20.iloc[-1] else -1
            sma_trend = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            rsi_signal = 1 if 40 < rsi < 60 else -1
            
            # Combine signals
            sentiment = np.mean([price_trend, sma_trend, rsi_signal])
            return float(sentiment)
        except Exception as e:
            print(f"Error in technical sentiment: {str(e)}")
            return 0.0
    
    def calculate_composite_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate composite sentiment including volume and volatility"""
        try:
            # Get technical sentiment
            tech_sentiment = self.calculate_technical_sentiment(data)
            
            # Calculate volume trend
            volume = data['Volume'].fillna(0)
            volume_trend = (volume[-5:].mean() / volume[-20:].mean() - 1) * 100
            volume_signal = 1 if volume_trend > 0 else -1
            
            # Calculate volatility trend
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            vol_score = 1 - np.clip(volatility / 0.4, 0, 1)  # Normalize volatility
            
            # Combine signals with weights
            sentiment = np.average([tech_sentiment, volume_signal, vol_score],
                                 weights=[0.5, 0.3, 0.2])
            return float(sentiment)
        except Exception as e:
            print(f"Error in composite sentiment: {str(e)}")
            return 0.0
    
    def detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime and transition probabilities"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Calculate regime indicators
            volatility = returns.std() * np.sqrt(252)
            momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
            trend = self._calculate_trend_strength(data['Close'])
            
            # Determine current regime
            if trend > 0.7 and momentum > 0:
                regime = 'BULLISH_TREND'
                probs = {'bullish_trend': 1.0, 'bearish_trend': 0.0, 'sideways': 0.0}
            elif trend < -0.7 and momentum < 0:
                regime = 'BEARISH_TREND'
                probs = {'bullish_trend': 0.0, 'bearish_trend': 1.0, 'sideways': 0.0}
            else:
                regime = 'SIDEWAYS'
                # Calculate transition probabilities
                bull_prob = max(0, min(1, (trend + 1) / 2))
                bear_prob = max(0, min(1, (-trend + 1) / 2))
                side_prob = 1 - (bull_prob + bear_prob)
                probs = {
                    'bullish_trend': bull_prob,
                    'bearish_trend': bear_prob,
                    'sideways': side_prob
                }
            
            return {
                'current_regime': regime,
                'probabilities': probs
            }
        except Exception as e:
            print(f"Error in market regime: {str(e)}")
            return {'current_regime': 'unknown'}
    
    def analyze_market_environment(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze overall market environment"""
        try:
            # Calculate trend
            returns = data['Close'].pct_change().dropna()
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            # Get the latest values
            last_close = data['Close'].iloc[-1]
            last_sma20 = sma_20.iloc[-1]
            last_sma50 = sma_50.iloc[-1]
            
            if last_close > last_sma20 and last_sma20 > last_sma50:
                trend = 'UPTREND'
            elif last_close < last_sma20 and last_sma20 < last_sma50:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            # Calculate volatility regime
            volatility = returns.std() * np.sqrt(252)
            if volatility < 0.15:
                vol_regime = 'LOW'
            elif volatility < 0.25:
                vol_regime = 'MODERATE'
            else:
                vol_regime = 'HIGH'
            
            # Determine market condition
            if trend == 'UPTREND' and vol_regime != 'HIGH':
                condition = 'BULLISH'
            elif trend == 'DOWNTREND' and vol_regime != 'LOW':
                condition = 'BEARISH'
            elif vol_regime == 'HIGH':
                condition = 'VOLATILE'
            else:
                condition = 'CONSOLIDATION'
            
            return {
                'trend': trend,
                'volatility_regime': vol_regime,
                'market_condition': condition
            }
        except Exception as e:
            print(f"Error in market environment: {str(e)}")
            return {
                'trend': 'UNKNOWN',
                'volatility_regime': 'UNKNOWN',
                'market_condition': 'UNKNOWN'
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
        except Exception as e:
            print(f"Error in RSI: {str(e)}")
            return 50.0
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (-1 to 1)"""
        try:
            # Calculate linear regression
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # Normalize trend strength
            strength = r_value * np.sign(slope)
            return float(strength)
        except Exception as e:
            print(f"Error in trend strength: {str(e)}")
            return 0.0
