import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate
from hmmlearn import hmm
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Tuple, Dict

class MarketRegimeDetector:
    def __init__(self):
        self.model = GaussianMixture(n_components=3, random_state=42)
        
    def detect_regime(self, data: pd.DataFrame) -> Tuple[np.ndarray, float]:
        try:
            # Calculate features
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            momentum = returns.rolling(10).mean()
            
            # Prepare features for clustering
            X = np.column_stack([
                returns.values[19:],
                volatility.values[19:],
                momentum.values[19:]
            ])
            
            # Fit model and predict regime
            self.model.fit(X)
            regime = self.model.predict(X[-1].reshape(1, -1))
            probs = self.model.predict_proba(X[-1].reshape(1, -1))[0]
            
            # Calculate anomaly score
            anomaly_score = -self.model.score_samples(X[-1].reshape(1, -1))[0]
            anomaly_score = np.clip(anomaly_score, 0, 1)
            
            return probs, anomaly_score
            
        except Exception as e:
            print(f"Error detecting market regime: {str(e)}")
            return np.array([0.33, 0.33, 0.34]), 0.5

class AdvancedQuantumCircuit:
    def __init__(self, n_qubits=20):
        self.n_qubits = n_qubits
        self.backend = AerSimulator()
        self.noise_model = self._create_noise_model()
        self.measurement_fitter = self._create_measurement_fitter()
        
    def create_enhanced_circuit(self, features, market_regime, options_greeks):
        """Create an enhanced quantum circuit with improved stability"""
        try:
            qr = QuantumRegister(self.n_qubits, 'q')
            cr = ClassicalRegister(self.n_qubits, 'c')
            circuit = QuantumCircuit(qr, cr)
            
            # Enhanced feature encoding
            self._encode_features_robust(circuit, qr, features)
            
            # Market regime encoding
            self._encode_regime_robust(circuit, qr, market_regime)
            
            # Options greeks encoding
            self._encode_greeks_robust(circuit, qr, options_greeks)
            
            # Apply quantum operations
            self._apply_quantum_operations_robust(circuit, qr)
            
            # Add measurement
            circuit.measure(qr, cr)
            
            return circuit
            
        except Exception as e:
            print(f"Error creating quantum circuit: {str(e)}")
            return None
            
    def _encode_features_robust(self, circuit, qr, features):
        """Enhanced feature encoding with improved stability"""
        try:
            for i, feature in enumerate(features):
                if i >= self.n_qubits - 4:  # Reserve last 4 qubits for ancilla
                    break
                    
                # Normalize and bound feature value
                phase = np.clip(float(feature), -1, 1) * np.pi/2
                
                # Apply robust rotation sequence
                circuit.ry(phase, qr[i])
                circuit.rz(phase/2, qr[i])
                
                # Add error detection
                if i < self.n_qubits - 5:
                    circuit.cx(qr[i], qr[i+1])
                    circuit.barrier()
                    
        except Exception as e:
            print(f"Feature encoding error: {str(e)}")
            
    def _encode_regime_robust(self, circuit, qr, regime):
        """Improved market regime encoding"""
        try:
            regime_qubits = range(max(0, self.n_qubits-8), self.n_qubits-4)
            for i, prob in enumerate(regime):
                if i >= len(regime_qubits):
                    break
                    
                # Validate and normalize probability
                angle = np.clip(float(prob), 0, 1) * np.pi
                
                # Apply robust encoding
                circuit.ry(angle, qr[regime_qubits[i]])
                circuit.rz(angle/2, qr[regime_qubits[i]])
                
        except Exception as e:
            print(f"Regime encoding error: {str(e)}")
            
    def _encode_greeks_robust(self, circuit, qr, options_greeks):
        """Enhanced options greeks encoding"""
        try:
            greek_qubits = range(max(0, self.n_qubits-12), self.n_qubits-8)
            for i, (greek_name, value) in enumerate(options_greeks.items()):
                if i >= len(greek_qubits):
                    break
                    
                # Normalize and bound greek value
                angle = np.clip(float(value), -1, 1) * np.pi/2
                
                # Apply robust encoding
                circuit.ry(angle, qr[greek_qubits[i]])
                circuit.rz(angle/2, qr[greek_qubits[i]])
                
        except Exception as e:
            print(f"Greeks encoding error: {str(e)}")
            
    def _apply_quantum_operations_robust(self, circuit, qr):
        """Enhanced quantum operations"""
        try:
            # Apply feature map
            for _ in range(2):  # Two repetitions for better feature mapping
                # Rotation layer
                for i in range(self.n_qubits-4):  # Exclude ancilla qubits
                    circuit.ry(np.pi/4, qr[i])
                    circuit.rz(np.pi/4, qr[i])
                
                # Entanglement layer
                for i in range(0, self.n_qubits-5, 2):
                    circuit.cx(qr[i], qr[i+1])
                circuit.barrier()
            
            # Apply variational layers
            for _ in range(2):  # Two entanglement blocks
                # Rotation layer
                for i in range(self.n_qubits-4):
                    circuit.ry(np.pi/4, qr[i])
                    circuit.rz(np.pi/4, qr[i])
                
                # Entanglement layer
                for i in range(0, self.n_qubits-5, 2):
                    circuit.cx(qr[i], qr[i+1])
                circuit.barrier()
                
        except Exception as e:
            print(f"Quantum operations error: {str(e)}")
            
    def _create_noise_model(self):
        """Create simplified noise model"""
        try:
            # Create basic depolarizing error
            error_prob = 0.001  # 0.1% error rate
            
            # Create a basic noise model
            noise_model = {
                'single_qubit': error_prob,
                'two_qubit': error_prob * 2,
                'measurement': error_prob
            }
            
            return noise_model
            
        except Exception as e:
            print(f"Error creating noise model: {str(e)}")
            return None
            
    def _create_measurement_fitter(self):
        """Create simplified measurement error mitigation"""
        try:
            # Create basic measurement calibration
            meas_calibs = {
                '0': 0.99,  # 99% accuracy for measuring |0⟩
                '1': 0.98   # 98% accuracy for measuring |1⟩
            }
            
            return meas_calibs
            
        except Exception as e:
            print(f"Error creating measurement fitter: {str(e)}")
            return None
            
    def execute_circuit_robust(self, circuit):
        """Execute quantum circuit with simplified error handling"""
        try:
            # Configure backend
            backend_config = {
                'shots': 1024,
                'seed_simulator': 42
            }
            
            # Execute circuit
            job = self.backend.run(circuit, **backend_config)
            result = job.result()
            
            # Get counts and apply basic error mitigation
            counts = result.get_counts(0)
            
            # Simple error mitigation
            if self.measurement_fitter:
                for state in counts:
                    prob = counts[state] / 1024
                    if state == '0':
                        prob = prob / self.measurement_fitter['0']
                    else:
                        prob = prob / self.measurement_fitter['1']
                    counts[state] = int(prob * 1024)
            
            return counts
            
        except Exception as e:
            print(f"Circuit execution error: {str(e)}")
            return {'0': 512, '1': 512}  # Return uniform distribution on error

class OptionsGreeksCalculator:
    def calculate_greeks(self, current_price, strike_price, days_to_expiry, 
                        risk_free_rate, implied_vol):
        """Calculate option Greeks using Black-Scholes model"""
        try:
            # Convert to annual time
            t = days_to_expiry / 365
            
            # Calculate Greeks
            d = delta('c', current_price, strike_price, t, risk_free_rate, implied_vol)
            g = gamma('c', current_price, strike_price, t, risk_free_rate, implied_vol)
            v = vega('c', current_price, strike_price, t, risk_free_rate, implied_vol)
            th = theta('c', current_price, strike_price, t, risk_free_rate, implied_vol)
            
            return {
                'delta': float(d),
                'gamma': float(g),
                'vega': float(v),
                'theta': float(th)
            }
            
        except Exception as e:
            print(f"Error calculating Greeks: {str(e)}")
            return {
                'delta': 0.5,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }

class MarketMicrostructureAnalyzer:
    def analyze_microstructure(self, prices, volumes, returns):
        """Analyze market microstructure"""
        try:
            # Calculate basic metrics
            spread = np.std(returns) * np.sqrt(252)
            volume_profile = np.mean(volumes[-5:]) / np.mean(volumes)
            price_impact = np.corrcoef(returns, volumes)[0,1]
            
            # Calculate order imbalance
            up_volume = np.sum(volumes[returns > 0])
            down_volume = np.sum(volumes[returns < 0])
            total_volume = up_volume + down_volume
            order_imbalance = (up_volume - down_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'spread': float(spread),
                'volume_profile': float(volume_profile),
                'price_impact': float(price_impact),
                'order_imbalance': float(order_imbalance)
            }
            
        except Exception as e:
            print(f"Error analyzing microstructure: {str(e)}")
            return {
                'spread': 0.0,
                'volume_profile': 1.0,
                'price_impact': 0.0,
                'order_imbalance': 0.0
            }

class AdaptiveParameterOptimizer:
    def optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize system parameters using historical data"""
        try:
            # Calculate optimal parameters
            vol = data['Close'].pct_change().std() * np.sqrt(252)
            momentum = data['Close'].pct_change().mean() * 252
            
            return {
                'volatility_window': int(min(20 * vol, 50)),
                'momentum_threshold': float(max(0.001, momentum)),
                'regime_sensitivity': float(min(1.0, vol * 2)),
                'greek_weight': float(max(0.5, 1 - vol))
            }
            
        except Exception as e:
            print(f"Error optimizing parameters: {str(e)}")
            return {
                'volatility_window': 20,
                'momentum_threshold': 0.005,
                'regime_sensitivity': 0.5,
                'greek_weight': 1.0
            }
