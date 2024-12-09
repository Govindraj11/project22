import { MarketData, MarketRegime, MarketEnvironment, TechnicalSignals } from '../types';

export const calculateTechnicalSignals = (data: MarketData): TechnicalSignals => {
  const prices = data.map(d => d.close);
  const volumes = data.map(d => d.volume);
  
  // Calculate moving averages
  const sma20 = calculateSMA(prices, 20);
  const sma50 = calculateSMA(prices, 50);
  const sma200 = calculateSMA(prices, 200);
  
  // Calculate RSI
  const rsi = calculateRSI(prices, 14);
  
  // Calculate MACD
  const macd = calculateMACD(prices);
  
  // Volume analysis
  const volumeProfile = analyzeVolumeProfile(volumes);
  
  return {
    trend: {
      shortTerm: sma20 > sma50 ? 'BULLISH' : 'BEARISH',
      longTerm: sma50 > sma200 ? 'BULLISH' : 'BEARISH'
    },
    momentum: {
      rsi,
      macd
    },
    volume: volumeProfile
  };
};

export const detectMarketRegime = (data: MarketData): MarketRegime => {
  const signals = calculateTechnicalSignals(data);
  const volatility = calculateVolatility(data.map(d => d.close));
  
  if (volatility > 0.25) {
    return 'VOLATILE';
  }
  
  if (signals.trend.shortTerm === 'BULLISH' && signals.trend.longTerm === 'BULLISH') {
    return 'BULLISH_TREND';
  }
  
  if (signals.trend.shortTerm === 'BEARISH' && signals.trend.longTerm === 'BEARISH') {
    return 'BEARISH_TREND';
  }
  
  return 'SIDEWAYS';
};

export const analyzeMarketEnvironment = (data: MarketData): MarketEnvironment => {
  const signals = calculateTechnicalSignals(data);
  const regime = detectMarketRegime(data);
  const volatility = calculateVolatility(data.map(d => d.close));
  
  return {
    trend: signals.trend.longTerm,
    volatilityRegime: categorizeVolatility(volatility),
    marketCondition: regime,
    strength: calculateMarketStrength(signals),
    detailedSignals: signals
  };
};

// Helper functions
const calculateSMA = (prices: number[], period: number): number => {
  if (prices.length < period) return 0;
  const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
  return sum / period;
};

const calculateRSI = (prices: number[], period: number): number => {
  if (prices.length < period + 1) return 50;
  
  const changes = prices.slice(1).map((price, i) => price - prices[i]);
  const gains = changes.map(change => change > 0 ? change : 0);
  const losses = changes.map(change => change < 0 ? -change : 0);
  
  const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
  const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
};

const calculateMACD = (prices: number[]): { value: number; signal: number } => {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  const macdLine = ema12 - ema26;
  const signalLine = calculateEMA([macdLine], 9);
  
  return {
    value: macdLine,
    signal: signalLine
  };
};

const calculateEMA = (prices: number[], period: number): number => {
  if (prices.length < period) return prices[prices.length - 1];
  
  const multiplier = 2 / (period + 1);
  let ema = prices[0];
  
  for (let i = 1; i < prices.length; i++) {
    ema = (prices[i] - ema) * multiplier + ema;
  }
  
  return ema;
};

const calculateVolatility = (prices: number[]): number => {
  const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
  return Math.sqrt(variance * 252); // Annualized volatility
};

const categorizeVolatility = (volatility: number): 'LOW' | 'MODERATE' | 'HIGH' => {
  if (volatility < 0.15) return 'LOW';
  if (volatility < 0.25) return 'MODERATE';
  return 'HIGH';
};

const analyzeVolumeProfile = (volumes: number[]) => {
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
  const recentVolume = volumes.slice(-5).reduce((a, b) => a + b, 0) / 5;
  
  return {
    trend: recentVolume > avgVolume ? 'INCREASING' : 'DECREASING',
    strength: recentVolume / avgVolume
  };
};

const calculateMarketStrength = (signals: TechnicalSignals): number => {
  let strength = 0;
  
  // Trend alignment
  if (signals.trend.shortTerm === signals.trend.longTerm) {
    strength += 0.3;
  }
  
  // RSI
  if (signals.momentum.rsi > 70) {
    strength += 0.2;
  } else if (signals.momentum.rsi < 30) {
    strength -= 0.2;
  }
  
  // MACD
  if (signals.momentum.macd.value > signals.momentum.macd.signal) {
    strength += 0.2;
  } else {
    strength -= 0.2;
  }
  
  // Volume
  if (signals.volume.trend === 'INCREASING') {
    strength += 0.3 * signals.volume.strength;
  }
  
  return Math.max(-1, Math.min(1, strength));
};