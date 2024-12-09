export type MarketData = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}[];

export type MarketRegime = 'BULLISH_TREND' | 'BEARISH_TREND' | 'SIDEWAYS' | 'VOLATILE';

export type TrendDirection = 'BULLISH' | 'BEARISH' | 'NEUTRAL';

export type VolatilityRegime = 'LOW' | 'MODERATE' | 'HIGH';

export type TechnicalSignals = {
  trend: {
    shortTerm: TrendDirection;
    longTerm: TrendDirection;
  };
  momentum: {
    rsi: number;
    macd: {
      value: number;
      signal: number;
    };
  };
  volume: {
    trend: 'INCREASING' | 'DECREASING';
    strength: number;
  };
};

export type MarketEnvironment = {
  trend: TrendDirection;
  volatilityRegime: VolatilityRegime;
  marketCondition: MarketRegime;
  strength: number;
  detailedSignals: TechnicalSignals;
};

export type HistoricalPrice = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type MarketIndex = {
  symbol: string;
  name: string;
  lastPrice: number;
  dailyChange: number;
  environment: MarketEnvironment;
  historicalPrices: HistoricalPrice[];
};