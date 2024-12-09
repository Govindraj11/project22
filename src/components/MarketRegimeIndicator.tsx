import { Box, Typography, CircularProgress } from '@mui/material';
import { MarketEnvironment } from '../types';

type Props = {
  environment?: MarketEnvironment;
};

export default function MarketRegimeIndicator({ environment }: Props) {
  if (!environment) {
    return (
      <Box className="h-64 flex items-center justify-center">
        <Typography>No market regime data available</Typography>
      </Box>
    );
  }

  const getRegimeColor = () => {
    switch (environment.marketCondition) {
      case 'BULLISH_TREND':
        return 'text-green-600';
      case 'BEARISH_TREND':
        return 'text-red-600';
      case 'VOLATILE':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStrengthPercentage = () => {
    return Math.round((environment.strength + 1) * 50);
  };

  return (
    <Box>
      <Typography variant="h6" className="mb-4">Market Regime</Typography>
      
      <Box className="flex items-center justify-between mb-4">
        <Typography variant="body1">Current Regime:</Typography>
        <Typography className={`font-bold ${getRegimeColor()}`}>
          {environment.marketCondition.replace('_', ' ')}
        </Typography>
      </Box>

      <Box className="flex items-center justify-between mb-4">
        <Typography variant="body1">Trend:</Typography>
        <Typography className="font-bold">
          {environment.trend}
        </Typography>
      </Box>

      <Box className="flex items-center justify-between mb-4">
        <Typography variant="body1">Volatility:</Typography>
        <Typography className="font-bold">
          {environment.volatilityRegime}
        </Typography>
      </Box>

      <Box className="text-center mt-6">
        <Typography variant="body2" className="mb-2">Regime Strength</Typography>
        <Box className="relative inline-flex">
          <CircularProgress 
            variant="determinate" 
            value={getStrengthPercentage()} 
            size={80}
            className={getRegimeColor()}
          />
          <Box className="absolute inset-0 flex items-center justify-center">
            <Typography variant="body2">
              {getStrengthPercentage()}%
            </Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}