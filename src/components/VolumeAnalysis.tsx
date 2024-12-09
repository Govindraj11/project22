import { Box, Typography, LinearProgress } from '@mui/material';
import { TechnicalSignals } from '../types';

type Props = {
  volumeData?: TechnicalSignals['volume'];
};

export default function VolumeAnalysis({ volumeData }: Props) {
  if (!volumeData) {
    return (
      <Box className="h-64 flex items-center justify-center">
        <Typography>No volume analysis data available</Typography>
      </Box>
    );
  }

  const volumeStrengthPercentage = Math.round(volumeData.strength * 100);

  return (
    <Box>
      <Typography variant="h6" className="mb-4">Volume Analysis</Typography>

      <Box className="mb-4">
        <Box className="flex justify-between mb-2">
          <Typography>Volume Trend:</Typography>
          <Typography className={
            volumeData.trend === 'INCREASING' ? 'text-green-600' : 'text-red-600'
          }>
            {volumeData.trend}
          </Typography>
        </Box>

        <Box className="mb-4">
          <Typography variant="body2" className="mb-1">Volume Strength</Typography>
          <LinearProgress 
            variant="determinate" 
            value={volumeStrengthPercentage}
            className={volumeData.trend === 'INCREASING' ? 'bg-green-200' : 'bg-red-200'}
          />
          <Typography variant="body2" className="mt-1 text-right">
            {volumeStrengthPercentage}%
          </Typography>
        </Box>
      </Box>

      <Box className="mt-4 p-3 bg-gray-100 rounded">
        <Typography variant="subtitle2" className="mb-2">Volume Analysis Summary</Typography>
        <Typography variant="body2">
          {volumeData.trend === 'INCREASING' 
            ? 'Increasing volume suggests strong market participation and trend confirmation.'
            : 'Decreasing volume might indicate weakening trend or potential reversal.'}
        </Typography>
      </Box>
    </Box>
  );
}