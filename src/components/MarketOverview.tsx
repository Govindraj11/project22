import { Box, Typography, Grid } from '@mui/material';
import { MarketIndex } from '../types';
import PriceChart from './charts/PriceChart';
import VolumeChart from './charts/VolumeChart';

type Props = {
  index: MarketIndex | null;
  isLoading: boolean;
};

export default function MarketOverview({ index, isLoading }: Props) {
  if (isLoading) {
    return (
      <Box className="h-64 flex items-center justify-center">
        <Typography>Loading market data...</Typography>
      </Box>
    );
  }

  if (!index) {
    return (
      <Box className="h-64 flex items-center justify-center">
        <Typography>Select a market index to view analysis</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box className="flex justify-between mb-4">
        <Typography variant="h6">{index.name}</Typography>
        <Box>
          <Typography variant="h6" className="text-right">
            {index.lastPrice.toFixed(2)}
          </Typography>
          <Typography 
            className={index.dailyChange >= 0 ? 'text-green-600' : 'text-red-600'}
          >
            {index.dailyChange >= 0 ? '+' : ''}{index.dailyChange.toFixed(2)}%
          </Typography>
        </Box>
      </Box>
      
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <PriceChart historicalPrices={index.historicalPrices} />
        </Grid>
        <Grid item xs={12}>
          <VolumeChart historicalPrices={index.historicalPrices} />
        </Grid>
      </Grid>
    </Box>
  );
}