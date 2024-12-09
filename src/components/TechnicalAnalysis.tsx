import { Box, Typography, Grid, Paper } from '@mui/material';
import { TechnicalSignals } from '../types';

type Props = {
  signals?: TechnicalSignals;
};

export default function TechnicalAnalysis({ signals }: Props) {
  if (!signals) {
    return (
      <Box className="h-64 flex items-center justify-center">
        <Typography>No technical analysis data available</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" className="mb-4">Technical Analysis</Typography>

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Paper className="p-3">
            <Typography variant="subtitle2">Trend Analysis</Typography>
            <Box className="mt-2">
              <Box className="flex justify-between">
                <Typography>Short Term:</Typography>
                <Typography className={
                  signals.trend.shortTerm === 'BULLISH' ? 'text-green-600' : 
                  signals.trend.shortTerm === 'BEARISH' ? 'text-red-600' : 'text-gray-600'
                }>
                  {signals.trend.shortTerm}
                </Typography>
              </Box>
              <Box className="flex justify-between mt-1">
                <Typography>Long Term:</Typography>
                <Typography className={
                  signals.trend.longTerm === 'BULLISH' ? 'text-green-600' : 
                  signals.trend.longTerm === 'BEARISH' ? 'text-red-600' : 'text-gray-600'
                }>
                  {signals.trend.longTerm}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper className="p-3">
            <Typography variant="subtitle2">Momentum Indicators</Typography>
            <Box className="mt-2">
              <Box className="flex justify-between">
                <Typography>RSI:</Typography>
                <Typography className={
                  signals.momentum.rsi > 70 ? 'text-red-600' :
                  signals.momentum.rsi < 30 ? 'text-green-600' : 'text-gray-600'
                }>
                  {signals.momentum.rsi.toFixed(2)}
                </Typography>
              </Box>
              <Box className="flex justify-between mt-1">
                <Typography>MACD:</Typography>
                <Typography className={
                  signals.momentum.macd.value > signals.momentum.macd.signal ? 
                  'text-green-600' : 'text-red-600'
                }>
                  {signals.momentum.macd.value.toFixed(2)}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}