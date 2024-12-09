import { useState } from 'react';
import { Box, Grid, Paper, Typography, Alert } from '@mui/material';
import MarketOverview from './MarketOverview';
import TechnicalAnalysis from './TechnicalAnalysis';
import MarketRegimeIndicator from './MarketRegimeIndicator';
import VolumeAnalysis from './VolumeAnalysis';
import IndexSelector from './IndexSelector';
import { useMarketData } from '../hooks/useMarketData';
import { handleAPIError } from '../utils/errorHandling';

export default function Dashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  
  const { 
    data: marketData,
    isLoading,
    error
  } = useMarketData(selectedSymbol);

  return (
    <Box className="p-6 bg-gray-100 min-h-screen">
      <Typography variant="h4" className="mb-6">
        Quantum Market Analysis Dashboard
      </Typography>

      <IndexSelector 
        onSelect={setSelectedSymbol}
        selectedSymbol={selectedSymbol}
      />

      {error && (
        <Alert severity="error" className="mb-4">
          {handleAPIError(error)}
        </Alert>
      )}

      <Grid container spacing={3} className="mt-4">
        <Grid item xs={12} md={8}>
          <Paper className="p-4">
            <MarketOverview 
              index={marketData} 
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper className="p-4">
            <MarketRegimeIndicator 
              environment={marketData?.environment}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className="p-4">
            <TechnicalAnalysis 
              signals={marketData?.environment.detailedSignals}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className="p-4">
            <VolumeAnalysis 
              volumeData={marketData?.environment.detailedSignals.volume}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}