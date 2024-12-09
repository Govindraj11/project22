import { Box, FormControl, InputLabel, MenuItem, Select, CircularProgress } from '@mui/material';
import { MarketIndex } from '../types';
import { useAvailableIndices } from '../hooks/useMarketData';
import { handleAPIError } from '../utils/errorHandling';

type Props = {
  onSelect: (symbol: string) => void;
  selectedSymbol: string | null;
};

export default function IndexSelector({ onSelect, selectedSymbol }: Props) {
  const { 
    data: indices, 
    isLoading, 
    error 
  } = useAvailableIndices();

  if (isLoading) {
    return (
      <Box className="flex justify-center p-4">
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box className="text-red-600 p-4">
        {handleAPIError(error)}
      </Box>
    );
  }

  return (
    <Box className="mb-6">
      <FormControl fullWidth>
        <InputLabel>Select Market Index</InputLabel>
        <Select
          value={selectedSymbol || ''}
          label="Select Market Index"
          onChange={(e) => onSelect(e.target.value)}
        >
          {indices?.map((index) => (
            <MenuItem key={index.symbol} value={index.symbol}>
              {index.name} ({index.symbol})
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
}