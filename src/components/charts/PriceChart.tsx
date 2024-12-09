import { Line } from 'react-chartjs-2';
import { Box } from '@mui/material';
import { CHART_OPTIONS } from '../../constants/market';
import { createPriceChartData } from '../../utils/chartHelpers';

type Props = {
  historicalPrices: any[];
  height?: number;
};

export default function PriceChart({ historicalPrices, height = 300 }: Props) {
  const data = createPriceChartData(historicalPrices);
  
  return (
    <Box style={{ height }}>
      <Line data={data} options={CHART_OPTIONS} />
    </Box>
  );
}