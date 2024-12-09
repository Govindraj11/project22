import { Bar } from 'react-chartjs-2';
import { Box } from '@mui/material';
import { CHART_OPTIONS } from '../../constants/market';
import { createVolumeChartData } from '../../utils/chartHelpers';

type Props = {
  historicalPrices: any[];
  height?: number;
};

export default function VolumeChart({ historicalPrices, height = 200 }: Props) {
  const data = createVolumeChartData(historicalPrices);
  
  return (
    <Box style={{ height }}>
      <Bar data={data} options={CHART_OPTIONS} />
    </Box>
  );
}