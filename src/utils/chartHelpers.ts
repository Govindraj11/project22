import { ChartData } from 'chart.js';
import { CHART_COLORS } from '../constants/market';

export const createPriceChartData = (historicalPrices: any[]): ChartData<'line'> => {
  const labels = historicalPrices.map(price => price.date);
  const prices = historicalPrices.map(price => price.close);

  return {
    labels,
    datasets: [
      {
        label: 'Price',
        data: prices,
        borderColor: CHART_COLORS.primary,
        backgroundColor: CHART_COLORS.primary + '40',
        fill: true,
        tension: 0.4,
      },
    ],
  };
};

export const createVolumeChartData = (historicalPrices: any[]): ChartData<'bar'> => {
  const labels = historicalPrices.map(price => price.date);
  const volumes = historicalPrices.map(price => price.volume);

  return {
    labels,
    datasets: [
      {
        label: 'Volume',
        data: volumes,
        backgroundColor: CHART_COLORS.secondary + '80',
        borderColor: CHART_COLORS.secondary,
        borderWidth: 1,
      },
    ],
  };
};