export const MARKET_REFRESH_INTERVAL = 30000; // 30 seconds
export const INDICES_REFRESH_INTERVAL = 60000; // 1 minute

export const CHART_COLORS = {
  primary: 'rgb(75, 192, 192)',
  secondary: 'rgb(255, 99, 132)',
  tertiary: 'rgb(153, 102, 255)',
  success: 'rgb(34, 197, 94)',
  danger: 'rgb(239, 68, 68)',
  warning: 'rgb(234, 179, 8)',
};

export const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    intersect: false,
    mode: 'index' as const,
  },
  plugins: {
    legend: {
      position: 'top' as const,
    },
    tooltip: {
      enabled: true,
    },
  },
  scales: {
    y: {
      beginAtZero: false,
    },
  },
};