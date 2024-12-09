import { useQuery } from '@tanstack/react-query';
import { fetchMarketData, fetchAvailableIndices } from '../api/market';
import { MarketIndex } from '../types';

export const useMarketData = (symbol: string | null) => {
  return useQuery<MarketIndex>({
    queryKey: ['marketData', symbol],
    queryFn: () => symbol ? fetchMarketData(symbol) : Promise.reject('No symbol provided'),
    enabled: !!symbol,
    staleTime: 30000, // Data considered fresh for 30 seconds
    retry: 2
  });
};

export const useAvailableIndices = () => {
  return useQuery({
    queryKey: ['indices'],
    queryFn: fetchAvailableIndices,
    staleTime: 60000, // Cache indices for 1 minute
    retry: 2
  });
};