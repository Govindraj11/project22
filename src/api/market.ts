import axios, { AxiosError } from 'axios';
import { MarketIndex } from '../types';
import { APIError } from '../utils/errorHandling';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const fetchMarketData = async (symbol: string): Promise<MarketIndex> => {
  try {
    const response = await api.get(`/market/${symbol}`);
    return response.data;
  } catch (error) {
    if (error instanceof AxiosError) {
      throw new APIError(
        error.response?.data?.detail || 'Failed to fetch market data',
        error.response?.status
      );
    }
    throw new APIError('Failed to fetch market data');
  }
};

export const fetchAvailableIndices = async () => {
  try {
    const response = await api.get('/indices');
    return response.data;
  } catch (error) {
    if (error instanceof AxiosError) {
      throw new APIError(
        error.response?.data?.detail || 'Failed to fetch indices',
        error.response?.status
      );
    }
    throw new APIError('Failed to fetch indices');
  }
};