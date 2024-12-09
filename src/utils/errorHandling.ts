export class APIError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'APIError';
  }
}

export const handleAPIError = (error: unknown): string => {
  if (error instanceof APIError) {
    return `Error: ${error.message}`;
  }
  if (error instanceof Error) {
    return `Unexpected error: ${error.message}`;
  }
  return 'An unknown error occurred';
};