// Typed Axios API client for the NightHaze backend.

import axios from 'axios';
import type { AxiosProgressEvent } from 'axios';
import type { DehazeResult, HealthResponse } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Error carrying the HTTP status alongside a user-facing message. */
export class APIError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

const client = axios.create({ baseURL: BASE_URL });

/**
 * Normalise any thrown error into an APIError.
 *
 * Message precedence: response.data.detail → response.data.error →
 * the axios error message (useful for timeouts/network failures) → "Request failed".
 */
function toAPIError(error: unknown): APIError {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status ?? 0;
    const data = error.response?.data as
      | { detail?: string | null; error?: string }
      | undefined;
    const message =
      data?.detail || data?.error || error.message || 'Request failed';
    return new APIError(message, status);
  }
  const message = error instanceof Error ? error.message : 'Request failed';
  return new APIError(message || 'Request failed', 0);
}

/**
 * Upload an image for dehazing.
 * POST /api/v1/dehaze/upload (multipart field "image").
 */
export async function dehazeImage(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<DehazeResult> {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await client.post<DehazeResult>(
      '/api/v1/dehaze/upload',
      formData,
      {
        timeout: 180000, // CPU inference is slow
        onUploadProgress: (event: AxiosProgressEvent) => {
          if (onProgress && event.total) {
            onProgress(Math.round((event.loaded * 100) / event.total));
          }
        },
      },
    );
    return response.data;
  } catch (error) {
    throw toAPIError(error);
  }
}

/**
 * Run the bundled demo image through the pipeline.
 * GET /api/v1/dehaze/demo.
 */
export async function getDemoResult(): Promise<DehazeResult> {
  try {
    const response = await client.get<DehazeResult>('/api/v1/dehaze/demo', {
      timeout: 180000,
    });
    return response.data;
  } catch (error) {
    throw toAPIError(error);
  }
}

/**
 * Fetch backend health / model-loaded status.
 * GET /api/v1/health.
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await client.get<HealthResponse>('/api/v1/health', {
      timeout: 5000,
    });
    return response.data;
  } catch (error) {
    throw toAPIError(error);
  }
}
