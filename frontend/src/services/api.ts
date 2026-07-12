/**
 * Typed Axios API client for the NightHaze backend.
 *
 * Provides typed wrappers around backend endpoints:
 * - POST /dehaze/upload: Upload image for dehazing
 * - GET /dehaze/demo: Process bundled demo image
 * - GET /health: Check model readiness
 *
 * Handles error normalization, timeouts, and upload progress tracking.
 */

import axios from 'axios';
import type { AxiosProgressEvent } from 'axios';
import type { DehazeResult, HealthResponse } from '../types';

/**
 * Backend base URL from environment or default to localhost dev server.
 * Vite injects VITE_API_URL at build time (frontend/.env or frontend/.env.production).
 * Example: http://localhost:8000 (dev) or https://api.example.com (production)
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/**
 * Custom error class combining HTTP status code with user-facing message.
 * Allows callers to distinguish network errors (status=0) from API errors (status=4xx/5xx).
 */
export class APIError extends Error {
  /** HTTP status code; 0 for network errors, timeouts. */
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

/** Axios instance configured with base URL for all requests. */
const client = axios.create({ baseURL: BASE_URL });

/**
 * Normalize any thrown error into an APIError with a user-facing message.
 *
 * Error message precedence (first non-empty wins):
 * 1. response.data.detail (backend-specific error detail)
 * 2. response.data.error (backend error code)
 * 3. error.message (Axios error: network, timeout, etc.)
 * 4. "Request failed" (fallback)
 *
 * Ensures all errors are consistent, facilitating error UI display.
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
 *
 * Sends a multipart form request with the image file to the backend.
 * Tracks upload progress and allows streaming feedback to the UI.
 *
 * Endpoint: POST /api/v1/dehaze/upload
 *
 * Args:
 *   file: File object from HTML input (e.g., from DropZone)
 *   onProgress: Optional callback fired during upload with percent complete [0-100]
 *     Example: file -> 30% uploaded -> callback(30)
 *
 * Returns:
 *   DehazeResult with dehazed image, metrics, and pipeline timing
 *
 * Throws:
 *   APIError with status and user-facing message on failure
 *   - status 400: Unsupported file type
 *   - status 413: File too large (exceeds max_image_size_mb)
 *   - status 503: Model not loaded (weights missing)
 *   - status 500: Pipeline error during processing
 *
 * TWEAK NOTE: timeout (180000ms = 3 min) allows for slow CPU inference (~10-20s for FFA-Net)
 *  plus network/serialization overhead. Increase if running on older hardware.
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
        // TWEAK NOTE: timeout controls how long to wait for response
        // 180000ms = 3 min. CPU-bound FFA-Net inference ~10-20s + overhead.
        timeout: 180000,
        // Track upload progress (before server-side processing starts)
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
 * Process the bundled demo image without uploading.
 *
 * Allows frontend to showcase dehazing without requiring user file upload.
 * Uses a pre-bundled sample nighttime photograph.
 *
 * Endpoint: GET /api/v1/dehaze/demo
 *
 * Returns:
 *   DehazeResult with dehazed demo image, metrics, and pipeline timing
 *
 * Throws:
 *   APIError with status and user-facing message on failure
 *   - status 404: Demo fixture missing (deployment issue)
 *   - status 503: Model not loaded
 *   - status 500: Pipeline error
 *
 * TWEAK NOTE: timeout (180000ms) same as dehazeImage (demo triggers same pipeline)
 */
export async function getDemoResult(): Promise<DehazeResult> {
  try {
    const response = await client.get<DehazeResult>('/api/v1/dehaze/demo', {
      timeout: 180000,  // Same as dehazeImage (same backend processing)
    });
    return response.data;
  } catch (error) {
    throw toAPIError(error);
  }
}

/**
 * Check backend health and model readiness.
 *
 * Frontend polls this periodically to:
 * - Verify backend is reachable
 * - Determine if FFA-Net model weights loaded successfully
 * - Display GPU vs CPU info to user
 *
 * Endpoint: GET /api/v1/health
 *
 * Returns:
 *   HealthResponse with:
 *   - status: "ok" (always true if endpoint reachable)
 *   - model_loaded: true if FFA-Net weights available
 *   - device: "cuda" or "cpu"
 *   - version: app version for tracking deployments
 *
 * Throws:
 *   APIError if backend unreachable (network error, status=0)
 *
 * TWEAK NOTE: timeout (5000ms) is short since this is a light query
 *  and users expect instant feedback on backend readiness
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await client.get<HealthResponse>('/api/v1/health', {
      // Short timeout for quick user feedback on backend availability
      timeout: 5000,
    });
    return response.data;
  } catch (error) {
    throw toAPIError(error);
  }
}
