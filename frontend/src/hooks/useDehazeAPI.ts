/**
 * React hook managing the full dehazing workflow: upload → processing → results.
 *
 * Encapsulates:
 * - State machine: idle → uploading → processing → done (or error)
 * - Upload progress tracking (0-100%)
 * - Error handling and normalization
 * - Result caching and reset logic
 *
 * Used by HomePage to coordinate UI state with backend API calls.
 * Usage:
 *   const { status, result, error, uploadProgress, processImage, loadDemo, reset } = useDehazeAPI();
 *   if (status === 'uploading') return <UploadBar progress={uploadProgress} />;
 *   if (status === 'done') return <ResultViewer result={result} />;
 *   ...
 */

import { useCallback, useState } from 'react';

import { APIError, dehazeImage, getDemoResult } from '../services/api';
import type { DehazeResult, ProcessingStatus } from '../types';

export function useDehazeAPI() {
  // Current UI state (idle → uploading → processing → done, or error at any point)
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  // Result of last successful processing (null until first success)
  const [result, setResult] = useState<DehazeResult | null>(null);
  // Last error message (human-readable, shown to user)
  const [error, setError] = useState<string | null>(null);
  // Upload progress [0-100%]; transitions to 'processing' when 100%
  const [uploadProgress, setUploadProgress] = useState(0);

  /**
   * Reset all state to idle (e.g., when user clicks "Start Over").
   * Clears result, error, and progress so a new upload can start fresh.
   */
  const reset = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setError(null);
    setUploadProgress(0);
  }, []);

  /**
   * Convert any error (APIError, Error, or unknown) to a user-facing message
   * and transition to 'error' state.
   */
  const fail = useCallback((err: unknown) => {
    const message =
      err instanceof APIError || err instanceof Error
        ? err.message
        : 'Request failed';
    setError(message);
    setStatus('error');
  }, []);

  /**
   * Upload and process a user-selected image.
   *
   * State transitions:
   * 1. Start: uploading (0%)
   * 2. Upload completes (100%) → transition to processing (server-side inference running)
   * 3. Server responds → done + result
   * 4. Error at any step → error + error message
   *
   * Args:
   *   file: File object from input or drop zone
   */
  const processImage = useCallback(
    async (file: File) => {
      // Reset any previous result/error before starting
      setStatus('uploading');
      setError(null);
      setResult(null);
      setUploadProgress(0);

      try {
        // Upload file and track progress
        const data = await dehazeImage(file, (pct) => {
          setUploadProgress(pct);
          // Upload finished; server is now running the expensive FFA-Net inference
          if (pct >= 100) setStatus('processing');
        });
        // Server responded with result
        setResult(data);
        setStatus('done');
      } catch (err) {
        fail(err);
      }
    },
    [fail],
  );

  /**
   * Process the bundled demo image (no upload).
   *
   * State transitions:
   * 1. Start: processing (no upload phase needed)
   * 2. Server responds → done + result
   * 3. Error → error + error message
   *
   * Used by the "Try Demo" button to show the feature without upload friction.
   */
  const loadDemo = useCallback(async () => {
    // No upload phase for demo; go straight to processing
    setStatus('processing');
    setError(null);
    setResult(null);
    setUploadProgress(0);

    try {
      const data = await getDemoResult();
      setResult(data);
      setStatus('done');
    } catch (err) {
      fail(err);
    }
  }, [fail]);

  return { status, result, error, uploadProgress, processImage, loadDemo, reset };
}
