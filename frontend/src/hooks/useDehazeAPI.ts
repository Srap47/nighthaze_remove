// Manages the full upload → process → result lifecycle.

import { useCallback, useState } from 'react';

import { APIError, dehazeImage, getDemoResult } from '../services/api';
import type { DehazeResult, ProcessingStatus } from '../types';

export function useDehazeAPI() {
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [result, setResult] = useState<DehazeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const reset = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setError(null);
    setUploadProgress(0);
  }, []);

  const fail = useCallback((err: unknown) => {
    const message =
      err instanceof APIError || err instanceof Error
        ? err.message
        : 'Request failed';
    setError(message);
    setStatus('error');
  }, []);

  const processImage = useCallback(
    async (file: File) => {
      setStatus('uploading');
      setError(null);
      setResult(null);
      setUploadProgress(0);

      try {
        const data = await dehazeImage(file, (pct) => {
          setUploadProgress(pct);
          // Upload finished; the server is now running the (slow) pipeline.
          if (pct >= 100) setStatus('processing');
        });
        setResult(data);
        setStatus('done');
      } catch (err) {
        fail(err);
      }
    },
    [fail],
  );

  const loadDemo = useCallback(async () => {
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
