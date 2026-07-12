// Shared type definitions for the NightHaze frontend.
// Mirrors the backend Pydantic schemas (backend/app/models/schemas.py).

export interface DehazeMetrics {
  psnr: number;
  ssim: number;
  niqe: number;
  brisque: number;
  visibility_score: number;
  colorfulness_before: number;
  colorfulness_after: number;
  colorfulness_improvement_pct: number;
  processing_time_ms: number;
}

export interface PipelineStage {
  stage: string;
  time_ms: number;
}

export interface DehazeResult {
  success: boolean;
  job_id: string;
  original_image_b64: string; // "data:image/png;base64,..."
  dehazed_image_b64: string;
  transmission_map_b64: string;
  glow_mask_b64: string;
  metrics: DehazeMetrics;
  pipeline_stages: PipelineStage[];
}

export type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'done' | 'error';
export type QualityLevel = 'good' | 'fair' | 'poor';

export interface MetricConfig {
  key: keyof DehazeMetrics;
  label: string;
  description: string;
  lowerIsBetter: boolean;
  thresholds: { good: number; fair: number };
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  device: string;
  version: string;
}

// Backend uniform error envelope: { success: false, error, detail }.
export interface ErrorEnvelope {
  success: false;
  error: string;
  detail: string | null;
}
