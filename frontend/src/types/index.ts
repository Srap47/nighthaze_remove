/**
 * Shared TypeScript type definitions for the NightHaze frontend.
 * Mirrors the backend Pydantic schemas (backend/app/models/schemas.py).
 * Keep these in sync with backend schemas; type mismatch = runtime bugs.
 */

/** Image quality metrics computed for dehazed output vs. original input. */
export interface DehazeMetrics {
  /** Peak Signal-to-Noise Ratio (dB). Higher = better. ~25+ is good. */
  psnr: number;
  /** Structural Similarity Index [-1, 1]. Higher = better. >0.9 is excellent. */
  ssim: number;
  /** Natural Image Quality Evaluator (simplified). Lower = better. ~0-2 is natural. */
  niqe: number;
  /** Blind Reference Image Spatial Quality Evaluator. Lower = better. -1 = failed. */
  brisque: number;
  /** Haze removal effectiveness [0,1]. Higher = more haze removed. */
  visibility_score: number;
  /** Colorfulness of original hazy input. Higher = more saturated. */
  colorfulness_before: number;
  /** Colorfulness of dehazed output. Higher = more saturated. */
  colorfulness_after: number;
  /** Percent improvement in colorfulness. Positive = color enhancement. */
  colorfulness_improvement_pct: number;
  /** Total end-to-end processing time in milliseconds. */
  processing_time_ms: number;
}

/** Timing information for a single pipeline stage. */
export interface PipelineStage {
  /** Stage name: "preprocessing", "glow_detection", "ffa_net_inference", etc. */
  stage: string;
  /** Time elapsed in this stage (milliseconds). */
  time_ms: number;
}

/** Complete response from POST /dehaze/upload or GET /dehaze/demo endpoints. */
export interface DehazeResult {
  /** Always true on success (errors handled separately with ErrorEnvelope). */
  success: boolean;
  /** Unique ID for this processing job (for tracking/logging). */
  job_id: string;
  /** Original input image (base64-encoded PNG data URI). */
  original_image_b64: string; // "data:image/png;base64,..."
  /** Dehazed output image (base64-encoded PNG data URI). */
  dehazed_image_b64: string;
  /** Transmission map visualization: brighter = clearer, darker = hazier (PNG data URI). */
  transmission_map_b64: string;
  /** Detected light source regions mask: green overlay on bright spots (PNG data URI). */
  glow_mask_b64: string;
  /** Quality metrics comparing original and dehazed. */
  metrics: DehazeMetrics;
  /** Timing breakdown for each of the 6 pipeline stages. */
  pipeline_stages: PipelineStage[];
}

/** UI state machine: which screen/mode the application is in. */
export type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'done' | 'error';
/** Quality level classification for a metric (used for color-coding UI badges). */
export type QualityLevel = 'good' | 'fair' | 'poor';

/** Configuration for displaying a single metric (labels, thresholds, icons). */
export interface MetricConfig {
  /** Metric key from DehazeMetrics (e.g., "psnr", "ssim"). */
  key: keyof DehazeMetrics;
  /** Human-readable metric name (e.g., "PSNR"). */
  label: string;
  /** Tooltip/description (e.g., "Peak Signal-to-Noise Ratio (dB)"). */
  description: string;
  /** True if lower values are better (e.g., NIQE, BRISQUE). False if higher is better (e.g., PSNR, SSIM). */
  lowerIsBetter: boolean;
  /** Thresholds for classifying metric as good/fair/poor. Used for UI badges and trend analysis. */
  thresholds: { good: number; fair: number };
}

/** Response from GET /api/v1/health endpoint (service readiness check). */
export interface HealthResponse {
  /** Always "ok" if endpoint is reachable. */
  status: string;
  /** True if FFA-Net model weights successfully loaded at startup. If false, /dehaze returns 503. */
  model_loaded: boolean;
  /** Identifier of the loaded model (e.g., "FFA-Net (its_train_ffa_3_19)"). */
  model_name: string;
  /** Compute device: "cuda" (GPU) or "cpu". Affects inference speed. */
  device: string;
  /** Application version (e.g., "1.0.0"). For deployment tracking. */
  version: string;
}

/** Uniform error response envelope used by all error endpoints. */
// All errors from the backend follow this structure: { success: false, error, detail }
export interface ErrorEnvelope {
  /** Always false for errors. */
  success: false;
  /** Error code/label: "invalid_image", "model_not_loaded", "validation_error", etc. */
  error: string;
  /** Detailed error message or null. Helps users understand what went wrong. */
  detail: string | null;
}
