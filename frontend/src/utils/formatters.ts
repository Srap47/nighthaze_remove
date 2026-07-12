/**
 * Formatting helpers and metric configuration for the metrics display.
 *
 * Handles:
 * - Number formatting (milliseconds, decimals, thousands separators)
 * - Quality level classification (good/fair/poor based on thresholds)
 * - Metric configuration (labels, descriptions, visual styling)
 */

import type { MetricConfig, QualityLevel } from '../types';

/** BRISQUE sentinel value: backend returns -1 when computation failed (graceful degradation). */
export const BRISQUE_UNAVAILABLE = -1;

/**
 * Format milliseconds as human-readable time.
 * Converts to seconds if >= 1000ms (e.g., 1234ms → "1.2s", 913ms → "913ms").
 * Used for pipeline stage timing display.
 */
export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

/**
 * Format a metric value with fixed decimals and thousands separators.
 * Example: formatMetric(12345.6789, 2) → "12,345.68"
 * Used for all metric displays in the results panel.
 */
export function formatMetric(value: number, decimals = 2): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Classify a metric value as good/fair/poor based on thresholds.
 * Respects lowerIsBetter flag:
 *   - If lowerIsBetter=true (NIQE, BRISQUE): good when value < good_threshold
 *   - If lowerIsBetter=false (PSNR, SSIM): good when value > good_threshold
 *
 * Used for UI color-coding (green/yellow/red badges) and trend indicators.
 *
 * Args:
 *   value: The metric value to classify
 *   config: MetricConfig with thresholds and direction
 *
 * Returns:
 *   'good' (green), 'fair' (yellow), or 'poor' (red)
 */
export function getQualityLevel(value: number, config: MetricConfig): QualityLevel {
  const { good, fair } = config.thresholds;

  if (config.lowerIsBetter) {
    // Lower is better: NIQE, BRISQUE
    if (value < good) return 'good';
    if (value < fair) return 'fair';
    return 'poor';
  }

  // Higher is better: PSNR, SSIM, Visibility
  if (value > good) return 'good';
  if (value > fair) return 'fair';
  return 'poor';
}

/**
 * Configuration for the five metrics displayed in results.
 * Defines labels, descriptions, thresholds, and sorting for the MetricsDisplay component.
 * TWEAK NOTE: Thresholds control badge colors and quality indicators.
 *   Adjust if metric interpretation changes or different models are used.
 *   Example: if using a different dehazing model, recalibrate expected good/fair values.
 */
export const METRIC_CONFIGS: MetricConfig[] = [
  {
    key: 'psnr',
    label: 'PSNR',
    description: 'Peak Signal-to-Noise Ratio — pixel-level fidelity',
    lowerIsBetter: false,
    thresholds: { good: 30, fair: 20 },
  },
  {
    key: 'ssim',
    label: 'SSIM',
    description: 'Structural Similarity — perceptual structure preservation',
    lowerIsBetter: false,
    thresholds: { good: 0.8, fair: 0.5 },
  },
  {
    key: 'niqe',
    label: 'NIQE',
    description: 'Naturalness Image Quality — lower means more natural',
    lowerIsBetter: true,
    thresholds: { good: 4, fair: 7 },
  },
  {
    key: 'brisque',
    label: 'BRISQUE',
    description: 'Blind Image Quality — lower means better perceptual quality',
    lowerIsBetter: true,
    thresholds: { good: 30, fair: 60 },
  },
  {
    key: 'visibility_score',
    label: 'Visibility',
    description: 'Estimated haze removed from the scene',
    lowerIsBetter: false,
    thresholds: { good: 0.5, fair: 0.3 },
  },
];
