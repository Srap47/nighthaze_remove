// Formatting helpers and metric configuration.

import type { MetricConfig, QualityLevel } from '../types';

/** BRISQUE sentinel: the backend returns -1 when the score is unavailable. */
export const BRISQUE_UNAVAILABLE = -1;

/** "12.1s" for >= 1000ms, otherwise "913ms". */
export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

/** Fixed-decimal formatting with thousands separators (e.g. "12,345.68"). */
export function formatMetric(value: number, decimals = 2): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Bucket a metric value into good / fair / poor, honouring `lowerIsBetter`. */
export function getQualityLevel(value: number, config: MetricConfig): QualityLevel {
  const { good, fair } = config.thresholds;

  if (config.lowerIsBetter) {
    if (value < good) return 'good';
    if (value < fair) return 'fair';
    return 'poor';
  }

  if (value > good) return 'good';
  if (value > fair) return 'fair';
  return 'poor';
}

/** The five metrics surfaced in the UI. */
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
