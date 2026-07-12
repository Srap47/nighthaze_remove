/**
 * MetricsDisplay — Results dashboard showing quality metrics and stage timings.
 *
 * Part of: Frontend viewer components
 * Used by: HomePage (in "done" state tab, displays results)
 *
 * Shows five metric cards (PSNR, SSIM, NIQE, BRISQUE, visibility) with animated
 * counting up to their final values. Also displays colorfulness before/after,
 * total processing time, and a horizontal stacked bar chart showing relative
 * time spent in each pipeline stage.
 */

import { useEffect, useRef, useState } from 'react';
import { TrendingUp } from 'lucide-react';

import type { DehazeMetrics, MetricConfig, PipelineStage } from '../../types';
import {
  BRISQUE_UNAVAILABLE,
  METRIC_CONFIGS,
  formatMetric,
  formatMs,
  getQualityLevel,
} from '../../utils/formatters';
import { Badge } from '../ui/Badge';
import type { BadgeLevel } from '../ui/Badge';
import { Card } from '../ui/Card';

// TWEAK NOTE: Pipeline stage bar colors
// Each stage segment gets a distinct color. Cycle through the palette if > 6 stages.
// Modify these hex colors to change the stage bar appearance.
const STAGE_COLORS = [
  '#6C63FF', // primary (purple)
  '#00D4FF', // accent (cyan)
  '#F59E0B', // amber
  '#10B981', // emerald
  '#EC4899', // pink
  '#A78BFA', // violet
];

// Easing function for smooth animation: starts slow, speeds up, ends slow
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * useCountUp — Animate a number from 0 to target over duration ms.
 *
 * Uses requestAnimationFrame for smooth 60fps animation and easeOutCubic
 * easing for natural-feeling motion. Used by MetricCard to animate metric
 * values from 0 to their final values when results load.
 */
function useCountUp(target: number, duration = 800): number {
  const [value, setValue] = useState(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    let startTime: number | null = null;

    const step = (timestamp: number) => {
      if (startTime === null) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      setValue(target * easeOutCubic(progress));
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(step);
      }
    };

    frameRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration]);

  return value;
}

interface MetricCardProps {
  config: MetricConfig;
  value: number;
}

/**
 * MetricCard — Single metric display card with animated count-up.
 *
 * Shows the metric name, animated numeric value, and a quality badge
 * (good/fair/poor/unavailable). The metric label has a title tooltip
 * explaining what the metric measures.
 */
function MetricCard({ config, value }: MetricCardProps) {
  // Special case: BRISQUE may not be available on some systems
  const unavailable =
    config.key === 'brisque' && value === BRISQUE_UNAVAILABLE;

  // useCountUp must run unconditionally (React rule). If unavailable, animate 0.
  const animated = useCountUp(unavailable ? 0 : value);

  // Determine quality level badge (good/fair/poor/unavailable)
  const level: BadgeLevel = unavailable
    ? 'unavailable'
    : getQualityLevel(value, config);

  return (
    <Card>
      <div className="flex items-start justify-between gap-2">
        <span
          className="cursor-help text-xs font-medium text-white/50"
          title={config.description}
        >
          {config.label}
        </span>
        <Badge level={level} />
      </div>
      <div className="mt-2 text-2xl font-bold tabular-nums">
        {unavailable ? 'N/A' : formatMetric(animated)}
      </div>
    </Card>
  );
}

interface MetricsDisplayProps {
  metrics: DehazeMetrics;  // Quality metrics from the pipeline
  stages: PipelineStage[]; // Per-stage timing information (always 6 entries)
}

export function MetricsDisplay({ metrics, stages }: MetricsDisplayProps) {
  // Sum all stage times to calculate proportions for the bar chart
  const totalStageMs =
    stages.reduce((sum, stage) => sum + stage.time_ms, 0) || 1;
  // Colorfulness improved if positive percentage
  const improved = metrics.colorfulness_improvement_pct > 0;

  // TWEAK NOTE: Metrics grid layout
  // Responsive grid: 2 columns on mobile, 3 on tablet, 5 on desktop (all metrics visible at once).
  // Adjust gap-4 for spacing, grid-cols-* for column count.
  return (
    <div className="space-y-6">
      {/* Section 1: Five metric cards (PSNR, SSIM, NIQE, BRISQUE, visibility)
          Each card animates its value from 0 to the final metric value. */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        {METRIC_CONFIGS.map((config) => (
          <MetricCard
            key={config.key}
            config={config}
            value={metrics[config.key]}
          />
        ))}
      </div>

      {/* Section 2: Summary card with processing time, colorfulness change, stage timings */}
      <Card className="space-y-3">
        {/* Total processing time line */}
        <div className="text-sm text-white/60">
          Processed in{' '}
          <span className="font-semibold text-white">
            {formatMs(metrics.processing_time_ms)}
          </span>
        </div>

        {/* Colorfulness comparison: before → after, with improvement percentage
            Icon turns green (emerald) if improved, stays muted if not. */}
        <div className="flex items-center gap-2 text-sm">
          <TrendingUp
            className={`h-4 w-4 ${improved ? 'text-emerald-400' : 'text-white/40'}`}
          />
          <span className="text-white/60">Colorfulness</span>
          <span className="font-medium tabular-nums">
            {formatMetric(metrics.colorfulness_before, 1)} →{' '}
            {formatMetric(metrics.colorfulness_after, 1)}
          </span>
          <span
            className={`font-semibold tabular-nums ${
              improved ? 'text-emerald-400' : 'text-white/40'
            }`}
          >
            ({improved ? '+' : ''}
            {formatMetric(metrics.colorfulness_improvement_pct, 1)}%)
          </span>
        </div>

        {/* Stage timing visualization: stacked horizontal bar + legend
            Shows relative time spent in each of the 6 pipeline stages.
            Hovering over a segment shows its name and duration in a tooltip. */}
        <div>
          {/* Stacked bar: each segment width represents percentage of total time */}
          <div className="flex h-3 w-full overflow-hidden rounded-full bg-white/5">
            {stages.map((stage, index) => (
              <div
                key={stage.stage}
                title={`${stage.stage}: ${formatMs(stage.time_ms)}`}
                style={{
                  width: `${(stage.time_ms / totalStageMs) * 100}%`,
                  backgroundColor: STAGE_COLORS[index % STAGE_COLORS.length],
                }}
              />
            ))}
          </div>

          {/* Legend: color dot + stage name + duration for each stage */}
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
            {stages.map((stage, index) => (
              <div
                key={stage.stage}
                className="flex items-center gap-1.5 text-xs text-white/60"
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{
                    backgroundColor: STAGE_COLORS[index % STAGE_COLORS.length],
                  }}
                />
                <span className="capitalize">
                  {stage.stage.replace(/_/g, ' ')}
                </span>
                <span className="tabular-nums text-white/40">
                  {formatMs(stage.time_ms)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
