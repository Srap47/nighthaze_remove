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

/** Fixed palette for the pipeline-stage bar segments. */
const STAGE_COLORS = [
  '#6C63FF', // primary
  '#00D4FF', // accent
  '#F59E0B',
  '#10B981',
  '#EC4899',
  '#A78BFA',
];

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/** Animate 0 → target over `duration` ms using requestAnimationFrame. */
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

function MetricCard({ config, value }: MetricCardProps) {
  const unavailable =
    config.key === 'brisque' && value === BRISQUE_UNAVAILABLE;

  // Hook must run unconditionally; feed it 0 when the metric is unavailable.
  const animated = useCountUp(unavailable ? 0 : value);

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
  metrics: DehazeMetrics;
  stages: PipelineStage[];
}

export function MetricsDisplay({ metrics, stages }: MetricsDisplayProps) {
  const totalStageMs =
    stages.reduce((sum, stage) => sum + stage.time_ms, 0) || 1;
  const improved = metrics.colorfulness_improvement_pct > 0;

  return (
    <div className="space-y-6">
      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        {METRIC_CONFIGS.map((config) => (
          <MetricCard
            key={config.key}
            config={config}
            value={metrics[config.key]}
          />
        ))}
      </div>

      {/* Summary lines */}
      <Card className="space-y-3">
        <div className="text-sm text-white/60">
          Processed in{' '}
          <span className="font-semibold text-white">
            {formatMs(metrics.processing_time_ms)}
          </span>
        </div>

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

        {/* Stage timing bar */}
        <div>
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
