/**
 * Badge — Status indicator for metric quality levels.
 *
 * Part of: Frontend UI primitives
 * Used by: MetricsDisplay (shows quality of each metric: PSNR, SSIM, NIQE, BRISQUE, visibility)
 *
 * Maps quality levels (good/fair/poor) and unavailable state to color-coded badges.
 * Uses emerald for good, amber for fair, red for poor, and muted white for unavailable.
 */

import type { QualityLevel } from '../../types';

export type BadgeLevel = QualityLevel | 'unavailable';

interface BadgeProps {
  level: BadgeLevel;
  className?: string;
}

// TWEAK NOTE: Badge styling by quality level
// Modify these to change the color scheme. Format: bg-{color}/opacity text-{color} border-{color}/opacity
// - good (emerald): indicates metric meets quality threshold; adjust emerald-400/500 for brightness
// - fair (amber): indicates marginal quality; adjust amber-400/500 for warning intensity
// - poor (red): indicates poor quality; adjust red-400/500 for alarm intensity
// - unavailable (white): metric could not be computed; stays neutral/muted
const STYLES: Record<BadgeLevel, string> = {
  good: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  fair: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  poor: 'bg-red-500/15 text-red-400 border-red-500/30',
  unavailable: 'bg-white/5 text-white/40 border-white/10',
};

export function Badge({ level, className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize ${STYLES[level]} ${className}`}
    >
      {level}
    </span>
  );
}
