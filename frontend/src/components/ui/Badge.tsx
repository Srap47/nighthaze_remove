import type { QualityLevel } from '../../types';

export type BadgeLevel = QualityLevel | 'unavailable';

interface BadgeProps {
  level: BadgeLevel;
  className?: string;
}

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
